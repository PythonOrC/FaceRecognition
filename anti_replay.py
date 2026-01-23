"""
Anti-Replay and Anti-Reuse Protection Module

Protects against:
- Photo/video replay attacks (showing a picture/video of someone)
- Rapid reuse attacks (same person unlocking multiple times too quickly)
- Image injection attacks
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
import hashlib

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("Warning: imagehash not available, using fallback hash method")

import config


@dataclass
class FrameRecord:
    """Record of a processed frame"""
    timestamp: datetime
    frame_hash: str
    embedding_hash: Optional[str]
    person_name: Optional[str]


@dataclass
class UnlockRecord:
    """Record of a successful unlock"""
    timestamp: datetime
    person_name: str
    confidence: float


@dataclass
class AntiReplayResult:
    """Result of anti-replay check"""
    passed: bool
    is_replay: bool
    is_reuse: bool
    liveness_score: float
    message: str


class AntiReplayChecker:
    """
    Checks for replay and reuse attacks.
    
    Methods:
    - Perceptual image hashing to detect replayed images
    - Temporal analysis to detect video playback
    - Cooldown tracking for reuse prevention
    - Basic liveness detection through frame variance
    """
    
    def __init__(self):
        """Initialize the anti-replay checker."""
        # Recent frames for replay detection
        self.recent_frames: deque[FrameRecord] = deque(maxlen=100)
        
        # Recent successful unlocks per person
        self.recent_unlocks: Dict[str, UnlockRecord] = {}
        
        # Frame variance history for liveness
        self.variance_history: deque[float] = deque(maxlen=30)
        
        # Previous frame for motion detection
        self.prev_frame: Optional[np.ndarray] = None
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """
        Compute perceptual hash of a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Hash string
        """
        if IMAGEHASH_AVAILABLE:
            # Use perceptual hash (more robust to minor changes)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            phash = imagehash.phash(pil_image, hash_size=16)
            return str(phash)
        else:
            # Fallback: simple downsampled hash
            small = cv2.resize(frame, (32, 32))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return hashlib.md5(gray.tobytes()).hexdigest()
    
    def _compute_embedding_hash(self, embedding: np.ndarray) -> str:
        """
        Compute hash of an embedding for quick comparison.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            Hash string
        """
        # Quantize embedding to reduce noise sensitivity
        quantized = (embedding * 100).astype(np.int8)
        return hashlib.md5(quantized.tobytes()).hexdigest()
    
    def _hash_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate hamming distance between two hashes.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Hamming distance
        """
        if IMAGEHASH_AVAILABLE:
            try:
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
                return h1 - h2
            except:
                pass
        
        # Fallback: character comparison
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def _check_replay(self, frame_hash: str) -> Tuple[bool, str]:
        """
        Check if this frame is a replay of a recent frame.
        
        Args:
            frame_hash: Hash of current frame
            
        Returns:
            Tuple of (is_replay, message)
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=config.REPLAY_TIME_WINDOW)
        
        for record in self.recent_frames:
            if record.timestamp < cutoff_time:
                continue
            
            distance = self._hash_distance(frame_hash, record.frame_hash)
            
            if distance < config.HASH_SIMILARITY_THRESHOLD:
                time_diff = (current_time - record.timestamp).total_seconds()
                return True, f"Replay detected (hash distance: {distance}, time diff: {time_diff:.1f}s)"
        
        return False, "No replay detected"
    
    def _check_reuse(self, person_name: str) -> Tuple[bool, str]:
        """
        Check if this person has unlocked too recently.
        
        Args:
            person_name: Name of the matched person
            
        Returns:
            Tuple of (is_reuse, message)
        """
        if person_name not in self.recent_unlocks:
            return False, "First unlock for this person"
        
        last_unlock = self.recent_unlocks[person_name]
        time_since = (datetime.now() - last_unlock.timestamp).total_seconds()
        
        if time_since < config.REUSE_COOLDOWN:
            remaining = config.REUSE_COOLDOWN - time_since
            return True, f"Please wait {remaining:.1f}s before unlocking again"
        
        return False, "Cooldown passed"
    
    def _calculate_motion(self, frame: np.ndarray) -> float:
        """
        Calculate motion between current and previous frame.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Motion score (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate percentage of changed pixels
        motion_score = np.sum(thresh > 0) / thresh.size
        
        self.prev_frame = gray
        return motion_score
    
    def _estimate_liveness(self, frame: np.ndarray) -> float:
        """
        Estimate liveness based on frame characteristics.
        
        Checks for:
        - Natural micro-movements
        - Frame variance over time
        - Screen/print artifacts
        
        Args:
            frame: BGR frame
            
        Returns:
            Liveness score (0-1, higher = more likely live)
        """
        liveness_score = 1.0
        
        # Check for natural motion
        motion = self._calculate_motion(frame)
        self.variance_history.append(motion)
        
        if len(self.variance_history) >= 10:
            avg_motion = np.mean(list(self.variance_history))
            motion_std = np.std(list(self.variance_history))
            
            # Live faces should have some micro-movement variance
            if motion_std < 0.001:
                liveness_score *= 0.5  # Too static
            
            # But not too much motion (could be video)
            if avg_motion > 0.1:
                liveness_score *= 0.8  # High motion could be video
        
        # Check for screen artifacts (Moiré patterns)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # High frequency components might indicate screen
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq_energy = np.mean(np.abs(laplacian))
        
        # Very high frequency might indicate screen display
        if high_freq_energy > 50:
            liveness_score *= 0.9
        
        # Check color distribution (prints often have limited color range)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_std = np.std(hsv[:, :, 1])
        
        if saturation_std < 20:
            liveness_score *= 0.7  # Low saturation variance might indicate print
        
        return min(1.0, max(0.0, liveness_score))
    
    def check(self, frame: np.ndarray, embedding: np.ndarray, 
              person_name: Optional[str] = None) -> AntiReplayResult:
        """
        Perform full anti-replay and anti-reuse check.
        
        Args:
            frame: BGR image from OpenCV
            embedding: Face embedding vector
            person_name: Name of matched person (if any)
            
        Returns:
            AntiReplayResult with check outcome
        """
        frame_hash = self._compute_frame_hash(frame)
        embedding_hash = self._compute_embedding_hash(embedding)
        
        # Check for replay attack
        is_replay, replay_msg = self._check_replay(frame_hash)
        
        if is_replay:
            return AntiReplayResult(
                passed=False,
                is_replay=True,
                is_reuse=False,
                liveness_score=0.0,
                message=replay_msg
            )
        
        # Check for reuse if person identified
        is_reuse = False
        reuse_msg = ""
        if person_name:
            is_reuse, reuse_msg = self._check_reuse(person_name)
            
            if is_reuse:
                return AntiReplayResult(
                    passed=False,
                    is_replay=False,
                    is_reuse=True,
                    liveness_score=0.0,
                    message=reuse_msg
                )
        
        # Estimate liveness
        liveness_score = self._estimate_liveness(frame)
        
        # Store this frame
        self.recent_frames.append(FrameRecord(
            timestamp=datetime.now(),
            frame_hash=frame_hash,
            embedding_hash=embedding_hash,
            person_name=person_name
        ))
        
        # Low liveness might indicate attack
        if liveness_score < 0.3:
            return AntiReplayResult(
                passed=False,
                is_replay=False,
                is_reuse=False,
                liveness_score=liveness_score,
                message=f"Low liveness score ({liveness_score:.2f})"
            )
        
        return AntiReplayResult(
            passed=True,
            is_replay=False,
            is_reuse=False,
            liveness_score=liveness_score,
            message="Anti-replay checks passed"
        )
    
    def record_unlock(self, person_name: str, confidence: float):
        """
        Record a successful unlock for cooldown tracking.
        
        Args:
            person_name: Name of person who unlocked
            confidence: Match confidence
        """
        self.recent_unlocks[person_name] = UnlockRecord(
            timestamp=datetime.now(),
            person_name=person_name,
            confidence=confidence
        )
    
    def reset(self):
        """Reset all tracking state."""
        self.recent_frames.clear()
        self.recent_unlocks.clear()
        self.variance_history.clear()
        self.prev_frame = None
