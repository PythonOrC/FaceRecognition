"""
Image Quality Assessment Module

Checks for:
- Proper lighting (not too dark/bright)
- Image blur/sharpness
- Face obstructions
- Frame integrity
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import config


@dataclass
class QualityResult:
    """Result of image quality assessment"""
    is_acceptable: bool
    brightness_ok: bool
    sharpness_ok: bool
    brightness_value: float
    sharpness_value: float
    issues: list[str]


def calculate_brightness(frame: np.ndarray) -> float:
    """
    Calculate average brightness of the frame.
    
    Args:
        frame: BGR image from OpenCV
        
    Returns:
        Average brightness value (0-255)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return np.mean(gray)


def calculate_sharpness(frame: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher values indicate sharper images.
    
    Args:
        frame: BGR image from OpenCV
        
    Returns:
        Sharpness score (Laplacian variance)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def check_local_brightness_uniformity(frame: np.ndarray, grid_size: int = 3) -> Tuple[bool, float]:
    """
    Check if lighting is relatively uniform across the frame.
    Detects harsh shadows or partial occlusions.
    
    Args:
        frame: BGR image from OpenCV
        grid_size: Number of grid divisions
        
    Returns:
        Tuple of (is_uniform, variance_score)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    
    cell_means = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_means.append(np.mean(cell))
    
    variance = np.var(cell_means)
    # High variance indicates uneven lighting
    is_uniform = variance < 2000  # Threshold for acceptable variance
    
    return is_uniform, variance


def detect_obstruction(frame: np.ndarray, face_region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, str]:
    """
    Detect potential obstructions in the face region.
    Uses edge density and color uniformity analysis.
    
    Args:
        frame: BGR image from OpenCV
        face_region: Optional (x, y, w, h) tuple for face bounding box
        
    Returns:
        Tuple of (has_obstruction, obstruction_type)
    """
    if face_region:
        x, y, w, h = face_region
        roi = frame[y:y+h, x:x+w]
    else:
        # Analyze center region if no face specified
        h, w = frame.shape[:2]
        center_y, center_x = h // 4, w // 4
        roi = frame[center_y:center_y*3, center_x:center_x*3]
    
    if roi.size == 0:
        return False, "none"
    
    # Check for large uniform color regions (potential mask/paper)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Check saturation - very low saturation might indicate grayscale obstruction
    avg_saturation = np.mean(hsv[:, :, 1])
    
    # Check for dominant single color (potential solid obstruction)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_ratio = np.max(hist) / np.sum(hist)
    
    if dominant_ratio > 0.5 and avg_saturation < 30:
        return True, "potential_flat_surface"
    
    return False, "none"


def assess_quality(frame: np.ndarray) -> QualityResult:
    """
    Perform comprehensive quality assessment on a frame.
    
    Args:
        frame: BGR image from OpenCV
        
    Returns:
        QualityResult with detailed assessment
    """
    issues = []
    
    # Check brightness
    brightness = calculate_brightness(frame)
    brightness_ok = config.MIN_BRIGHTNESS <= brightness <= config.MAX_BRIGHTNESS
    
    if brightness < config.MIN_BRIGHTNESS:
        issues.append(f"Too dark (brightness: {brightness:.1f})")
    elif brightness > config.MAX_BRIGHTNESS:
        issues.append(f"Too bright (brightness: {brightness:.1f})")
    
    # Check sharpness
    sharpness = calculate_sharpness(frame)
    sharpness_ok = sharpness >= config.BLUR_THRESHOLD
    
    if not sharpness_ok:
        issues.append(f"Image too blurry (sharpness: {sharpness:.1f})")
    
    # Check lighting uniformity
    uniform_ok, variance = check_local_brightness_uniformity(frame)
    if not uniform_ok:
        issues.append(f"Uneven lighting (variance: {variance:.1f})")
    
    # Overall assessment
    is_acceptable = brightness_ok and sharpness_ok and uniform_ok
    
    return QualityResult(
        is_acceptable=is_acceptable,
        brightness_ok=brightness_ok,
        sharpness_ok=sharpness_ok,
        brightness_value=brightness,
        sharpness_value=sharpness,
        issues=issues
    )


def check_face_quality(frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Tuple[bool, list[str]]:
    """
    Check quality specifically in the face region.
    
    Args:
        frame: BGR image from OpenCV
        face_location: (top, right, bottom, left) face location from face_recognition
        
    Returns:
        Tuple of (is_acceptable, list_of_issues)
    """
    issues = []
    
    top, right, bottom, left = face_location
    face_roi = frame[top:bottom, left:right]
    
    if face_roi.size == 0:
        return False, ["Invalid face region"]
    
    # Check face region brightness
    face_brightness = calculate_brightness(face_roi)
    if face_brightness < config.MIN_BRIGHTNESS:
        issues.append("Face region too dark")
    elif face_brightness > config.MAX_BRIGHTNESS:
        issues.append("Face region overexposed")
    
    # Check face region sharpness
    face_sharpness = calculate_sharpness(face_roi)
    if face_sharpness < config.BLUR_THRESHOLD * 0.8:  # Slightly more lenient for face
        issues.append("Face region blurry")
    
    # Check for potential obstruction
    has_obstruction, obstruction_type = detect_obstruction(
        frame, 
        (left, top, right - left, bottom - top)
    )
    if has_obstruction:
        issues.append(f"Potential obstruction detected: {obstruction_type}")
    
    # Check face size relative to frame
    frame_area = frame.shape[0] * frame.shape[1]
    face_area = (bottom - top) * (right - left)
    face_percent = (face_area / frame_area) * 100
    
    if face_percent < config.MIN_FACE_AREA_PERCENT:
        issues.append(f"Face too small ({face_percent:.1f}% of frame)")
    elif face_percent > config.MAX_FACE_AREA_PERCENT:
        issues.append(f"Face too close ({face_percent:.1f}% of frame)")
    
    return len(issues) == 0, issues
