"""
Face Detection and Embedding Extraction Module

Handles:
- Face detection using multiple backends (dlib, mediapipe, opencv, yolo)
- Landmark detection
- Embedding vector extraction using multiple backends (dlib, deepface, adaface)
- Finding the largest face in frame
"""

import cv2
import numpy as np
import face_recognition
from dataclasses import dataclass
from typing import Optional, Tuple, List

import config
from debug_panel import get_debug_panel, debug_enabled
from detection_backends import get_detector, DetectedFace
from recognition_backends import get_embedder


@dataclass
class FaceData:
    """Container for detected face data"""

    location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    landmarks: Optional[dict]
    embedding: Optional[np.ndarray]
    area: int
    confidence: float


@dataclass
class DetectionResult:
    """Result of face detection and processing"""

    success: bool
    face_data: Optional[FaceData]
    all_faces_count: int
    message: str


def detect_faces(
    frame: np.ndarray,
) -> Tuple[List[Tuple[int, int, int, int]], List[Optional[dict]]]:
    """
    Detect all faces in the frame using configured backend.

    Args:
        frame: BGR image from OpenCV

    Returns:
        Tuple of (face_locations, landmarks_list)
        - face_locations: List of (top, right, bottom, left) tuples
        - landmarks_list: List of landmark dicts (may be None for some backends)
    """
    detector = get_detector()
    detected_faces = detector.detect(frame)

    face_locations = []
    landmarks_list = []

    for face in detected_faces:
        face_locations.append(face.location)
        landmarks_list.append(face.landmarks)

    return face_locations, landmarks_list


def get_face_area(location: Tuple[int, int, int, int]) -> int:
    """Calculate the area of a face bounding box."""
    top, right, bottom, left = location
    return (bottom - top) * (right - left)


def find_largest_face(
    face_locations: List[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the largest face from a list of face locations.

    Args:
        face_locations: List of face locations

    Returns:
        Location of the largest face, or None if no faces
    """
    if not face_locations:
        return None

    largest_face = max(face_locations, key=get_face_area)
    return largest_face


def detect_landmarks(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    predetected_landmarks: Optional[dict] = None,
) -> Optional[dict]:
    """
    Detect facial landmarks for a specific face.

    Args:
        frame: BGR image from OpenCV
        face_location: Face location tuple
        predetected_landmarks: Optional landmarks from detection backend

    Returns:
        Dictionary of facial landmarks or None if detection fails
    """
    # Use pre-detected landmarks if available and sufficient
    if predetected_landmarks and len(predetected_landmarks) >= 3:
        return predetected_landmarks

    # Fall back to face_recognition for full landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    landmarks_list = face_recognition.face_landmarks(
        rgb_frame, face_locations=[face_location]
    )

    if landmarks_list:
        return landmarks_list[0]
    return None


def count_landmarks(landmarks: dict) -> int:
    """Count total number of landmark points."""
    if not landmarks:
        return 0
    return sum(len(points) for points in landmarks.values())


def has_sufficient_landmarks(landmarks: Optional[dict]) -> bool:
    """
    Check if we have enough landmarks for reliable embedding.

    Args:
        landmarks: Dictionary of facial landmarks

    Returns:
        True if sufficient landmarks detected
    """
    if not landmarks:
        return False

    # Check for essential facial features
    required_features = ["left_eye", "right_eye", "nose_tip"]
    for feature in required_features:
        if feature not in landmarks or not landmarks[feature]:
            return False

    total_landmarks = count_landmarks(landmarks)
    return total_landmarks >= config.MIN_LANDMARKS


def _crop_face_with_padding(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    padding_ratio: float,
) -> np.ndarray:
    """
    Crop a face region with optional padding.

    Args:
        frame: BGR image from OpenCV
        face_location: (top, right, bottom, left)
        padding_ratio: Padding as a fraction of max(face width, height)

    Returns:
        Cropped BGR face region
    """
    top, right, bottom, left = face_location
    h, w = frame.shape[:2]

    face_w = right - left
    face_h = bottom - top
    pad = int(max(face_w, face_h) * padding_ratio)

    x1 = max(0, left - pad)
    y1 = max(0, top - pad)
    x2 = min(w, right + pad)
    y2 = min(h, bottom + pad)

    return frame[y1:y2, x1:x2]


def _pad_to_square(image: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    Pad an image to a square shape, preserving aspect ratio.
    """
    h, w = image.shape[:2]
    size = max(h, w)

    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )


def _get_eye_centers(
    landmarks: dict,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Get the center points of left and right eyes from landmarks.

    Note: Returns eyes in IMAGE coordinates (left = smaller x, right = larger x)
    regardless of which eye is which from the person's perspective.

    Returns:
        (left_eye_center, right_eye_center) in image coordinates, or (None, None) if not found
    """
    if not landmarks:
        return None, None

    left_eye_pts = landmarks.get("left_eye", [])
    right_eye_pts = landmarks.get("right_eye", [])

    if not left_eye_pts or not right_eye_pts:
        return None, None

    # Calculate center of each eye
    eye1_center = (
        sum(p[0] for p in left_eye_pts) / len(left_eye_pts),
        sum(p[1] for p in left_eye_pts) / len(left_eye_pts),
    )
    eye2_center = (
        sum(p[0] for p in right_eye_pts) / len(right_eye_pts),
        sum(p[1] for p in right_eye_pts) / len(right_eye_pts),
    )

    # Ensure left eye (in image) has smaller x coordinate
    # This handles different naming conventions between backends
    # (MediaPipe uses person's perspective, dlib uses viewer's perspective)
    if eye1_center[0] < eye2_center[0]:
        left_center = eye1_center
        right_center = eye2_center
    else:
        left_center = eye2_center
        right_center = eye1_center

    return left_center, right_center


def _align_face(
    image: np.ndarray,
    landmarks: dict,
    face_location: Tuple[int, int, int, int],
    output_size: int = 160,
) -> Optional[np.ndarray]:
    """
    Align face by rotating to make eyes horizontal and cropping.

    Args:
        image: BGR image
        landmarks: Face landmarks dict
        face_location: (top, right, bottom, left)
        output_size: Output square image size

    Returns:
        Aligned face image or None if alignment fails
    """
    left_eye, right_eye = _get_eye_centers(landmarks)

    if left_eye is None or right_eye is None:
        return None

    # Calculate angle between eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Sanity check: angle should be small for normal faces
    # If angle is extreme, skip alignment (face might be sideways or upside down)
    if abs(angle) > 45:
        # Don't try to align extremely rotated faces
        return None

    # Calculate center point between eyes
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2,
    )

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

    # Rotate the image
    h, w = image.shape[:2]
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    # Now crop the face from rotated image
    # Transform face location corners through the rotation
    top, right, bottom, left = face_location

    # Add padding for the crop
    face_w = right - left
    face_h = bottom - top
    pad = int(max(face_w, face_h) * 0.25)

    # Calculate new crop bounds (using rotated eye center as reference)
    cx, cy = eye_center

    # Estimate face bounds relative to eye center
    crop_w = int(face_w * 1.5)
    crop_h = int(face_h * 1.5)

    x1 = max(0, int(cx - crop_w // 2))
    y1 = max(0, int(cy - crop_h // 3))  # Eyes are typically in upper third
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    # Crop
    aligned = rotated[y1:y2, x1:x2]

    if aligned.size == 0:
        return None

    # Resize to output size
    aligned = cv2.resize(
        aligned, (output_size, output_size), interpolation=cv2.INTER_LINEAR
    )

    return aligned


def _prepare_face_for_embedding(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    landmarks: Optional[dict] = None,
    landmarks_from_detector: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Optionally pad/resize the face crop based on config settings.

    Args:
        frame: BGR image
        face_location: (top, right, bottom, left)
        landmarks: Facial landmarks dict
        landmarks_from_detector: True if landmarks came from detection backend

    Returns:
        (bgr_image, face_location_for_crop)
    """
    mode = config.EMBEDDING_PREPROCESS_MODE
    debug = get_debug_panel()
    target = config.EMBEDDING_TARGET_SIZE

    # Try alignment if enabled and landmarks available
    aligned_face = None
    alignment_angle = 0.0

    # Check if we should attempt alignment
    should_align = getattr(config, "ENABLE_FACE_ALIGNMENT", False) and landmarks

    # If config requires detector landmarks, skip if landmarks are from fallback
    if should_align and getattr(
        config, "ALIGNMENT_REQUIRE_DETECTOR_LANDMARKS", False
    ):
        if not landmarks_from_detector:
            should_align = False
            if debug_enabled():
                debug.set_face_aligned(
                    np.zeros((100, 100, 3), dtype=np.uint8), 0.0
                )  # Show empty for skipped alignment

    if should_align:
        left_eye, right_eye = _get_eye_centers(landmarks)
        if left_eye and right_eye:
            # Calculate angle
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            alignment_angle = np.degrees(np.arctan2(dy, dx))

            aligned_face = _align_face(frame, landmarks, face_location, target)

            if aligned_face is not None and debug_enabled():
                debug.set_face_aligned(aligned_face.copy(), alignment_angle)

    # If alignment succeeded and mode allows, use aligned face
    if aligned_face is not None:
        crop_location = (0, target, target, 0)

        if debug_enabled():
            debug.set_face_crop(aligned_face.copy(), "pre-aligned")
            debug.set_face_padded(aligned_face.copy(), "aligned")
            debug.set_embedding_input(aligned_face.copy(), target)

        return aligned_face, crop_location

    if mode == "none":
        if debug_enabled():
            # Capture cropped region for debug
            top, right, bottom, left = face_location
            crop = frame[top:bottom, left:right].copy()
            debug.set_face_crop(crop, "mode: none")
            debug.set_face_padded(crop, "none")
            debug.set_embedding_input(crop, 0)
        return frame, face_location

    # Crop face with padding first
    face_crop = _crop_face_with_padding(
        frame,
        face_location,
        config.EMBEDDING_PADDING_RATIO,
    )

    if face_crop.size == 0:
        return frame, face_location

    # Debug: capture initial crop
    if debug_enabled():
        debug.set_face_crop(
            face_crop.copy(), f"pad_ratio: {config.EMBEDDING_PADDING_RATIO}"
        )

    if mode == "pad_resize":
        face_crop = _pad_to_square(
            face_crop, pad_value=config.EMBEDDING_PAD_VALUE
        )
        # Debug: capture after padding
        if debug_enabled():
            debug.set_face_padded(face_crop.copy(), "pad_resize")

    # Resize to target
    target = config.EMBEDDING_TARGET_SIZE
    face_crop = cv2.resize(
        face_crop, (target, target), interpolation=cv2.INTER_LINEAR
    )

    # Debug: capture final embedding input
    if debug_enabled():
        debug.set_embedding_input(face_crop.copy(), target)

    # Full crop covers the face
    crop_location = (0, target, target, 0)
    return face_crop, crop_location


def extract_embedding(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    landmarks: Optional[dict] = None,
    landmarks_from_detector: bool = False,
) -> Optional[np.ndarray]:
    """
    Extract face embedding vector using configured backend.

    All backends go through the same preprocessing pipeline
    (alignment, cropping, padding, resizing) for consistent behavior.

    Args:
        frame: BGR image from OpenCV
        face_location: Face location tuple
        landmarks: Optional face landmarks for alignment
        landmarks_from_detector: True if landmarks came from detection backend

    Returns:
        Embedding numpy array or None if extraction fails
    """
    backend = config.FACE_RECOGNITION_BACKEND

    # Always use preprocessing pipeline for all backends
    bgr_frame, embed_location = _prepare_face_for_embedding(
        frame, face_location, landmarks, landmarks_from_detector
    )

    if backend == "dlib":
        # face_recognition expects RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        try:
            encodings = face_recognition.face_encodings(
                rgb_frame,
                known_face_locations=[embed_location],
                num_jitters=config.ENCODING_NUM_JITTERS,
            )

            if encodings:
                return encodings[0]
        except Exception as e:
            print(f"dlib embedding error: {e}")

        return None
    else:
        # Use modular embedder backend with preprocessed BGR frame
        embedder = get_embedder()
        return embedder.extract(bgr_frame, embed_location, landmarks)


def process_frame(frame: np.ndarray) -> DetectionResult:
    """
    Complete face processing pipeline for a single frame.

    1. Detect all faces
    2. Find the largest face
    3. Check landmarks
    4. Extract embedding if landmarks sufficient

    Args:
        frame: BGR image from OpenCV

    Returns:
        DetectionResult with face data if successful
    """
    debug = get_debug_panel()

    # Debug: capture original frame
    if debug_enabled():
        debug.set_original(frame)

    # Step 1: Detect all faces (returns locations and pre-detected landmarks)
    face_locations, predetected_landmarks = detect_faces(frame)

    if not face_locations:
        # Debug: show no detection
        if debug_enabled():
            debug.set_face_detection(frame, [])
        return DetectionResult(
            success=False,
            face_data=None,
            all_faces_count=0,
            message="No faces detected",
        )

    # Step 2: Find the largest face
    largest_face = find_largest_face(face_locations)
    face_area = get_face_area(largest_face)

    # Get index and pre-detected landmarks for largest face
    largest_idx = (
        face_locations.index(largest_face)
        if largest_face in face_locations
        else 0
    )
    largest_predetected_landmarks = (
        predetected_landmarks[largest_idx]
        if predetected_landmarks and largest_idx < len(predetected_landmarks)
        else None
    )

    # Debug: capture face detection result
    if debug_enabled():
        debug.set_face_detection(frame, face_locations, largest_idx)

    # Step 3: Detect landmarks (use pre-detected if available, otherwise use face_recognition)
    landmarks = detect_landmarks(
        frame, largest_face, largest_predetected_landmarks
    )

    # Track if landmarks came from detector (for alignment decision)
    landmarks_from_detector = (
        largest_predetected_landmarks is not None
        and len(largest_predetected_landmarks) >= 3
    )

    # Debug: capture landmarks
    if debug_enabled():
        debug.set_landmarks(frame, largest_face, landmarks)

    if not has_sufficient_landmarks(landmarks):
        landmark_count = count_landmarks(landmarks) if landmarks else 0
        return DetectionResult(
            success=False,
            face_data=FaceData(
                location=largest_face,
                landmarks=landmarks,
                embedding=None,
                area=face_area,
                confidence=0.0,
            ),
            all_faces_count=len(face_locations),
            message=f"Insufficient landmarks ({landmark_count} detected, need {config.MIN_LANDMARKS})",
        )

    # Step 4: Extract embedding (debug images captured inside _prepare_face_for_embedding)
    embedding = extract_embedding(
        frame, largest_face, landmarks, landmarks_from_detector
    )

    if embedding is None:
        return DetectionResult(
            success=False,
            face_data=FaceData(
                location=largest_face,
                landmarks=landmarks,
                embedding=None,
                area=face_area,
                confidence=0.0,
            ),
            all_faces_count=len(face_locations),
            message="Failed to extract face embedding",
        )

    # Success!
    return DetectionResult(
        success=True,
        face_data=FaceData(
            location=largest_face,
            landmarks=landmarks,
            embedding=embedding,
            area=face_area,
            confidence=1.0,
        ),
        all_faces_count=len(face_locations),
        message="Face processed successfully",
    )


def calculate_face_distance(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> float:
    """
    Calculate Euclidean distance between two face embeddings.

    Args:
        embedding1: First 128-dimensional embedding
        embedding2: Second 128-dimensional embedding

    Returns:
        Distance value (lower = more similar)
    """
    return np.linalg.norm(embedding1 - embedding2)


def draw_face_box(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    name: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw a bounding box around a face with optional name label.

    Args:
        frame: BGR image from OpenCV
        face_location: (top, right, bottom, left) tuple
        name: Optional name to display
        color: BGR color tuple

    Returns:
        Frame with drawn annotations
    """
    top, right, bottom, left = face_location

    # Draw box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw name label if provided
    if name:
        cv2.rectangle(
            frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED
        )
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return frame
