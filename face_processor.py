"""
Face Detection and Embedding Extraction Module

Handles:
- Face detection using face_recognition library
- Landmark detection
- Embedding vector extraction
- Finding the largest face in frame
"""

import cv2
import numpy as np
import face_recognition
from dataclasses import dataclass
from typing import Optional, Tuple, List

import config
from debug_panel import get_debug_panel, debug_enabled


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


def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect all faces in the frame.

    Args:
        frame: BGR image from OpenCV

    Returns:
        List of face locations as (top, right, bottom, left) tuples
    """
    # Convert BGR to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(
        rgb_frame,
        number_of_times_to_upsample=config.FACE_DETECTION_UPSAMPLES,
        model=config.FACE_DETECTION_MODEL,
    )

    return face_locations


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
    frame: np.ndarray, face_location: Tuple[int, int, int, int]
) -> Optional[dict]:
    """
    Detect facial landmarks for a specific face.

    Args:
        frame: BGR image from OpenCV
        face_location: Face location tuple

    Returns:
        Dictionary of facial landmarks or None if detection fails
    """
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

    Returns:
        (left_eye_center, right_eye_center) or (None, None) if not found
    """
    if not landmarks:
        return None, None

    left_eye_pts = landmarks.get("left_eye", [])
    right_eye_pts = landmarks.get("right_eye", [])

    if not left_eye_pts or not right_eye_pts:
        return None, None

    # Calculate center of each eye
    left_center = (
        sum(p[0] for p in left_eye_pts) / len(left_eye_pts),
        sum(p[1] for p in left_eye_pts) / len(left_eye_pts),
    )
    right_center = (
        sum(p[0] for p in right_eye_pts) / len(right_eye_pts),
        sum(p[1] for p in right_eye_pts) / len(right_eye_pts),
    )

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
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Optionally pad/resize the face crop based on config settings.

    Returns:
        (rgb_image, face_location_for_crop)
    """
    mode = config.EMBEDDING_PREPROCESS_MODE
    debug = get_debug_panel()
    target = config.EMBEDDING_TARGET_SIZE

    # Try alignment if enabled and landmarks available
    aligned_face = None
    alignment_angle = 0.0

    if getattr(config, "ENABLE_FACE_ALIGNMENT", False) and landmarks:
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
        rgb_crop = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        crop_location = (0, target, target, 0)

        if debug_enabled():
            debug.set_face_crop(aligned_face.copy(), "pre-aligned")
            debug.set_face_padded(aligned_face.copy(), "aligned")
            debug.set_embedding_input(aligned_face.copy(), target)

        return rgb_crop, crop_location

    if mode == "none":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if debug_enabled():
            # Capture cropped region for debug
            top, right, bottom, left = face_location
            crop = frame[top:bottom, left:right].copy()
            debug.set_face_crop(crop, "mode: none")
            debug.set_face_padded(crop, "none")
            debug.set_embedding_input(crop, 0)
        return rgb_frame, face_location

    # Crop face with padding first
    face_crop = _crop_face_with_padding(
        frame,
        face_location,
        config.EMBEDDING_PADDING_RATIO,
    )

    if face_crop.size == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, face_location

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

    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    # Full crop covers the face
    crop_location = (0, target, target, 0)
    return rgb_crop, crop_location


def extract_embedding(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    landmarks: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Extract the 128-dimensional face embedding vector.

    Args:
        frame: BGR image from OpenCV
        face_location: Face location tuple
        landmarks: Optional face landmarks for alignment

    Returns:
        128-dimensional numpy array or None if extraction fails
    """
    rgb_frame, embed_location = _prepare_face_for_embedding(
        frame, face_location, landmarks
    )

    try:
        encodings = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=[embed_location],
            num_jitters=config.ENCODING_NUM_JITTERS,
        )

        if encodings:
            return encodings[0]
    except Exception as e:
        print(f"Embedding extraction error: {e}")

    return None


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

    # Step 1: Detect all faces
    face_locations = detect_faces(frame)

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

    # Debug: capture face detection result
    if debug_enabled():
        largest_idx = (
            face_locations.index(largest_face)
            if largest_face in face_locations
            else 0
        )
        debug.set_face_detection(frame, face_locations, largest_idx)

    # Step 3: Detect landmarks
    landmarks = detect_landmarks(frame, largest_face)

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
    embedding = extract_embedding(frame, largest_face, landmarks)

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
