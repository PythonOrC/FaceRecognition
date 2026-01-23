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
        model=config.FACE_DETECTION_MODEL
    )
    
    return face_locations


def get_face_area(location: Tuple[int, int, int, int]) -> int:
    """Calculate the area of a face bounding box."""
    top, right, bottom, left = location
    return (bottom - top) * (right - left)


def find_largest_face(face_locations: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
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


def detect_landmarks(frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[dict]:
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
        rgb_frame,
        face_locations=[face_location]
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
    required_features = ['left_eye', 'right_eye', 'nose_tip']
    for feature in required_features:
        if feature not in landmarks or not landmarks[feature]:
            return False
    
    total_landmarks = count_landmarks(landmarks)
    return total_landmarks >= config.MIN_LANDMARKS


def extract_embedding(frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract the 128-dimensional face embedding vector.
    
    Args:
        frame: BGR image from OpenCV
        face_location: Face location tuple
        
    Returns:
        128-dimensional numpy array or None if extraction fails
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        encodings = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=[face_location],
            num_jitters=config.ENCODING_NUM_JITTERS
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
    # Step 1: Detect all faces
    face_locations = detect_faces(frame)
    
    if not face_locations:
        return DetectionResult(
            success=False,
            face_data=None,
            all_faces_count=0,
            message="No faces detected"
        )
    
    # Step 2: Find the largest face
    largest_face = find_largest_face(face_locations)
    face_area = get_face_area(largest_face)
    
    # Step 3: Detect landmarks
    landmarks = detect_landmarks(frame, largest_face)
    
    if not has_sufficient_landmarks(landmarks):
        landmark_count = count_landmarks(landmarks) if landmarks else 0
        return DetectionResult(
            success=False,
            face_data=FaceData(
                location=largest_face,
                landmarks=landmarks,
                embedding=None,
                area=face_area,
                confidence=0.0
            ),
            all_faces_count=len(face_locations),
            message=f"Insufficient landmarks ({landmark_count} detected, need {config.MIN_LANDMARKS})"
        )
    
    # Step 4: Extract embedding
    embedding = extract_embedding(frame, largest_face)
    
    if embedding is None:
        return DetectionResult(
            success=False,
            face_data=FaceData(
                location=largest_face,
                landmarks=landmarks,
                embedding=None,
                area=face_area,
                confidence=0.0
            ),
            all_faces_count=len(face_locations),
            message="Failed to extract face embedding"
        )
    
    # Success!
    return DetectionResult(
        success=True,
        face_data=FaceData(
            location=largest_face,
            landmarks=landmarks,
            embedding=embedding,
            area=face_area,
            confidence=1.0
        ),
        all_faces_count=len(face_locations),
        message="Face processed successfully"
    )


def calculate_face_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two face embeddings.
    
    Args:
        embedding1: First 128-dimensional embedding
        embedding2: Second 128-dimensional embedding
        
    Returns:
        Distance value (lower = more similar)
    """
    return np.linalg.norm(embedding1 - embedding2)


def draw_face_box(frame: np.ndarray, face_location: Tuple[int, int, int, int], 
                  name: str = "", color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
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
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame
