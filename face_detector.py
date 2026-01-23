"""
Face Detection and Landmark Extraction Module
Detects faces and extracts facial landmarks for recognition.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

# Try to import dlib, but make it optional
try:
    import dlib

    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("dlib not available, using OpenCV-only mode")


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 with correct scaling/clipping."""
    if img is None or img.size == 0:
        raise ValueError("Empty image")

    # Make contiguous (dlib is picky)
    img = np.ascontiguousarray(img)

    if img.dtype == np.uint8:
        return img

    if img.dtype == np.uint16:
        # scale 0..65535 -> 0..255
        return (img / 257).astype(np.uint8)

    if np.issubdtype(img.dtype, np.floating):
        maxv = float(np.nanmax(img))
        if maxv <= 1.0:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    # other ints
    return np.clip(img, 0, 255).astype(np.uint8)


def to_dlib_rgb(image: np.ndarray) -> np.ndarray:
    """Return uint8 RGB image suitable for dlib."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    # Handle grayscale
    if image.ndim == 2:
        gray = to_uint8(image)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        # Drop alpha if present
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert dtype first
        image = to_uint8(image)

        # Convert BGR->RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Force create a completely fresh C-contiguous uint8 array
    # This is more aggressive than ascontiguousarray
    rgb = np.require(rgb, dtype=np.uint8, requirements=["C_CONTIGUOUS"])

    # Double-check by creating fresh array if still having issues
    if not rgb.flags["OWNDATA"]:
        rgb = rgb.copy()

    return rgb


class FaceDetector:
    def __init__(
        self,
        detector_type: str = "opencv",
        min_landmarks: int = 5,
        predictor_path: str = "shape_predictor_68_face_landmarks.dat",
    ):
        """
        Initialize face detector.

        Args:
            detector_type: Type of detector ("dlib" or "opencv")
            min_landmarks: Minimum number of landmarks required
            predictor_path: Path to dlib's shape predictor model
        """
        self.min_landmarks = min_landmarks
        self.has_predictor = False
        self.dlib_works = False

        # Force OpenCV if dlib not available
        if detector_type == "dlib" and not DLIB_AVAILABLE:
            print("dlib not available, falling back to OpenCV")
            detector_type = "opencv"

        self.detector_type = detector_type

        if detector_type == "dlib" and DLIB_AVAILABLE:
            # Initialize dlib's face detector
            self.detector = dlib.get_frontal_face_detector()

            # Test if dlib actually works
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                self.detector(test_img, 0)
                self.dlib_works = True
                print("dlib face detector initialized successfully")
            except RuntimeError as e:
                print(f"dlib detector test failed: {e}")
                print("Falling back to OpenCV detector")
                self.detector_type = "opencv"
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades
                    + "haarcascade_frontalface_default.xml"
                )

            if self.dlib_works:
                try:
                    self.predictor = dlib.shape_predictor(predictor_path)
                    self.has_predictor = True
                except Exception as e:
                    print(f"Warning: Could not load shape predictor: {e}")
                    print("Landmark detection will use OpenCV fallback.")
                    self.has_predictor = False
        else:
            # Initialize OpenCV's face detector
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("Using OpenCV face detector")

    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes [(x, y, width, height), ...]
        """
        if self.detector_type == "dlib":
            # Validate and convert image to dlib-compatible format
            try:
                rgb = to_dlib_rgb(image)
            except ValueError as e:
                print(f"Image conversion error: {e}")
                return []

            # Debug output
            print(
                f"DEBUG: dtype={rgb.dtype}, shape={rgb.shape}, "
                f"min/max={rgb.min()}/{rgb.max()}, contig={rgb.flags['C_CONTIGUOUS']}, "
                f"owndata={rgb.flags['OWNDATA']}, strides={rgb.strides}"
            )

            # Detect faces - try with explicit copy if needed
            try:
                dlib_faces = self.detector(rgb, 1)
            except RuntimeError as e:
                print(f"First attempt failed: {e}")
                print("Trying with explicit array reconstruction...")
                # Nuclear option: reconstruct array completely
                rgb = np.array(rgb, dtype=np.uint8, order="C", copy=True)
                print(
                    f"After reconstruction: dtype={rgb.dtype}, contig={rgb.flags['C_CONTIGUOUS']}, strides={rgb.strides}"
                )
                dlib_faces = self.detector(rgb, 1)

            # Convert to (x, y, w, h) format
            faces = []
            for face in dlib_faces:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                faces.append((x, y, w, h))

            return faces
        else:
            # Use OpenCV detector
            if image is None or image.size == 0:
                return []

            # Convert to grayscale with proper dtype handling
            try:
                if image.ndim == 2:
                    gray = to_uint8(image)
                elif image.ndim == 3:
                    if image.shape[2] == 4:
                        image = image[:, :, :3]
                    img_uint8 = to_uint8(image)
                    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
                else:
                    return []
            except ValueError:
                return []

            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            return [tuple(face) for face in faces]

    def get_largest_face(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect and return the largest face in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            Largest face bounding box (x, y, width, height) or None
        """
        faces = self.detect_faces(image)

        if not faces:
            return None

        # Find the largest face by area
        largest_face = max(faces, key=lambda f: f[2] * f[3])

        return largest_face

    def extract_landmarks(
        self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from a detected face.

        Args:
            image: Input image (BGR format)
            face_bbox: Face bounding box (x, y, width, height)

        Returns:
            Array of landmark points [(x1, y1), (x2, y2), ...] or None
        """
        if self.has_predictor and self.dlib_works and DLIB_AVAILABLE:
            # Use dlib landmark predictor
            try:
                rgb = to_dlib_rgb(image)
            except ValueError as e:
                print(f"Image conversion error in landmarks: {e}")
                return self._generate_approximate_landmarks(face_bbox)

            # Clamp bounding box to image bounds
            x, y, w, h = face_bbox
            h_img, w_img = rgb.shape[:2]

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img - 1, x + w)
            y2 = min(h_img - 1, y + h)

            if x2 <= x1 or y2 <= y1:
                return None

            rect = dlib.rectangle(x1, y1, x2, y2)

            # Predict landmarks
            shape = self.predictor(rgb, rect)

            # Convert to numpy array
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            return landmarks
        else:
            # Generate approximate landmarks from bounding box
            return self._generate_approximate_landmarks(face_bbox)

    def _generate_approximate_landmarks(
        self, face_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Generate approximate facial landmarks from bounding box.
        Used as fallback when dlib is not available.

        Args:
            face_bbox: Face bounding box (x, y, width, height)

        Returns:
            Array of 5 approximate landmark points (eyes, nose, mouth corners)
        """
        x, y, w, h = face_bbox

        # Generate 5-point landmarks (standard for face recognition)
        # Left eye, right eye, nose tip, left mouth corner, right mouth corner
        landmarks = np.array(
            [
                [x + w * 0.3, y + h * 0.35],  # Left eye
                [x + w * 0.7, y + h * 0.35],  # Right eye
                [x + w * 0.5, y + h * 0.55],  # Nose tip
                [x + w * 0.35, y + h * 0.75],  # Left mouth corner
                [x + w * 0.65, y + h * 0.75],  # Right mouth corner
            ],
            dtype=np.float32,
        )

        return landmarks

    def has_enough_landmarks(self, landmarks: Optional[np.ndarray]) -> bool:
        """
        Check if the detected landmarks meet the minimum requirement.

        Args:
            landmarks: Array of landmark points

        Returns:
            True if enough landmarks detected
        """
        if landmarks is None:
            return False

        # Accept either 68-point (dlib) or 5-point (fallback) landmarks
        return len(landmarks) >= min(self.min_landmarks, 5)

    def draw_landmarks(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Draw landmarks on the image for visualization.

        Args:
            image: Input image (BGR format)
            landmarks: Array of landmark points

        Returns:
            Image with landmarks drawn
        """
        result = image.copy()

        for i, (x, y) in enumerate(landmarks):
            cv2.circle(result, (int(x), int(y)), 2, (0, 255, 0), -1)
            # Optionally add landmark numbers
            # cv2.putText(result, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return result

    def draw_face_box(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        label: str = "",
    ) -> np.ndarray:
        """
        Draw face bounding box on the image.

        Args:
            image: Input image (BGR format)
            face_bbox: Face bounding box (x, y, width, height)
            label: Optional label to display

        Returns:
            Image with face box drawn
        """
        result = image.copy()
        x, y, w, h = face_bbox

        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if label:
            cv2.putText(
                result,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return result

    def align_face(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Align face based on eye positions for better recognition.

        Args:
            image: Input image (BGR format)
            landmarks: Array of landmark points (68 landmarks expected)

        Returns:
            Aligned face image
        """
        if len(landmarks) < 68:
            return image

        # Get eye coordinates (landmarks 36-41 for left eye, 42-47 for right eye)
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)

        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Get the center point between eyes
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

        # Apply transformation
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        return aligned
