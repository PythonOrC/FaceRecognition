"""
Face Detection Backends Module

Provides a unified interface for multiple face detection backends:
- dlib HOG (fast, CPU)
- dlib CNN (accurate, GPU)
- MediaPipe (very fast)
- OpenCV DNN (no extra deps)
- YOLO (fast, accurate)
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass

import config


@dataclass
class DetectedFace:
    """Standardized face detection result."""

    # Bounding box in (top, right, bottom, left) format for face_recognition compatibility
    location: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[dict] = None


class BaseDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in a frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of DetectedFace objects
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the detector name."""
        pass


class DlibHOGDetector(BaseDetector):
    """dlib HOG-based face detector."""

    def __init__(self):
        import dlib

        self.detector = dlib.get_frontal_face_detector()
        self._name = "dlib_hog"

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets, scores, _ = self.detector.run(rgb, config.FACE_DETECTION_UPSAMPLES, -1)

        faces = []
        for det, score in zip(dets, scores):
            # Convert to (top, right, bottom, left)
            location = (det.top(), det.right(), det.bottom(), det.left())
            faces.append(DetectedFace(location=location, confidence=float(score)))

        return faces

    def name(self) -> str:
        return self._name


class DlibCNNDetector(BaseDetector):
    """dlib CNN-based face detector (more accurate, needs GPU)."""

    def __init__(self, model_path: str = "mmod_human_face_detector.dat"):
        import dlib

        try:
            self.detector = dlib.cnn_face_detection_model_v1(model_path)
            self._available = True
        except Exception as e:
            print(f"Warning: Could not load dlib CNN model: {e}")
            print("Falling back to HOG detector")
            self.detector = dlib.get_frontal_face_detector()
            self._available = False
        self._name = "dlib_cnn"

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._available:
            dets = self.detector(rgb, config.FACE_DETECTION_UPSAMPLES)
            faces = []
            for det in dets:
                rect = det.rect
                location = (rect.top(), rect.right(), rect.bottom(), rect.left())
                faces.append(
                    DetectedFace(location=location, confidence=float(det.confidence))
                )
        else:
            # Fallback to HOG
            dets = self.detector(rgb, config.FACE_DETECTION_UPSAMPLES)
            faces = [
                DetectedFace(
                    location=(d.top(), d.right(), d.bottom(), d.left()), confidence=1.0
                )
                for d in dets
            ]

        return faces

    def name(self) -> str:
        return self._name


class MediaPipeDetector(BaseDetector):
    """MediaPipe face detector (very fast, good accuracy)."""

    def __init__(self):
        self._available = False
        self._name = "mediapipe"
        
        try:
            import mediapipe as mp
            
            # Try new Tasks API (MediaPipe >= 0.10)
            if hasattr(mp, 'tasks'):
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # Create face detector with Tasks API
                base_options = python.BaseOptions(
                    model_asset_path=self._get_model_path()
                )
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                )
                self.detector = vision.FaceDetector.create_from_options(options)
                self._use_tasks_api = True
                self._available = True
                print("MediaPipe initialized with Tasks API")
                
            # Try legacy solutions API (MediaPipe < 0.10)
            elif hasattr(mp, 'solutions'):
                self.mp_face = mp.solutions.face_detection
                self.detector = self.mp_face.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                )
                self._use_tasks_api = False
                self._available = True
                print("MediaPipe initialized with Solutions API")
            else:
                print("Warning: MediaPipe API not recognized")
                
        except Exception as e:
            print(f"Warning: MediaPipe initialization failed: {e}")
            print("Tip: For Tasks API, you may need the model file.")
            # Try fallback to OpenCV
            self._available = False
    
    def _get_model_path(self) -> str:
        """Get or download the MediaPipe face detection model."""
        import os
        
        model_name = "blaze_face_short_range.tflite"
        
        # Check if model exists locally
        if os.path.exists(model_name):
            return model_name
        
        # Try to download
        try:
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            print(f"Downloading MediaPipe face model...")
            urllib.request.urlretrieve(url, model_name)
            print(f"Downloaded {model_name}")
            return model_name
        except Exception as e:
            print(f"Could not download model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        if not self._available:
            return []

        h, w = frame.shape[:2]
        faces = []
        
        if self._use_tasks_api:
            # Tasks API
            import mediapipe as mp
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            result = self.detector.detect(mp_image)
            
            for detection in result.detections:
                bbox = detection.bounding_box
                
                left = bbox.origin_x
                top = bbox.origin_y
                right = left + bbox.width
                bottom = top + bbox.height
                
                # Clamp to image bounds
                left = max(0, left)
                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)
                
                # Extract keypoints as landmarks (6 keypoints from Face Detection)
                landmarks = None
                if detection.keypoints:
                    kps = detection.keypoints
                    # Keypoints order: right_eye, left_eye, nose_tip, mouth_center, right_ear, left_ear
                    landmarks = {}
                    if len(kps) > 0:
                        landmarks["right_eye"] = [(int(kps[0].x * w), int(kps[0].y * h))]
                    if len(kps) > 1:
                        landmarks["left_eye"] = [(int(kps[1].x * w), int(kps[1].y * h))]
                    if len(kps) > 2:
                        landmarks["nose_tip"] = [(int(kps[2].x * w), int(kps[2].y * h))]
                    if len(kps) > 3:
                        landmarks["mouth_center"] = [(int(kps[3].x * w), int(kps[3].y * h))]
                    if len(kps) > 4:
                        landmarks["right_ear"] = [(int(kps[4].x * w), int(kps[4].y * h))]
                    if len(kps) > 5:
                        landmarks["left_ear"] = [(int(kps[5].x * w), int(kps[5].y * h))]
                
                conf = detection.categories[0].score if detection.categories else 0.5
                
                faces.append(
                    DetectedFace(
                        location=(top, right, bottom, left),
                        confidence=conf,
                        landmarks=landmarks,
                    )
                )
        else:
            # Legacy Solutions API
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    left = int(bbox.xmin * w)
                    top = int(bbox.ymin * h)
                    right = int((bbox.xmin + bbox.width) * w)
                    bottom = int((bbox.ymin + bbox.height) * h)

                    left = max(0, left)
                    top = max(0, top)
                    right = min(w, right)
                    bottom = min(h, bottom)

                    landmarks = None
                    if detection.location_data.relative_keypoints:
                        kps = detection.location_data.relative_keypoints
                        landmarks = {
                            "right_eye": [(int(kps[0].x * w), int(kps[0].y * h))],
                            "left_eye": [(int(kps[1].x * w), int(kps[1].y * h))],
                            "nose_tip": [(int(kps[2].x * w), int(kps[2].y * h))],
                        }
                        if len(kps) > 3:
                            landmarks["mouth_center"] = [(int(kps[3].x * w), int(kps[3].y * h))]
                        if len(kps) > 4:
                            landmarks["right_ear"] = [(int(kps[4].x * w), int(kps[4].y * h))]
                        if len(kps) > 5:
                            landmarks["left_ear"] = [(int(kps[5].x * w), int(kps[5].y * h))]

                    faces.append(
                        DetectedFace(
                            location=(top, right, bottom, left),
                            confidence=detection.score[0],
                            landmarks=landmarks,
                        )
                    )

        return faces

    def name(self) -> str:
        return self._name


class OpenCVDNNDetector(BaseDetector):
    """OpenCV DNN face detector (Res10 SSD, no extra dependencies)."""

    def __init__(self):
        # Try to load the model from OpenCV's data or download
        self._available = False

        # Paths to try for the model files
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"

        try:
            self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            self._available = True
        except Exception:
            # Try with OpenCV's built-in face detector (Haar cascade as fallback)
            try:
                self.cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self._use_cascade = True
                self._available = True
                print("Using Haar cascade fallback for OpenCV detection")
            except Exception as e:
                print(f"Warning: OpenCV DNN detector not available: {e}")
                self._use_cascade = False

        self._name = "opencv_dnn"

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        if not self._available:
            return []

        h, w = frame.shape[:2]

        if hasattr(self, "_use_cascade") and self._use_cascade:
            # Haar cascade fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            faces = []
            for x, y, fw, fh in rects:
                location = (y, x + fw, y + fh, x)  # (top, right, bottom, left)
                faces.append(DetectedFace(location=location, confidence=1.0))
            return faces

        # DNN detector
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > config.FACE_DETECTION_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                left, top, right, bottom = box.astype("int")

                # Clamp to image bounds
                left = max(0, left)
                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)

                faces.append(
                    DetectedFace(
                        location=(top, right, bottom, left), confidence=float(confidence)
                    )
                )

        return faces

    def name(self) -> str:
        return self._name


class YOLODetector(BaseDetector):
    """YOLO face detector (fast, accurate)."""

    def __init__(self):
        self._available = False
        self._face_mode = False
        self._has_keypoints = False

        try:
            from ultralytics import YOLO

            # Try to load YOLOv8 face model
            # You can use yolov8n-face.pt or similar
            try:
                self.model = YOLO("yolov8n-face.pt")
                self._face_mode = True
                self._has_keypoints = True  # Face models usually have keypoints
                print("YOLO face model loaded with keypoints support")
            except Exception:
                # Fallback to general YOLO model (will detect 'person', not ideal)
                print("YOLOv8 face model not found, trying general model...")
                try:
                    self.model = YOLO("yolov8n.pt")
                    self._person_class = 0  # COCO class for person
                    self._face_mode = False
                    self._has_keypoints = False
                    print("Using general YOLO model (no face keypoints)")
                except Exception:
                    raise ImportError("No YOLO model available")

            self._available = True

        except ImportError:
            print("Warning: ultralytics not installed. Run: pip install ultralytics")

        self._name = "yolo"

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        if not self._available:
            return []

        results = self.model(frame, verbose=False)[0]

        faces = []
        
        for i, box in enumerate(results.boxes):
            # For face model, all detections are faces
            # For general model, filter by class
            if not self._face_mode and int(box.cls[0]) != self._person_class:
                continue

            conf = float(box.conf[0])
            if conf < config.FACE_DETECTION_CONFIDENCE:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Convert to (top, right, bottom, left)
            location = (int(y1), int(x2), int(y2), int(x1))

            # Extract keypoints if available (YOLOv8-face provides 5 keypoints)
            landmarks = None
            if self._has_keypoints and results.keypoints is not None:
                try:
                    kpts = results.keypoints[i].data[0].cpu().numpy()
                    # YOLOv8-face keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
                    if len(kpts) >= 5:
                        landmarks = {
                            "left_eye": [(int(kpts[0][0]), int(kpts[0][1]))],
                            "right_eye": [(int(kpts[1][0]), int(kpts[1][1]))],
                            "nose_tip": [(int(kpts[2][0]), int(kpts[2][1]))],
                            "left_mouth": [(int(kpts[3][0]), int(kpts[3][1]))],
                            "right_mouth": [(int(kpts[4][0]), int(kpts[4][1]))],
                        }
                except Exception:
                    pass  # No keypoints available

            faces.append(DetectedFace(location=location, confidence=conf, landmarks=landmarks))

        return faces

    def name(self) -> str:
        return self._name


# ═══════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════

_detector_instance: Optional[BaseDetector] = None


def get_detector(backend: str = None) -> BaseDetector:
    """
    Get a face detector instance.

    Args:
        backend: Detector backend name, or None to use config default

    Returns:
        BaseDetector instance
    """
    global _detector_instance

    if backend is None:
        backend = config.FACE_DETECTION_BACKEND

    # Return cached instance if same backend
    if _detector_instance is not None and _detector_instance.name() == backend:
        return _detector_instance

    detectors = {
        "dlib_hog": DlibHOGDetector,
        "dlib_cnn": DlibCNNDetector,
        "mediapipe": MediaPipeDetector,
        "opencv_dnn": OpenCVDNNDetector,
        "yolo": YOLODetector,
    }

    if backend not in detectors:
        print(f"Unknown detector backend: {backend}, using dlib_hog")
        backend = "dlib_hog"

    try:
        _detector_instance = detectors[backend]()
        print(f"Initialized face detector: {backend}")
    except Exception as e:
        print(f"Failed to initialize {backend}: {e}")
        print("Falling back to dlib_hog")
        _detector_instance = DlibHOGDetector()

    return _detector_instance
