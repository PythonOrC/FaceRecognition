"""
Face Recognition/Embedding Backends Module

Provides a unified interface for multiple face embedding backends:
- dlib ResNet (128-D, default)
- DeepFace (multiple models: ArcFace, Facenet, VGG-Face, etc.)
- AdaFace (512-D, SOTA for low quality images)
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass

import config


@dataclass
class EmbeddingResult:
    """Standardized embedding result."""

    embedding: np.ndarray
    dimension: int
    model_name: str


class BaseEmbedder(ABC):
    """Abstract base class for face embedding extractors."""

    @abstractmethod
    def extract(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        landmarks: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from a frame.

        Args:
            frame: BGR image from OpenCV
            face_location: (top, right, bottom, left) tuple
            landmarks: Optional facial landmarks

        Returns:
            Embedding vector or None if extraction fails
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the embedder name."""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class DlibEmbedder(BaseEmbedder):
    """dlib ResNet face embedder (128-D)."""

    def __init__(self):
        import face_recognition

        self._face_recognition = face_recognition
        self._name = "dlib"
        self._dim = 128

    def extract(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        landmarks: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            encodings = self._face_recognition.face_encodings(
                rgb,
                known_face_locations=[face_location],
                num_jitters=config.ENCODING_NUM_JITTERS,
            )

            if encodings:
                return encodings[0]
        except Exception as e:
            print(f"dlib embedding error: {e}")

        return None

    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        return self._dim


class DeepFaceEmbedder(BaseEmbedder):
    """DeepFace embedder (supports multiple models)."""

    def __init__(self, model_name: str = None):
        self._available = False
        self._model_name = model_name or config.DEEPFACE_MODEL

        try:
            from deepface import DeepFace

            self._deepface = DeepFace

            # Model dimensions
            self._model_dims = {
                "VGG-Face": 2622,
                "Facenet": 128,
                "Facenet512": 512,
                "OpenFace": 128,
                "DeepFace": 4096,
                "DeepID": 160,
                "ArcFace": 512,
                "Dlib": 128,
                "SFace": 128,
            }

            self._dim = self._model_dims.get(self._model_name, 512)
            self._available = True
            print(f"DeepFace initialized with model: {self._model_name}")

        except ImportError:
            print("Warning: DeepFace not installed. Run: pip install deepface")

        self._name = f"deepface_{self._model_name}"

    def extract(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        landmarks: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        if not self._available:
            return None

        top, right, bottom, left = face_location

        # Add some padding
        h, w = frame.shape[:2]
        pad = int(max(right - left, bottom - top) * 0.1)
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(h, bottom + pad)
        right = min(w, right + pad)

        # Crop face
        face_img = frame[top:bottom, left:right]

        if face_img.size == 0:
            return None

        try:
            # DeepFace.represent returns embedding
            result = self._deepface.represent(
                face_img,
                model_name=self._model_name,
                enforce_detection=False,
                detector_backend="skip",  # We already detected the face
            )

            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"])
                # Normalize to unit vector so compute_distance uses cosine distance,
                # keeping thresholds comparable across all backends
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding

        except Exception as e:
            print(f"DeepFace embedding error: {e}")

        return None

    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        return self._dim


class AdaFaceEmbedder(BaseEmbedder):
    """AdaFace embedder (512-D, SOTA for low quality images)."""

    def __init__(self):
        self._available = False
        self._name = "adaface"
        self._dim = 512

        try:
            # AdaFace requires specific setup
            # Try to import and initialize
            import torch

            self._torch = torch
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # Try to load AdaFace model
            # This requires the adaface package or manual model loading
            try:
                from adaface import AdaFace

                self.model = AdaFace()
                self.model.to(self._device)
                self.model.eval()
                self._available = True
                print(f"AdaFace initialized on {self._device}")
            except ImportError:
                # Try alternative: load from pretrained weights
                self._load_adaface_manual()

        except ImportError:
            print("Warning: PyTorch not installed for AdaFace")

    def _load_adaface_manual(self):
        """Try to load AdaFace model manually."""
        try:
            import torch
            import torch.nn as nn

            # Check for model file
            model_path = "adaface_ir50_webface4m.ckpt"

            import os

            if os.path.exists(model_path):
                # Load the model (requires proper model definition)
                print(f"Loading AdaFace from {model_path}")
                # This would require the full model architecture
                # For now, mark as unavailable
                print("AdaFace manual loading not implemented")
                self._available = False
            else:
                print(
                    f"AdaFace model not found at {model_path}. "
                    "Download from https://github.com/mk-minchul/AdaFace"
                )
                self._available = False

        except Exception as e:
            print(f"AdaFace initialization failed: {e}")
            self._available = False

    def _preprocess(self, face_img: np.ndarray) -> "torch.Tensor":
        """Preprocess face image for AdaFace."""
        import torch
        from torchvision import transforms

        # Resize to 112x112
        face_img = cv2.resize(face_img, (112, 112))

        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Normalize
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        tensor = transform(face_img).unsqueeze(0)
        return tensor.to(self._device)

    def extract(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        landmarks: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        if not self._available:
            return None

        top, right, bottom, left = face_location

        # Add padding
        h, w = frame.shape[:2]
        pad = int(max(right - left, bottom - top) * 0.2)
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(h, bottom + pad)
        right = min(w, right + pad)

        face_img = frame[top:bottom, left:right]

        if face_img.size == 0:
            return None

        try:
            with self._torch.no_grad():
                tensor = self._preprocess(face_img)
                embedding = self.model(tensor)
                embedding = embedding.cpu().numpy().flatten()

                # Normalize
                embedding = embedding / np.linalg.norm(embedding)

                return embedding

        except Exception as e:
            print(f"AdaFace embedding error: {e}")

        return None

    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════

_embedder_instance: Optional[BaseEmbedder] = None
_embedder_requested_backend: Optional[str] = None


def reset_embedder():
    """Reset the cached embedder instance. Call before switching backends."""
    global _embedder_instance, _embedder_requested_backend
    _embedder_instance = None
    _embedder_requested_backend = None


def get_embedder(backend: str = None) -> BaseEmbedder:
    """
    Get a face embedder instance.

    Args:
        backend: Embedder backend name, or None to use config default

    Returns:
        BaseEmbedder instance
    """
    global _embedder_instance, _embedder_requested_backend

    if backend is None:
        backend = config.FACE_RECOGNITION_BACKEND

    # Return cached instance if same backend was requested
    # (covers both successful init and fallback cases)
    if _embedder_instance is not None and _embedder_requested_backend == backend:
        return _embedder_instance

    embedders = {
        "dlib": DlibEmbedder,
        "deepface": DeepFaceEmbedder,
        "adaface": AdaFaceEmbedder,
    }

    if backend not in embedders:
        print(f"Unknown embedder backend: {backend}, using dlib")
        backend = "dlib"

    _embedder_requested_backend = backend

    try:
        _embedder_instance = embedders[backend]()
        print(f"Initialized face embedder: {_embedder_instance.name()}")
    except Exception as e:
        print(f"Failed to initialize {backend}: {e}")
        print("Falling back to dlib")
        _embedder_instance = DlibEmbedder()

    return _embedder_instance


def compute_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute distance between two embeddings.

    Uses cosine distance for normalized embeddings, Euclidean otherwise.
    """
    # Check if embeddings are normalized (unit vectors)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if abs(norm1 - 1.0) < 0.1 and abs(norm2 - 1.0) < 0.1:
        # Use cosine distance for normalized embeddings
        similarity = np.dot(embedding1, embedding2)
        return 1.0 - similarity
    else:
        # Use Euclidean distance
        return np.linalg.norm(embedding1 - embedding2)
