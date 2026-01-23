"""
Configuration settings for the Face Recognition Door Unlock System
"""

# ═══════════════════════════════════════════════════════════════════
# IMAGE QUALITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

# Brightness thresholds (0-255 scale)
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

# Blur detection (Laplacian variance threshold)
# Lower values = more blur allowed
BLUR_THRESHOLD = 100.0

# Minimum face area as percentage of frame
MIN_FACE_AREA_PERCENT = 2.0

# Maximum face area as percentage of frame (too close)
MAX_FACE_AREA_PERCENT = 60.0


# ═══════════════════════════════════════════════════════════════════
# FACE DETECTION SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Face detection backend:
#   - "dlib_hog": dlib HOG detector (fast, CPU)
#   - "dlib_cnn": dlib CNN detector (accurate, GPU recommended)
#   - "mediapipe": MediaPipe Face Detection (very fast, good accuracy)
#   - "opencv_dnn": OpenCV DNN Res10 SSD (fast, no extra deps)
#   - "yolo": YOLOv8 face detection (fast, accurate)
FACE_DETECTION_BACKEND = "yolo"

# Minimum detection confidence (0.0 - 1.0)
FACE_DETECTION_CONFIDENCE = 0.5

# Minimum number of landmarks required
MIN_LANDMARKS = 5

# Legacy: Face detection model for dlib backend
FACE_DETECTION_MODEL = "hog"

# Number of times to upsample image for face detection (dlib)
FACE_DETECTION_UPSAMPLES = 1

# ═══════════════════════════════════════════════════════════════════
# FACE RECOGNITION / EMBEDDING SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Face recognition backend:
#   - "dlib": dlib ResNet (128-D, default)
#   - "deepface": DeepFace library (multiple models)
#   - "adaface": AdaFace (512-D, SOTA for low quality)
FACE_RECOGNITION_BACKEND = "dlib"

# DeepFace model (if using deepface backend):
#   - "VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace", etc.
DEEPFACE_MODEL = "ArcFace"

# Number of re-samples for encoding (higher = more accurate but slower)
ENCODING_NUM_JITTERS = 1

# Embedding pre-processing mode:
#   - "none": use full frame + face location
#   - "resize": crop face and resize to target size
#   - "pad_resize": pad face crop to square, then resize
#   - "align": align face using eye landmarks, then resize
EMBEDDING_PREPROCESS_MODE = "resize"

# Enable face alignment (rotate to make eyes horizontal)
# Works with any mode, applied before other preprocessing
ENABLE_FACE_ALIGNMENT = True

# Only use landmarks from detector for alignment (don't fall back to face_recognition)
# Set to True if detector landmarks don't match face_recognition (e.g., YOLO)
ALIGNMENT_REQUIRE_DETECTOR_LANDMARKS = True

# Target size for embedding crop (square)
EMBEDDING_TARGET_SIZE = 160

# Padding around face before resize (fraction of max face dimension)
EMBEDDING_PADDING_RATIO = 0.25

# Padding value for pad_resize (0 = black)
EMBEDDING_PAD_VALUE = 0


# ═══════════════════════════════════════════════════════════════════
# FACE MATCHING SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Distance threshold for face match (lower = stricter)
# Typical range: 0.4 (strict) to 0.6 (lenient)
MATCH_THRESHOLD = 0.5

# Minimum margin between best match and second best match
# Ensures the match is unambiguous
MATCH_MARGIN = 0.1


# ═══════════════════════════════════════════════════════════════════
# ANTI-REPLAY / ANTI-REUSE SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Time window to check for replay attacks (seconds)
REPLAY_TIME_WINDOW = 30.0

# Minimum time between successful unlocks for same person (seconds)
REUSE_COOLDOWN = 5.0

# Perceptual hash similarity threshold (0-64, lower = more similar)
HASH_SIMILARITY_THRESHOLD = 10


# ═══════════════════════════════════════════════════════════════════
# DATABASE SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Path to face database file
DATABASE_PATH = "face_database.pkl"

# Maximum embeddings to store per person
MAX_EMBEDDINGS_PER_PERSON = 10


# ═══════════════════════════════════════════════════════════════════
# WEBCAM SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Camera index (usually 0 for built-in webcam)
CAMERA_INDEX = 0

# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Target FPS
TARGET_FPS = 30

# Consecutive good frames required before processing
CONSECUTIVE_GOOD_FRAMES = 3
