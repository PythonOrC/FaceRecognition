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

# Minimum number of landmarks required
MIN_LANDMARKS = 5

# Face detection model: 'hog' (faster, CPU) or 'cnn' (more accurate, GPU)
FACE_DETECTION_MODEL = 'hog'

# Number of times to upsample image for face detection
FACE_DETECTION_UPSAMPLES = 1

# Number of re-samples for encoding (higher = more accurate but slower)
ENCODING_NUM_JITTERS = 1


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
DATABASE_PATH = 'face_database.pkl'

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
