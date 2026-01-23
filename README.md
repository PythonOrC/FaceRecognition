# Face Recognition Door Unlock System

A real-time face recognition system for door access control with anti-spoofing protection and multiple detection/recognition backends.

## Features

- **Multiple Detection Backends**: dlib HOG, dlib CNN, MediaPipe, OpenCV DNN, YOLO
- **Multiple Recognition Backends**: dlib ResNet, DeepFace (ArcFace, Facenet, etc.), AdaFace
- **Real-time Face Detection**: Captures and processes webcam frames in real-time
- **Image Quality Assessment**: Checks lighting, blur, and obstructions before processing
- **Face Alignment**: Optional eye-based alignment for improved accuracy
- **Debug Panel**: Visual debugging with intermediate processing images
- **Threshold-based Matching**: Configurable distance threshold with margin verification
- **Anti-Replay Protection**: Detects photo/video replay attacks
- **Multi-embedding Storage**: Stores multiple embeddings per person for improved accuracy

## System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    WEBCAM CAPTURE LOOP                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   IMAGE QUALITY CHECK  │
                 │  - Brightness          │
                 │  - Blur/Sharpness      │
                 │  - Lighting uniformity │
                 └────────────────────────┘
                              │
                    Quality OK? ──No──▶ Capture Next Frame
                              │
                             Yes
                              ▼
                 ┌────────────────────────┐
                 │   FACE DETECTION       │
                 │  - Backend: dlib/MP/   │
                 │    OpenCV/YOLO         │
                 │  - Find largest face   │
                 │  - Detect landmarks    │
                 └────────────────────────┘
                              │
              Sufficient landmarks? ──No──▶ Capture Next Frame
                              │
                             Yes
                              ▼
                 ┌────────────────────────┐
                 │  PREPROCESSING         │
                 │  - Crop + Pad/Resize   │
                 │  - Face Alignment      │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  EMBEDDING EXTRACTION  │
                 │  - Backend: dlib/      │
                 │    DeepFace/AdaFace    │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   DATABASE SEARCH      │
                 │  - Find best match     │
                 │  - Check threshold     │
                 │  - Verify margin       │
                 └────────────────────────┘
                              │
                 Match >= threshold ──No──▶ Access Denied
                 AND margin OK?
                              │
                             Yes
                              ▼
                 ┌────────────────────────┐
                 │  ANTI-REPLAY CHECK     │
                 │  - Frame hash compare  │
                 │  - Reuse cooldown      │
                 └────────────────────────┘
                              │
                  Checks passed? ──No──▶ Access Denied
                              │
                             Yes
                              ▼
              ┌────────────────────────────────┐
              │  🔓 DOOR UNLOCKED              │
              │  - Add embedding to database   │
              │  - Record unlock timestamp     │
              └────────────────────────────────┘
```

## Installation

### Option 1: Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml
conda activate ML
```

### Option 2: Pip

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### Install Optional Backends

```bash
# MediaPipe (fast detection)
pip install mediapipe

# DeepFace (multiple recognition models)
pip install deepface

# YOLO detection
pip install ultralytics

# AdaFace (requires PyTorch)
pip install torch torchvision
```

### Note on dlib + numpy

⚠️ **dlib requires numpy < 2.0**. If you get "Unsupported image type" errors:
```bash
pip install "numpy<2.0"
```

## Usage

### Basic Operation

```bash
# Run the door unlock system
python main.py

# Use a specific camera
python main.py --camera 1
```

### Controls (when running)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Register new face (interactive) |
| `d` | Toggle debug info overlay |
| `p` | Toggle debug panel (intermediate images) |

### Command Line Options

```bash
# List all registered persons
python main.py --list-persons

# Register a face
python main.py --register "John Doe"

# Remove a person
python main.py --remove "John Doe"
```

## Configuration

Edit `config.py` to customize:

### Detection Backend
```python
# Options: "dlib_hog", "dlib_cnn", "mediapipe", "opencv_dnn", "yolo"
FACE_DETECTION_BACKEND = "dlib_hog"
FACE_DETECTION_CONFIDENCE = 0.5
```

### Recognition Backend
```python
# Options: "dlib", "deepface", "adaface"
FACE_RECOGNITION_BACKEND = "dlib"

# For DeepFace: "ArcFace", "Facenet512", "VGG-Face", etc.
DEEPFACE_MODEL = "ArcFace"
```

### Preprocessing
```python
# Options: "none", "resize", "pad_resize"
EMBEDDING_PREPROCESS_MODE = "pad_resize"
EMBEDDING_TARGET_SIZE = 160

# Face alignment (rotate to make eyes horizontal)
ENABLE_FACE_ALIGNMENT = True
ALIGNMENT_REQUIRE_DETECTOR_LANDMARKS = True
```

### Face Matching
```python
MATCH_THRESHOLD = 0.5     # Max distance for match (lower = stricter)
MATCH_MARGIN = 0.1        # Required margin over second-best match
```

### Image Quality
```python
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220
BLUR_THRESHOLD = 100.0
```

## Debug Panel

Press `p` to open the debug panel showing a 3x3 grid of intermediate images:

| Original | Quality Check | Face Detection |
|----------|---------------|----------------|
| **Face Crop** | **Aligned** | **Padded/Resized** |
| **Landmarks** | **Embedding Input** | **Match Result** |

## Project Structure

```
FaceRecognition/
├── main.py                 # Main application and CLI
├── config.py               # Configuration settings
├── face_processor.py       # Face detection and embedding pipeline
├── detection_backends.py   # Detection backends (dlib, mediapipe, opencv, yolo)
├── recognition_backends.py # Recognition backends (dlib, deepface, adaface)
├── image_quality.py        # Image quality assessment
├── database.py             # Face database management
├── anti_replay.py          # Anti-spoofing protection
├── debug_panel.py          # Debug visualization panel
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
├── face_database.pkl       # Stored face data (runtime)
└── README.md
```

## Backend Comparison

### Detection Backends

| Backend | Speed | Accuracy | Landmarks | Notes |
|---------|-------|----------|-----------|-------|
| `dlib_hog` | Fast | Good | 68 pts | CPU, default |
| `dlib_cnn` | Slow | Better | 68 pts | GPU recommended |
| `mediapipe` | Very Fast | Good | 6 pts | Best for real-time |
| `opencv_dnn` | Fast | Good | None | No extra deps |
| `yolo` | Fast | Excellent | 5 pts* | Needs face model |

*YOLO landmarks require YOLOv8-face model

### Recognition Backends

| Backend | Dimensions | Accuracy | Notes |
|---------|------------|----------|-------|
| `dlib` | 128-D | Good | Default, fast |
| `deepface` | Varies | Excellent | Multiple models |
| `adaface` | 512-D | SOTA | Best for low quality |

## Troubleshooting

### "Unsupported image type" error
```bash
pip install "numpy<2.0"
```

### Camera not detected
- Try different camera index: `--camera 0`, `--camera 1`
- Check no other app is using the camera
- Run `python test_webcam.py` to diagnose

### MediaPipe initialization failed
Model downloads automatically. If it fails:
```bash
# Check internet connection
# Or manually download blaze_face_short_range.tflite
```

### Distorted face in debug panel
- Set `EMBEDDING_PREPROCESS_MODE = "pad_resize"` (not "resize")
- Check `ALIGNMENT_REQUIRE_DETECTOR_LANDMARKS = True` if using YOLO

### Poor recognition accuracy
- Ensure good lighting
- Register multiple embeddings per person
- Try `FACE_RECOGNITION_BACKEND = "deepface"` with `DEEPFACE_MODEL = "ArcFace"`
- Lower `MATCH_THRESHOLD` (e.g., 0.4)

## Security Considerations

⚠️ This is a demonstration system. For production:

- Add hardware liveness detection (IR, depth camera)
- Implement multi-factor authentication
- Use encrypted storage for embeddings
- Add comprehensive audit logging
- Consider edge cases (twins, makeup, aging)

## License

MIT License - Feel free to use and modify for your projects.
