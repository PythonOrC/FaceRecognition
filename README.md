# Face Recognition Door Unlock System

A real-time face recognition system for door access control with anti-spoofing protection.

## Features

- **Real-time Face Detection**: Captures and processes webcam frames in real-time
- **Image Quality Assessment**: Checks lighting, blur, and obstructions before processing
- **Face Embedding Extraction**: Uses 128-dimensional face embeddings for matching
- **Threshold-based Matching**: Configurable distance threshold with margin verification
- **Anti-Replay Protection**: Detects photo/video replay attacks using perceptual hashing
- **Anti-Reuse Protection**: Cooldown period between successive unlocks
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
                 │  - Find largest face   │
                 │  - Detect landmarks    │
                 └────────────────────────┘
                              │
              Sufficient landmarks? ──No──▶ Capture Next Frame
                              │
                             Yes
                              ▼
                 ┌────────────────────────┐
                 │  EMBEDDING EXTRACTION  │
                 │  - 128-dim vector      │
                 └────────────────────────┘
                              │
                Extraction OK? ──No──▶ Capture Next Frame
                              │
                             Yes
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
                 │  - Liveness estimation │
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

### Prerequisites

- Python 3.10 or higher
- Webcam
- Windows/Linux/macOS

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Note on dlib Installation

The `face-recognition` library depends on `dlib`, which may require additional setup:

**Windows:**
- Install Visual Studio Build Tools with C++ support
- Or install pre-built wheel: `pip install dlib`

**Linux:**
```bash
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
```

**macOS:**
```bash
brew install cmake
```

## Usage

### Basic Operation

```bash
# Run the door unlock system
python main.py

# Use a specific camera (default is 0)
python main.py --camera 1
```

### Controls (when running)

- `q` - Quit the application
- `r` - Register a new face (interactive)
- `d` - Toggle debug display

### Command Line Options

```bash
# List all registered persons
python main.py --list-persons

# Register a face from command line
python main.py --register "John Doe"

# Remove a person from database
python main.py --remove "John Doe"
```

## Configuration

Edit `config.py` to customize thresholds and behavior:

### Image Quality
```python
MIN_BRIGHTNESS = 40       # Minimum acceptable brightness (0-255)
MAX_BRIGHTNESS = 220      # Maximum acceptable brightness
BLUR_THRESHOLD = 100.0    # Laplacian variance threshold for blur detection
```

### Face Matching
```python
MATCH_THRESHOLD = 0.5     # Maximum distance for a match (lower = stricter)
MATCH_MARGIN = 0.1        # Required margin between best and second-best match
```

### Anti-Replay Protection
```python
REPLAY_TIME_WINDOW = 30.0          # Seconds to check for replay attacks
REUSE_COOLDOWN = 5.0               # Minimum seconds between unlocks for same person
HASH_SIMILARITY_THRESHOLD = 10     # Perceptual hash distance threshold
```

## Project Structure

```
FaceRecognition/
├── main.py              # Main application and CLI
├── config.py            # Configuration settings
├── image_quality.py     # Image quality assessment
├── face_processor.py    # Face detection and embedding
├── database.py          # Face database management
├── anti_replay.py       # Anti-spoofing protection
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── face_database.pkl    # Stored face data (created at runtime)
```

## Security Considerations

This system implements several security measures:

1. **Quality Gating**: Poor quality images are rejected to prevent low-confidence matches
2. **Threshold + Margin**: Matches must exceed threshold AND have sufficient margin over alternatives
3. **Perceptual Hashing**: Detects replayed photos/videos
4. **Liveness Estimation**: Basic checks for screen displays and printed photos
5. **Cooldown Period**: Prevents rapid-fire unlock attempts

### Limitations

⚠️ This is a demonstration system. For production security applications:

- Add hardware liveness detection (IR, depth camera)
- Implement multi-factor authentication
- Use encrypted storage for embeddings
- Add audit logging
- Consider edge cases (twins, makeup, aging)

## Troubleshooting

### Camera not detected
- Check camera index with `--camera 0`, `--camera 1`, etc.
- Ensure no other application is using the camera

### Poor recognition accuracy
- Ensure good lighting conditions
- Register multiple embeddings per person
- Adjust `MATCH_THRESHOLD` in config

### Slow performance
- Use `FACE_DETECTION_MODEL = 'hog'` (faster, CPU)
- Reduce `ENCODING_NUM_JITTERS` to 1
- Lower camera resolution in config

## License

MIT License - Feel free to use and modify for your projects.
