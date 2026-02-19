# Demo Threshold Snapshot

CLI overrides supported in `video_demo.py`:

- `--detection-threshold`
- `--match-threshold`
- `--blur-threshold`
- `--reset-gallery`

## Clip1

Thresholds used:

- `detection-threshold = 0.45`
- `match-threshold = 0.42`
- `blur-threshold = 15.0`
- `reset-gallery = true`

Command:

```powershell
conda run -n ML python video_demo.py --person-dir test_data/clips/benji --clips test_data/clips/clip1.mp4 --reset-gallery --detection-threshold 0.45 --match-threshold 0.42 --blur-threshold 15.0
```

Reference run result:

- `clip1.mp4: total=9361, detected=833, matched=9`

## Clip2

Latest thresholds used:

- `detection-threshold = 0.50`
- `match-threshold = 0.32`
- `blur-threshold = 15.0`
- `reset-gallery = true`

Command:

```powershell
conda run -n ML python video_demo.py --person-dir test_data/clips/benji --clips test_data/clips/clip2.mp4 --detection-threshold 0.50 --match-threshold 0.32 --blur-threshold 15.0 --reset-gallery
```

Latest run result:

- `clip2.mp4: total=6486, detected=1061, matched=17`
