"""
Generate annotated demo videos for face recognition on prerecorded clips.

Example:
    conda run -n ML python video_demo.py ^
      --person-dir test_data/clips/benji ^
      --clips test_data/clips/clip1.mp4 test_data/clips/clip2.mp4
"""

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2

import config
from anti_replay import AntiReplayChecker
from database import FaceDatabase
from face_processor import process_frame, draw_face_box
from image_quality import assess_quality, check_face_quality


def overlay_text(frame, text, color=(255, 255, 255), y=30):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


@dataclass
class PipelineState:
    consecutive_good_frames: int = 0
    anti_replay: AntiReplayChecker = field(default_factory=AntiReplayChecker)


@dataclass
class PipelineResult:
    unlocked: bool
    detected: bool
    status: str
    person_name: Optional[str]
    location: Optional[Tuple[int, int, int, int]]
    distance: Optional[float]


def build_gallery_db(person_dir: Path, reset: bool = False) -> tuple[FaceDatabase, str, int]:
    if not person_dir.exists() or not person_dir.is_dir():
        raise FileNotFoundError(f"Person directory not found: {person_dir}")

    person_name = person_dir.name
    image_paths = sorted(
        [p for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise RuntimeError(f"No images found in: {person_dir}")

    db_path = Path("test_data") / "clips" / "demo_outputs" / ".gallery_tmp.pkl"
    if reset and db_path.exists():
        db_path.unlink()
    db = FaceDatabase(db_path=str(db_path))

    existing_person = db.get_person_info(person_name)
    if existing_person is not None and existing_person.embeddings:
        existing_count = len(existing_person.embeddings)
        print(
            f"[INFO] Reusing gallery for '{person_name}' with {existing_count} existing embeddings."
        )
        return db, person_name, existing_count

    success = 0
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        detection = process_frame(frame)
        if not detection.success:
            print(f"[WARN] Failed face processing for {image_path.name}: {detection.message}")
            continue

        db.register_person(person_name, detection.face_data.embedding)
        success += 1

    if success == 0:
        raise RuntimeError(
            f"Failed to build gallery from {person_dir}; no valid embeddings extracted."
        )

    print(f"[INFO] Gallery built for '{person_name}' with {success}/{len(image_paths)} images.")
    return db, person_name, success


def run_main_pipeline_on_frame(
    frame, db: FaceDatabase, state: PipelineState
) -> PipelineResult:
    """Mirror main.py process_single_frame pipeline for a video frame."""
    quality = assess_quality(frame)
    if not quality.is_acceptable:
        state.consecutive_good_frames = 0
        return PipelineResult(
            unlocked=False,
            detected=False,
            status=f"Quality issue: {', '.join(quality.issues)}",
            person_name=None,
            location=None,
            distance=None,
        )

    state.consecutive_good_frames += 1
    if state.consecutive_good_frames < config.CONSECUTIVE_GOOD_FRAMES:
        return PipelineResult(
            unlocked=False,
            detected=False,
            status=(
                f"Stabilizing... "
                f"({state.consecutive_good_frames}/{config.CONSECUTIVE_GOOD_FRAMES})"
            ),
            person_name=None,
            location=None,
            distance=None,
        )

    detection = process_frame(frame)
    if not detection.success:
        return PipelineResult(
            unlocked=False,
            detected=False,
            status=detection.message,
            person_name=None,
            location=None,
            distance=None,
        )

    face_ok, face_issues = check_face_quality(frame, detection.face_data.location)
    if not face_ok:
        return PipelineResult(
            unlocked=False,
            detected=True,
            status=f"Face quality: {', '.join(face_issues)}",
            person_name=None,
            location=detection.face_data.location,
            distance=None,
        )

    embedding = detection.face_data.embedding
    match = db.search(embedding)

    if not match.found:
        return PipelineResult(
            unlocked=False,
            detected=True,
            status=match.message,
            person_name=match.person_name,
            location=detection.face_data.location,
            distance=match.distance,
        )

    if match.distance > config.MATCH_THRESHOLD:
        return PipelineResult(
            unlocked=False,
            detected=True,
            status=f"Match below threshold ({match.distance:.3f})",
            person_name=match.person_name,
            location=detection.face_data.location,
            distance=match.distance,
        )

    if match.margin is not None and match.margin < config.MATCH_MARGIN:
        return PipelineResult(
            unlocked=False,
            detected=True,
            status=f"Match margin too small ({match.margin:.3f})",
            person_name=match.person_name,
            location=detection.face_data.location,
            distance=match.distance,
        )

    anti_result = state.anti_replay.check(frame, embedding, match.person_name)
    if not anti_result.passed:
        return PipelineResult(
            unlocked=False,
            detected=True,
            status=f"Security check failed: {anti_result.message}",
            person_name=match.person_name,
            location=detection.face_data.location,
            distance=match.distance,
        )

    db.add_embedding_to_matched_person(match.person_name, embedding)
    state.anti_replay.record_unlock(match.person_name, match.confidence)

    return PipelineResult(
        unlocked=True,
        detected=True,
        status=f"ACCESS GRANTED: {match.person_name}",
        person_name=match.person_name,
        location=detection.face_data.location,
        distance=match.distance,
    )


def process_clip(
    clip_path: Path,
    output_path: Path,
    matched_frames_dir: Path,
    db: FaceDatabase,
    person_name: str,
):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip: {clip_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    if matched_frames_dir.exists():
        shutil.rmtree(matched_frames_dir)
    matched_frames_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    detected_frames = 0
    matched_frames = 0
    state = PipelineState()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1
        annotated = frame.copy()
        result = run_main_pipeline_on_frame(frame, db, state)

        if result.detected:
            detected_frames += 1

        if result.location is not None:
            if result.unlocked and result.person_name == person_name:
                matched_frames += 1
                label = f"{result.person_name}  d={result.distance:.3f}"
                draw_face_box(annotated, result.location, label, (0, 255, 0))
                overlay_text(annotated, "MATCH", (0, 255, 0), y=30)
                matched_path = matched_frames_dir / f"{clip_path.stem}_frame_{total_frames:06d}.jpg"
                cv2.imwrite(str(matched_path), annotated)
            else:
                if result.distance is not None:
                    label = f"Unknown  d={result.distance:.3f}"
                else:
                    label = "Unknown"
                draw_face_box(annotated, result.location, label, (0, 0, 255))
                overlay_text(annotated, "NO MATCH", (0, 0, 255), y=30)

        if not result.unlocked:
            overlay_text(annotated, result.status, (0, 165, 255), y=90)

        overlay_text(
            annotated,
            f"Frames: {total_frames}  Detected: {detected_frames}  Matched: {matched_frames}",
            (255, 255, 255),
            y=60,
        )

        writer.write(annotated)

    cap.release()
    writer.release()

    print(
        f"[INFO] {clip_path.name}: total={total_frames}, detected={detected_frames}, matched={matched_frames}"
    )
    print(f"[INFO] Saved matched frames to: {matched_frames_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create annotated face-recognition demo videos from clips."
    )
    parser.add_argument(
        "--person-dir",
        type=Path,
        required=True,
        help="Folder with person reference images (folder name is used as identity label).",
    )
    parser.add_argument(
        "--clips",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input clip paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data") / "clips" / "demo_outputs",
        help="Directory where annotated videos are saved.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=15.0,
        help=(
            "Override sharpness threshold for this demo run "
            "(Laplacian variance; lower allows blurrier frames)."
        ),
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=config.FACE_DETECTION_CONFIDENCE,
        help="Override face detection confidence threshold for this demo run.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=config.MATCH_THRESHOLD,
        help="Override verification (distance) threshold for this demo run.",
    )
    parser.add_argument(
        "--reset-gallery",
        action="store_true",
        help="Delete the temporary gallery DB and rebuild from person-dir images.",
    )
    args = parser.parse_args()

    config.BLUR_THRESHOLD = args.blur_threshold
    config.FACE_DETECTION_CONFIDENCE = args.detection_threshold
    config.MATCH_THRESHOLD = args.match_threshold
    print(f"[INFO] Using demo blur threshold: {config.BLUR_THRESHOLD}")
    print(
        f"[INFO] Using demo detection threshold: {config.FACE_DETECTION_CONFIDENCE}"
    )
    print(f"[INFO] Using demo match threshold: {config.MATCH_THRESHOLD}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    db, person_name, _ = build_gallery_db(args.person_dir, reset=args.reset_gallery)

    for clip_path in args.clips:
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")
        output_path = args.output_dir / f"{clip_path.stem}_demo.mp4"
        matched_frames_dir = args.output_dir / f"{clip_path.stem}_matched_frames"
        process_clip(clip_path, output_path, matched_frames_dir, db, person_name)
        print(f"[INFO] Wrote: {output_path}")


if __name__ == "__main__":
    main()
