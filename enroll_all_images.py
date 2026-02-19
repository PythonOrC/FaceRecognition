"""
Enroll all face images from one person's folder into a database.

Example:
    conda run -n ML python enroll_all_images.py ^
      --person-dir test_data/clips/benji ^
      --db-path face_database.pkl ^
      --clean
"""

import argparse
import os
from pathlib import Path

import cv2

from database import FaceDatabase
from face_processor import process_frame


def main():
    parser = argparse.ArgumentParser(
        description="Enroll all images in a person folder into a face database."
    )
    parser.add_argument(
        "--person-dir",
        type=Path,
        required=True,
        help="Directory containing one person's reference images.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("face_database.pkl"),
        help="Path to database pickle file.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing database file before enrollment.",
    )
    args = parser.parse_args()

    if not args.person_dir.exists() or not args.person_dir.is_dir():
        raise FileNotFoundError(f"Person directory not found: {args.person_dir}")

    image_paths = sorted(
        [p for p in args.person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.person_dir}")

    if args.clean and args.db_path.exists():
        os.remove(args.db_path)
        print(f"[INFO] Removed existing database: {args.db_path}")

    person_name = args.person_dir.name
    db = FaceDatabase(str(args.db_path))
    if person_name in db.get_all_persons():
        db.remove_person(person_name)
        print(f"[INFO] Removed existing person record: {person_name}")

    success = 0
    failed = 0

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image: {image_path.name}")
            failed += 1
            continue

        detection = process_frame(frame)
        if not detection.success:
            print(f"[WARN] Failed face processing for {image_path.name}: {detection.message}")
            failed += 1
            continue

        db.register_person(person_name, detection.face_data.embedding)
        success += 1

    info = db.get_person_info(person_name)
    stored = len(info.embeddings) if info else 0
    print(
        f"[INFO] Enrollment done for '{person_name}': "
        f"images={len(image_paths)}, success={success}, failed={failed}, stored={stored}"
    )


if __name__ == "__main__":
    main()
