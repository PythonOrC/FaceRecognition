"""
Ablation Study Framework for Face Recognition Pipeline

Systematically evaluates different configurations of:
  - Detection backends  (dlib_hog, dlib_cnn, mediapipe, opencv_dnn, yolo)
  - Recognition backends (dlib, deepface w/ various models, adaface)
  - Preprocessing modes  (none, resize, pad_resize)
  - Face alignment       (on / off)
  - Match threshold      (sweep)

Usage:
  1. Organize a test dataset (see --create-template):
       test_data/
         gallery/          # Reference images for enrollment
           person_A/
             01.jpg
             02.jpg
           person_B/
             01.jpg
         probes/           # Images to test recognition against
           person_A/       # Genuine probes (person IS in gallery)
             test_01.jpg
           person_B/
             test_01.jpg
           impostor/       # Negative probes (person NOT in gallery)
             unknown_01.jpg

  2. Run the study:
       python ablation_study.py --dataset test_data
       python ablation_study.py --dataset test_data --output results.csv

  3. Run with a custom config grid:
       python ablation_study.py --dataset test_data --config ablation_config.json
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from detection_backends import get_detector, reset_detector
from recognition_backends import (
    get_embedder,
    reset_embedder,
    compute_distance,
)
from face_processor import (
    detect_faces,
    find_largest_face,
    detect_landmarks,
    extract_embedding,
    has_sufficient_landmarks,
)


# ═══════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset(
    dataset_path: Path,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    Load gallery and probe images from the dataset directory.

    Returns:
        (gallery, probes) – each mapping person_name -> [image_paths]
        Probes with folder name "impostor" or "_impostor" are treated
        as negative (unknown) identities.
    """
    gallery_dir = dataset_path / "gallery"
    probes_dir = dataset_path / "probes"

    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")
    if not probes_dir.is_dir():
        raise FileNotFoundError(f"Probes directory not found: {probes_dir}")

    def _scan(root: Path) -> Dict[str, List[Path]]:
        result = {}
        for person_dir in sorted(root.iterdir()):
            if not person_dir.is_dir():
                continue
            imgs = sorted(
                p
                for p in person_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if imgs:
                result[person_dir.name] = imgs
        return result

    return _scan(gallery_dir), _scan(probes_dir)


def create_dataset_template(dataset_path: Path):
    """Create an empty dataset directory structure with a README."""
    dirs = [
        dataset_path / "gallery" / "person_A",
        dataset_path / "gallery" / "person_B",
        dataset_path / "probes" / "person_A",
        dataset_path / "probes" / "person_B",
        dataset_path / "probes" / "impostor",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    readme = dataset_path / "README.txt"
    readme.write_text(
        "Ablation Study Dataset\n"
        "======================\n\n"
        "gallery/<person_name>/  – Reference images used for enrollment.\n"
        "                         Add 1-5 images per person.\n\n"
        "probes/<person_name>/   – Test images. Folder name must match a\n"
        "                         gallery person for genuine probes.\n\n"
        "probes/impostor/        – Images of people NOT in the gallery.\n"
        "                         Used to measure false-accept rate.\n"
    )
    print(f"Created dataset template at: {dataset_path}")
    print("Add images and re-run the study.")


# ═══════════════════════════════════════════════════════════════════
# Per-image processing (reuses the existing pipeline)
# ═══════════════════════════════════════════════════════════════════


def _extract_from_image(
    img_path: Path,
) -> Tuple[Optional[np.ndarray], float, float, bool]:
    """
    Detect the largest face and extract its embedding.

    Returns:
        (embedding_or_None, detection_ms, embedding_ms, face_detected)
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None, 0.0, 0.0, False

    # Detection
    t0 = time.perf_counter()
    face_locations, predetected_landmarks = detect_faces(frame)
    det_ms = (time.perf_counter() - t0) * 1000

    if not face_locations:
        return None, det_ms, 0.0, False

    # Largest face
    largest_face = find_largest_face(face_locations)
    idx = (
        face_locations.index(largest_face) if largest_face in face_locations else 0
    )
    pre_lm = (
        predetected_landmarks[idx]
        if predetected_landmarks and idx < len(predetected_landmarks)
        else None
    )

    # Landmarks
    landmarks = detect_landmarks(frame, largest_face, pre_lm)
    from_det = pre_lm is not None and len(pre_lm) >= 3

    # Embedding
    t1 = time.perf_counter()
    embedding = extract_embedding(frame, largest_face, landmarks, from_det)
    emb_ms = (time.perf_counter() - t1) * 1000

    return embedding, det_ms, emb_ms, True


# ═══════════════════════════════════════════════════════════════════
# Single-configuration evaluation
# ═══════════════════════════════════════════════════════════════════


def _apply_config(cfg: dict):
    """Write a configuration dict into the runtime config module."""
    config.FACE_DETECTION_BACKEND = cfg["detection_backend"]
    config.FACE_RECOGNITION_BACKEND = cfg["recognition_backend"]
    config.EMBEDDING_PREPROCESS_MODE = cfg["preprocess_mode"]
    config.ENABLE_FACE_ALIGNMENT = cfg["alignment"]

    # Optional: DeepFace model override
    if "deepface_model" in cfg:
        config.DEEPFACE_MODEL = cfg["deepface_model"]

    # Force re-creation of cached singletons
    reset_detector()
    reset_embedder()


def evaluate_configuration(
    cfg: dict,
    gallery: Dict[str, List[Path]],
    probes: Dict[str, List[Path]],
    thresholds: List[float],
) -> List[dict]:
    """
    Evaluate one (detector + embedder + preprocess) configuration
    across a sweep of match thresholds.

    Returns one result dict per threshold value.
    """
    _apply_config(cfg)

    # ── Enroll gallery ──────────────────────────────────────────
    gallery_embeddings: Dict[str, List[np.ndarray]] = {}
    gallery_det_times: List[float] = []
    gallery_emb_times: List[float] = []
    gallery_detected = 0
    gallery_total = 0

    for person_name, images in gallery.items():
        embs = []
        for img_path in images:
            gallery_total += 1
            emb, dt, et, detected = _extract_from_image(img_path)
            gallery_det_times.append(dt)
            gallery_emb_times.append(et)
            if detected:
                gallery_detected += 1
            if emb is not None:
                embs.append(emb)
        if embs:
            gallery_embeddings[person_name] = embs

    if not gallery_embeddings:
        print("    WARNING: no gallery embeddings extracted, skipping.")
        return []

    # ── Extract probe embeddings ────────────────────────────────
    probe_records = []  # (true_label, embedding, det_ms, emb_ms)
    probe_det_times: List[float] = []
    probe_emb_times: List[float] = []
    probes_total = 0
    probes_detected = 0
    probes_embedded = 0

    for person_name, images in probes.items():
        is_impostor = person_name.lower() in ("impostor", "_impostor")
        label = None if is_impostor else person_name

        for img_path in images:
            probes_total += 1
            emb, dt, et, detected = _extract_from_image(img_path)
            probe_det_times.append(dt)
            probe_emb_times.append(et)
            if detected:
                probes_detected += 1
            if emb is not None:
                probes_embedded += 1
                probe_records.append((label, emb, dt, et))

    # ── Pre-compute distances to gallery ────────────────────────
    # For each probe, find best match person + distance
    probe_matches = []  # (true_label, best_person, best_distance)

    for label, emb, _, _ in probe_records:
        best_dist = float("inf")
        best_person = None
        for gal_name, gal_embs in gallery_embeddings.items():
            for gal_emb in gal_embs:
                d = compute_distance(emb, gal_emb)
                if d < best_dist:
                    best_dist = d
                    best_person = gal_name
        probe_matches.append((label, best_person, best_dist))

    # ── Evaluate at each threshold ──────────────────────────────
    results = []
    base_info = {
        "detection_backend": cfg["detection_backend"],
        "recognition_backend": cfg["recognition_backend"],
        "preprocess_mode": cfg["preprocess_mode"],
        "alignment": cfg["alignment"],
        "deepface_model": cfg.get("deepface_model", ""),
        "gallery_persons": len(gallery_embeddings),
        "gallery_images": gallery_total,
        "gallery_detection_rate": gallery_detected / max(gallery_total, 1),
        "probe_images": probes_total,
        "probe_detection_rate": probes_detected / max(probes_total, 1),
        "probe_embedding_rate": probes_embedded / max(probes_total, 1),
        "avg_detection_ms": float(np.mean(gallery_det_times + probe_det_times))
        if (gallery_det_times or probe_det_times)
        else 0,
        "avg_embedding_ms": float(np.mean(gallery_emb_times + probe_emb_times))
        if (gallery_emb_times or probe_emb_times)
        else 0,
    }

    for thresh in thresholds:
        tp = fp = fn = tn = 0

        for true_label, best_person, best_dist in probe_matches:
            matched = best_dist <= thresh
            is_impostor = true_label is None
            is_genuine = not is_impostor
            in_gallery = true_label in gallery_embeddings if is_genuine else False

            if is_impostor:
                if matched:
                    fp += 1  # False accept
                else:
                    tn += 1  # Correct reject
            elif not in_gallery:
                # Genuine person but failed gallery enrollment – count as FN
                fn += 1
            else:
                if matched and best_person == true_label:
                    tp += 1  # Correct accept
                elif matched and best_person != true_label:
                    fp += 1  # Misidentification
                else:
                    fn += 1  # Missed match

        total = tp + fp + fn + tn
        result = {
            **base_info,
            "threshold": thresh,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": (tp + tn) / max(total, 1),
            "tpr": tp / max(tp + fn, 1),
            "fpr": fp / max(fp + tn, 1),
            "precision": tp / max(tp + fp, 1),
            "f1": (2 * tp) / max(2 * tp + fp + fn, 1),
        }
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════
# Default configuration grid
# ═══════════════════════════════════════════════════════════════════


def default_config_grid() -> List[dict]:
    """
    Generate the default ablation grid.

    The study is designed to vary ONE factor at a time while holding
    the others at a reasonable baseline, so the effect of each factor
    is isolated.
    """
    baseline = {
        "detection_backend": "dlib_hog",
        "recognition_backend": "dlib",
        "preprocess_mode": "pad_resize",
        "alignment": True,
    }

    configs = []

    # ── A) Detection backend sweep ──────────────────────────────
    for det in ["dlib_hog", "dlib_cnn", "mediapipe", "opencv_dnn", "yolo"]:
        configs.append({**baseline, "detection_backend": det})

    # ── B) Recognition backend sweep ────────────────────────────
    for rec in ["dlib", "deepface"]:
        cfg = {**baseline, "recognition_backend": rec}
        if rec == "deepface":
            for model in ["ArcFace", "Facenet512"]:
                configs.append({**cfg, "deepface_model": model})
        else:
            configs.append(cfg)

    # ── C) Preprocessing mode sweep ─────────────────────────────
    for mode in ["none", "resize", "pad_resize"]:
        configs.append({**baseline, "preprocess_mode": mode})

    # ── D) Alignment on / off ───────────────────────────────────
    for align in [True, False]:
        # Use mediapipe which provides detector landmarks for alignment
        configs.append(
            {**baseline, "detection_backend": "mediapipe", "alignment": align}
        )

    # Deduplicate (same config may appear in multiple sweeps)
    seen = set()
    unique = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ═══════════════════════════════════════════════════════════════════
# Pretty-print summary
# ═══════════════════════════════════════════════════════════════════


def print_summary(results: List[dict]):
    """Print a compact results table to stdout."""
    if not results:
        print("No results to display.")
        return

    print()
    print("=" * 120)
    print("ABLATION STUDY RESULTS")
    print("=" * 120)

    header = (
        f"{'Detection':<12} {'Recog':<10} {'DFModel':<10} {'Preproc':<10} "
        f"{'Align':<6} {'Thresh':<7} "
        f"{'Acc':>6} {'TPR':>6} {'FPR':>6} {'F1':>6} "
        f"{'Det ms':>7} {'Emb ms':>7}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        line = (
            f"{r['detection_backend']:<12} "
            f"{r['recognition_backend']:<10} "
            f"{r.get('deepface_model', ''):<10} "
            f"{r['preprocess_mode']:<10} "
            f"{'Y' if r['alignment'] else 'N':<6} "
            f"{r['threshold']:<7.2f} "
            f"{r['accuracy']:>6.3f} "
            f"{r['tpr']:>6.3f} "
            f"{r['fpr']:>6.3f} "
            f"{r['f1']:>6.3f} "
            f"{r['avg_detection_ms']:>7.1f} "
            f"{r['avg_embedding_ms']:>7.1f}"
        )
        print(line)

    print("=" * 120)


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Path to test dataset directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="ablation_results.csv",
        help="Output CSV path (default: ablation_results.csv)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to JSON file with custom config grid",
    )
    parser.add_argument(
        "--thresholds",
        "-t",
        type=str,
        default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated threshold values to sweep (default: 0.3,0.4,0.5,0.6,0.7)",
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create an empty dataset template directory and exit",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    # Template creation mode
    if args.create_template:
        create_dataset_template(dataset_path)
        return

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    gallery, probes = load_dataset(dataset_path)
    print(
        f"  Gallery: {sum(len(v) for v in gallery.values())} images, "
        f"{len(gallery)} persons"
    )
    print(
        f"  Probes:  {sum(len(v) for v in probes.values())} images, "
        f"{len(probes)} identities"
    )

    # Load config grid
    if args.config:
        with open(args.config, "r") as f:
            configs = json.load(f)
        print(f"  Loaded {len(configs)} configurations from {args.config}")
    else:
        configs = default_config_grid()
        print(f"  Using default grid: {len(configs)} configurations")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    print(f"  Thresholds: {thresholds}")

    total_runs = len(configs) * len(thresholds)
    print(f"  Total evaluations: {len(configs)} configs x {len(thresholds)} thresholds = {total_runs}")
    print()

    # Run ablation
    all_results = []

    for i, cfg in enumerate(configs):
        label = (
            f"{cfg['detection_backend']}+{cfg['recognition_backend']}"
            f"{'(' + cfg['deepface_model'] + ')' if 'deepface_model' in cfg else ''}"
            f" preproc={cfg['preprocess_mode']}"
            f" align={'Y' if cfg['alignment'] else 'N'}"
        )
        print(f"[{i + 1}/{len(configs)}] {label}")

        try:
            results = evaluate_configuration(cfg, gallery, probes, thresholds)
            all_results.extend(results)

            # Print best threshold for this config
            if results:
                best = max(results, key=lambda r: r["f1"])
                print(
                    f"    Best F1={best['f1']:.3f} @ thresh={best['threshold']:.2f}  "
                    f"(Acc={best['accuracy']:.3f} TPR={best['tpr']:.3f} FPR={best['fpr']:.3f})  "
                    f"Det={best['avg_detection_ms']:.1f}ms Emb={best['avg_embedding_ms']:.1f}ms"
                )
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()

    # Save results
    if all_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to: {args.output}")

    # Print summary (show best threshold per config)
    best_per_config = {}
    for r in all_results:
        key = (
            r["detection_backend"],
            r["recognition_backend"],
            r.get("deepface_model", ""),
            r["preprocess_mode"],
            r["alignment"],
        )
        if key not in best_per_config or r["f1"] > best_per_config[key]["f1"]:
            best_per_config[key] = r

    print_summary(list(best_per_config.values()))


if __name__ == "__main__":
    main()
