"""
Download and organize the LFW (Labeled Faces in the Wild) dataset
into the gallery/probes format expected by ablation_study.py.

Strategy:
  - Select people who have >= 4 images for gallery+genuine probes
  - For each person: put first 2 images in gallery, remaining in probes
  - Select people with only 1-2 images as impostors (not in gallery)
  - Limit to a manageable subset for reasonable runtime
"""

import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_lfw_people


def main():
    output_dir = Path("test_data")

    print("Downloading LFW dataset (this may take a few minutes on first run)...")
    # min_faces_per_person=1 gets everyone; we'll filter ourselves
    lfw = fetch_lfw_people(
        min_faces_per_person=1,
        resize=1.0,    # Keep original resolution
        color=True,     # Get color images
        slice_=None,    # Don't crop — get full images
    )

    images = lfw.images          # (N, H, W, 3) float64 0-255 range or 0-1
    target = lfw.target          # person index per image
    target_names = lfw.target_names  # person name per index

    print(f"Total images: {len(images)}")
    print(f"Total people: {len(target_names)}")

    # Count images per person
    from collections import Counter
    counts = Counter(target)

    # People with >= 4 images -> gallery + genuine probes
    gallery_people_idx = [idx for idx, count in counts.items() if count >= 4]
    # People with exactly 1-2 images -> impostors
    impostor_people_idx = [idx for idx, count in counts.items() if count <= 2]

    # Limit for manageable test size
    np.random.seed(42)
    # Pick up to 15 gallery people
    if len(gallery_people_idx) > 15:
        gallery_people_idx = list(
            np.random.choice(gallery_people_idx, size=15, replace=False)
        )
    # Pick up to 10 impostors
    if len(impostor_people_idx) > 10:
        impostor_people_idx = list(
            np.random.choice(impostor_people_idx, size=10, replace=False)
        )

    print(f"\nSelected {len(gallery_people_idx)} people for gallery/probes")
    print(f"Selected {len(impostor_people_idx)} impostors")

    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)

    gallery_dir = output_dir / "gallery"
    probes_dir = output_dir / "probes"
    impostor_dir = probes_dir / "impostor"

    gallery_dir.mkdir(parents=True, exist_ok=True)
    probes_dir.mkdir(parents=True, exist_ok=True)
    impostor_dir.mkdir(parents=True, exist_ok=True)

    total_gallery = 0
    total_probes = 0
    total_impostor = 0

    # Process gallery people
    for person_idx in gallery_people_idx:
        person_name = target_names[person_idx].replace(" ", "_")
        # Get all image indices for this person
        img_indices = np.where(target == person_idx)[0]

        # First 2 for gallery, rest for probes
        gallery_indices = img_indices[:2]
        probe_indices = img_indices[2:6]  # up to 4 probe images

        # Save gallery images
        person_gallery_dir = gallery_dir / person_name
        person_gallery_dir.mkdir(parents=True, exist_ok=True)

        for i, img_idx in enumerate(gallery_indices):
            img = images[img_idx]
            # sklearn returns float images, convert to uint8
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Save as BGR for OpenCV compatibility (sklearn gives RGB)
            import cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_path = person_gallery_dir / f"{i+1:02d}.jpg"
            cv2.imwrite(str(save_path), img_bgr)
            total_gallery += 1

        # Save probe images
        person_probes_dir = probes_dir / person_name
        person_probes_dir.mkdir(parents=True, exist_ok=True)

        for i, img_idx in enumerate(probe_indices):
            img = images[img_idx]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            import cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_path = person_probes_dir / f"test_{i+1:02d}.jpg"
            cv2.imwrite(str(save_path), img_bgr)
            total_probes += 1

    # Process impostors
    for person_idx in impostor_people_idx:
        img_indices = np.where(target == person_idx)[0]

        for i, img_idx in enumerate(img_indices[:1]):  # 1 image per impostor
            img = images[img_idx]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            import cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            person_name = target_names[person_idx].replace(" ", "_")
            save_path = impostor_dir / f"{person_name}.jpg"
            cv2.imwrite(str(save_path), img_bgr)
            total_impostor += 1

    print(f"\nDataset created at: {output_dir}")
    print(f"  Gallery:   {total_gallery} images across {len(gallery_people_idx)} people")
    print(f"  Probes:    {total_probes} genuine probe images")
    print(f"  Impostors: {total_impostor} impostor images")
    print(f"\nReady to run: python ablation_study.py --dataset test_data")


if __name__ == "__main__":
    main()
