import os
import random
import json
from pathlib import Path
from PIL import Image
import numpy as np
from .augmentations import get_augmentation_pipeline


def setup_output_dirs(out_base: Path, splits: list = ['train', 'val', 'test']) -> None:
    """Create output directories for images and labels."""
    for split in splits:
        (out_base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_base / 'labels' / split).mkdir(parents=True, exist_ok=True)


def load_crops_by_class(crops_dir: Path) -> dict:
    """Load crop files organized by class."""
    crops_by_class = {0: [], 1: [], 2: []}
    for crop_file in crops_dir.glob("*.png"):
        name = crop_file.stem
        if name.startswith('purple'):
            cls = 0
        elif name.startswith('white'):
            cls = 1
        elif name.startswith('yellow'):
            cls = 2
        else:
            continue
        crops_by_class[cls].append(str(crop_file))
    return crops_by_class


def load_backgrounds(bg_dir: Path) -> list:
    """Load list of background image paths."""
    return [str(f) for f in bg_dir.glob("*.jpg")]


def generate_synthetic_image(bg_path: str, crop_path: str, aug_pipeline) -> tuple:
    """Generate a single synthetic image with logo overlay."""
    bg = Image.open(bg_path).convert("RGB")
    W, H = bg.size
    ref = Image.open(crop_path).convert("RGBA")

    # Random scale and rotate
    scale = random.uniform(0.15, 0.45)
    nw = int(W * scale)
    nh = int(ref.height * scale)
    ref_t = ref.resize((nw, nh), Image.LANCZOS).rotate(random.uniform(-25, 25), expand=True)

    x = random.randint(0, max(0, W - ref_t.width))
    y = random.randint(0, max(0, H - ref_t.height))

    out = bg.copy()
    out.paste(ref_t, (x, y), ref_t)

    # Augment
    arr = np.array(out)
    auged = aug_pipeline(image=arr)['image']
    out = Image.fromarray(auged)

    # Calculate YOLO bbox
    cx = (x + ref_t.width / 2) / W
    cy = (y + ref_t.height / 2) / H
    bw = ref_t.width / W
    bh = ref_t.height / H

    return out, (cx, cy, bw, bh)


def save_yolo_label(lbl_path: Path, cls: int, bbox: tuple) -> None:
    """Save YOLO format label."""
    cx, cy, bw, bh = bbox
    with open(lbl_path, 'w') as f:
        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def generate_synthetic_dataset(
    crops_dir: Path,
    bg_dir: Path,
    out_base: Path,
    N: int,
    aug_pipeline=None
) -> None:
    """Generate synthetic dataset."""
    if aug_pipeline is None:
        aug_pipeline = get_augmentation_pipeline()

    setup_output_dirs(out_base)

    crops_by_class = load_crops_by_class(crops_dir)
    bgs = load_backgrounds(bg_dir)

    if not all(len(crops) > 0 for crops in crops_by_class.values()) or not bgs:
        raise ValueError("No crops or backgrounds found.")

    for i in range(N):
        bg_path = random.choice(bgs)
        cls = random.choice([0, 1, 2])
        crop_path = random.choice(crops_by_class[cls])

        out_img, bbox = generate_synthetic_image(bg_path, crop_path, aug_pipeline)

        # Split assignment (80/10/10)
        if i < 0.8 * N:
            split = 'train'
        elif i < 0.9 * N:
            split = 'val'
        else:
            split = 'test'

        fname = f"synth_{i:05d}.jpg"
        out_path = out_base / 'images' / split / fname
        out_img.save(out_path, quality=90)

        lbl_path = out_base / 'labels' / split / fname.replace('.jpg', '.txt')
        save_yolo_label(lbl_path, cls, bbox)

    print(f"Generated {N} synthetic images: {int(0.8*N)} train, {int(0.1*N)} val, {int(0.1*N)} test in {out_base}")