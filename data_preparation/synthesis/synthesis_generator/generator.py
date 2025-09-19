import os
import random
import json
from pathlib import Path
from PIL import Image
import numpy as np
from .augmentations import get_augmentation_pipeline, get_background_aug_pipeline, get_neg_aug_pipeline, get_logo_aug_pipeline


def calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    """Calculate IoU between two bounding boxes in YOLO format (cx, cy, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to corner coordinates
    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2

    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def load_background_objects(bg_objects_dir: Path) -> list:
    """Load list of background object (distractor) image paths."""
    return [str(f) for f in bg_objects_dir.glob("*.png")]


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


def place_distractors(bg: Image.Image, bg_objects: list, neg_aug_pipeline, max_neg: int = 15) -> Image.Image:
    """Place distractor objects on background without labels."""
    if not bg_objects:
        return bg

    W, H = bg.size
    num_neg = random.randint(0, max_neg)
    placed_positions = []

    for _ in range(num_neg):
        obj_path = random.choice(bg_objects)
        obj = Image.open(obj_path).convert("RGBA")

        # Random augment distractor
        obj_arr = np.array(obj)
        auged_obj = neg_aug_pipeline(image=obj_arr)['image']
        obj = Image.fromarray(auged_obj)

        # Random scale and rotate
        scale = random.uniform(0.05, 0.2)
        nw = int(W * scale)
        nh = int(obj.height * scale)
        obj_t = obj.resize((nw, nh), Image.LANCZOS).rotate(random.uniform(-45, 45), expand=True)

        # Random position
        attempts = 0
        while attempts < 5:
            x = random.randint(0, max(0, W - obj_t.width))
            y = random.randint(0, max(0, H - obj_t.height))

            # Check IoU with existing distractors (loose check)
            current_bbox = (x/W, y/H, obj_t.width/W, obj_t.height/H)
            overlap = False
            for prev_bbox in placed_positions:
                if calculate_iou(current_bbox, prev_bbox) > 0.5:
                    overlap = True
                    break

            if not overlap:
                placed_positions.append(current_bbox)
                bg.paste(obj_t, (x, y), obj_t)
                break
            attempts += 1

    return bg


def place_multi_logos(bg: Image.Image, crops_by_class: dict, logo_aug_pipeline, iou_threshold: float = 0.4, max_logos: int = 10) -> tuple:
    """Place multiple logos on background with IoU control."""
    W, H = bg.size
    num_logos = random.randint(1, max_logos)
    placed_bboxes = []
    bboxes_info = []

    for _ in range(num_logos):
        attempts = 0
        max_placement_attempts = 5

        while attempts < max_placement_attempts:
            # Choose class and crop
            cls = random.choice([0, 1, 2])
            crop_path = random.choice(crops_by_class[cls])

            ref = Image.open(crop_path).convert("RGBA")

            # Random scale and rotate
            scale = random.uniform(0.15, 0.45)
            nw = int(W * scale)
            nh = int(ref.height * scale)
            ref_t = ref.resize((nw, nh), Image.LANCZOS).rotate(random.uniform(-25, 25), expand=True)

            # Random position with clipping control
            pos_attempts = 0
            while pos_attempts < 10:
                x = random.randint(0, max(0, W - ref_t.width))
                y = random.randint(0, max(0, H - ref_t.height))

                # Calculate bbox
                cx = (x + ref_t.width / 2) / W
                cy = (y + ref_t.height / 2) / H
                bw = ref_t.width / W
                bh = ref_t.height / H
                current_bbox = (cx, cy, bw, bh)

                # Check visibility (IoU with full image >= 0.8)
                full_image_bbox = (0.5, 0.5, 1.0, 1.0)
                visibility_iou = calculate_iou(current_bbox, full_image_bbox)

                if visibility_iou >= 0.8:
                    # Check IoU with existing logos
                    overlap = False
                    for prev_bbox in placed_bboxes:
                        if calculate_iou(current_bbox, prev_bbox) > iou_threshold:
                            overlap = True
                            break

                    if not overlap:
                        # Augment logo
                        ref_arr = np.array(ref_t)
                        auged_ref = logo_aug_pipeline(image=ref_arr)['image']
                        ref_t = Image.fromarray(auged_ref)

                        # Place logo
                        bg.paste(ref_t, (x, y), ref_t)
                        placed_bboxes.append(current_bbox)
                        bboxes_info.append((cls, current_bbox))
                        break

                pos_attempts += 1
            else:
                # Couldn't place, try next logo
                break

            attempts += 1

    return bg, bboxes_info


def generate_synthetic_image(bg_path: str, crop_path: str, aug_pipeline, min_scale_down: float = 0.5) -> tuple:
    """Generate a single synthetic image with logo overlay and random background resize."""
    bg = Image.open(bg_path).convert("RGB")
    W_orig, H_orig = bg.size

    # Random resize background (scale down 0.5-1.0)
    scale_down = random.uniform(min_scale_down, 1.0)
    W = int(W_orig * scale_down)
    H = int(H_orig * scale_down)
    bg = bg.resize((W, H), Image.LANCZOS)

    ref = Image.open(crop_path).convert("RGBA")

    # Random scale and rotate
    scale = random.uniform(0.15, 0.45)
    nw = int(W * scale)
    nh = int(ref.height * scale)
    ref_t = ref.resize((nw, nh), Image.LANCZOS).rotate(random.uniform(-25, 25), expand=True)

    # Random position with clipping control
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
        x = random.randint(0, max(0, W - ref_t.width))
        y = random.randint(0, max(0, H - ref_t.height))

        # Check if bbox is at least 80% visible
        cx = (x + ref_t.width / 2) / W
        cy = (y + ref_t.height / 2) / H
        bw = ref_t.width / W
        bh = ref_t.height / H

        # IoU with full image
        full_image_bbox = (0.5, 0.5, 1.0, 1.0)  # Center of full image
        current_bbox = (cx, cy, bw, bh)
        iou = calculate_iou(current_bbox, full_image_bbox)

        if iou >= 0.8:
            break
        attempts += 1
    else:
        # If couldn't find good position, place at center
        x = (W - ref_t.width) // 2
        y = (H - ref_t.height) // 2
        cx = (x + ref_t.width / 2) / W
        cy = (y + ref_t.height / 2) / H
        bw = ref_t.width / W
        bh = ref_t.height / H

    out = bg.copy()
    out.paste(ref_t, (x, y), ref_t)

    # Augment
    arr = np.array(out)
    auged = aug_pipeline(image=arr)['image']
    out = Image.fromarray(auged)

    return out, (cx, cy, bw, bh), (W_orig, H_orig, W, H)


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
    aug_pipeline=None,
    bg_objects_dir: Path = None,
    min_scale_down: float = 0.5,
    iou_threshold: float = 0.4,
    max_neg: int = 15
) -> None:
    """Generate synthetic dataset with advanced features."""
    if aug_pipeline is None:
        aug_pipeline = get_augmentation_pipeline()

    # Get specialized pipelines
    bg_aug_pipeline = get_background_aug_pipeline()
    neg_aug_pipeline = get_neg_aug_pipeline()
    logo_aug_pipeline = get_logo_aug_pipeline()

    setup_output_dirs(out_base)

    crops_by_class = load_crops_by_class(crops_dir)
    bgs = load_backgrounds(bg_dir)
    bg_objects = load_background_objects(bg_objects_dir) if bg_objects_dir and bg_objects_dir.exists() else []

    if not all(len(crops) > 0 for crops in crops_by_class.values()) or not bgs:
        raise ValueError("No crops or backgrounds found.")

    # Balanced class generation
    targets_per_class = N // 3
    class_counts = {0: 0, 1: 0, 2: 0}
    max_per_class = targets_per_class + (N % 3)  # Distribute remainder

    i = 0
    while sum(class_counts.values()) < N:
        # Choose class with remaining targets
        available_classes = [cls for cls in [0, 1, 2] if class_counts[cls] < max_per_class]
        if not available_classes:
            break
        cls = random.choice(available_classes)

        bg_path = random.choice(bgs)
        crop_path = random.choice(crops_by_class[cls])

        # Load and resize background
        bg = Image.open(bg_path).convert("RGB")
        W_orig, H_orig = bg.size
        scale_down = random.uniform(min_scale_down, 1.0)
        W = int(W_orig * scale_down)
        H = int(H_orig * scale_down)
        bg = bg.resize((W, H), Image.LANCZOS)

        # Apply background augmentations
        bg_arr = np.array(bg)
        auged_bg = bg_aug_pipeline(image=bg_arr)['image']
        bg = Image.fromarray(auged_bg)

        # Place distractors
        if bg_objects:
            bg = place_distractors(bg, bg_objects, neg_aug_pipeline, max_neg)

        # Place multi-logos
        bg, bboxes_info = place_multi_logos(bg, crops_by_class, logo_aug_pipeline, iou_threshold)

        # If no logos were placed, skip this image
        if not bboxes_info:
            continue

        # Split assignment (80/10/10)
        if i < 0.8 * N:
            split = 'train'
        elif i < 0.9 * N:
            split = 'val'
        else:
            split = 'test'

        fname = f"synth_{i:05d}.jpg"
        out_path = out_base / 'images' / split / fname
        bg.save(out_path, quality=90)

        # Save labels for all placed logos
        lbl_path = out_base / 'labels' / split / fname.replace('.jpg', '.txt')
        with open(lbl_path, 'w') as f:
            for cls_label, bbox in bboxes_info:
                cx, cy, bw, bh = bbox
                f.write(f"{cls_label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                class_counts[cls_label] += 1

        i += 1

    print(f"Generated {i} synthetic images with balanced classes: {class_counts}")
    print(f"Splits: {int(0.8*i)} train, {int(0.1*i)} val, {int(0.1*i)} test in {out_base}")
    print(f"Background resize: min_scale_down={min_scale_down}, distractors: {len(bg_objects) if bg_objects else 0} objects")