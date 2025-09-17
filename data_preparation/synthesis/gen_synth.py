# gen_synth.py — synthetic data generator using crops and backgrounds
import os
import random
import json
from PIL import Image, ImageEnhance
try:
    import albumentations as A
    use_alb = True
except ImportError:
    use_alb = False
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent

# Paths
crops_dir = script_dir / "crops"
bg_dir = script_dir / "backgrounds"
out_base = Path("/app/data") / "data_synt"
N = int(os.getenv('N', 20))  # Total synthetic images, set via env var for testing

# Debug prints
print(f"Script dir: {script_dir}")
print(f"Crops dir: {crops_dir}")
print(f"Bg dir: {bg_dir}")
print(f"Out base: {out_base}")
print(f"N: {N}")

# Create output dirs
for split in ['train', 'val', 'test']:
    (Path(out_base) / 'images' / split).mkdir(parents=True, exist_ok=True)
    (Path(out_base) / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Augmentations
if use_alb:
    aug = A.Compose([
        A.RandomBrightnessContrast(p=0.7, brightness_limit=0.3, contrast_limit=0.3),
        A.GaussNoise(p=0.3),
        A.MotionBlur(p=0.2),
        A.HueSaturationValue(p=0.6, hue_shift_limit=15),
    ])
else:
    aug = None  # Fallback to PIL: brightness/contrast only

# Load crops by class (purple:0, white:1, yellow:2)
crops_by_class = {0: [], 1: [], 2: []}
crops_path = Path(crops_dir)
for crop_file in crops_path.glob("*.png"):
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

bgs = [str(f) for f in Path(bg_dir).glob("*.jpg")]

if not all(len(crops) > 0 for crops in crops_by_class.values()) or not bgs:
    raise ValueError("No crops or backgrounds found. Run crop_logos.py and download_backgrounds.py first.")

# Generate
for i in range(N):
    bg_path = random.choice(bgs)
    bg = Image.open(bg_path).convert("RGB")
    W, H = bg.size
    cls = random.choice([0, 1, 2])
    crop_path = random.choice(crops_by_class[cls])
    ref = Image.open(crop_path).convert("RGBA")

    print(f"Crops by class: {crops_by_class}")
    print(f"Number of bgs: {len(bgs)}")
    print(f"Bgs sample: {bgs[:3] if bgs else 'None'}")

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
    if use_alb:
        arr = np.array(out)
        auged = aug(image=arr)['image']
        out = Image.fromarray(auged)
    else:
        # PIL fallback: brightness and contrast (simple, no noise/blur)
        factor_b = random.uniform(0.7, 1.3)
        out = ImageEnhance.Brightness(out).enhance(factor_b)
        factor_c = random.uniform(0.7, 1.3)
        out = ImageEnhance.Contrast(out).enhance(factor_c)

    # Split assignment (80/10/10)
    if i < 0.8 * N:
        split = 'train'
    elif i < 0.9 * N:
        split = 'val'
    else:
        split = 'test'

    fname = f"synth_{i:05d}.jpg"
    out_path = Path(out_base) / 'images' / split / fname
    print(f"Saving image to: {out_path}")
    try:
        out.save(out_path, quality=90)
        print(f"Saved image: {out_path.exists()}")
    except Exception as e:
        print(f"Error saving image: {e}")

    # YOLO label
    cx = (x + ref_t.width / 2) / W
    cy = (y + ref_t.height / 2) / H
    bw = ref_t.width / W
    bh = ref_t.height / H
    lbl_path = Path(out_base) / 'labels' / split / fname.replace('.jpg', '.txt')
    print(f"Saving label to: {lbl_path}")
    with open(lbl_path, 'w') as f:
        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"Saved label: {lbl_path.exists()}")

print(f"Generated {N} synthetic images: {int(0.8*N)} train, {int(0.1*N)} val, {int(0.1*N)} test in {out_base}")