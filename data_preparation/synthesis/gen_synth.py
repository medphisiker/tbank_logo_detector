# gen_synth.py — advanced synthetic data generator using crops, backgrounds and distractors
import os
import argparse
from pathlib import Path
from synthesis_generator.generator import generate_synthetic_dataset

# Parse arguments
parser = argparse.ArgumentParser(description="Generate synthetic T-Bank logo images with advanced features")
parser.add_argument("--N", type=int, default=1000, help="Number of synthetic images to generate (default: 1000)")
parser.add_argument("--min_scale_down", type=float, default=0.5, help="Minimum background scale down factor (default: 0.5)")
parser.add_argument("--iou_threshold", type=float, default=0.4, help="IoU threshold for logo placement (default: 0.4)")
parser.add_argument("--max_neg", type=int, default=15, help="Maximum number of distractors per image (default: 15)")
args = parser.parse_args()

# Set env var for consistency
os.environ['N'] = str(args.N)

script_dir = Path(__file__).parent

# Paths
crops_dir = script_dir / "crops"
bg_dir = script_dir / "backgrounds"
bg_objects_dir = script_dir / "background_objects"  # New: distractors directory

# Determine output base path (Docker vs local)
if Path("/app/data").exists():
    out_base = Path("/app/data") / "data_synt"
else:
    out_base = script_dir.parent.parent / "data" / "data_synt"

N = int(os.getenv('N', 1000))  # Total synthetic images, from arg or env var

# Debug prints
print(f"Script dir: {script_dir}")
print(f"Crops dir: {crops_dir}")
print(f"Bg dir: {bg_dir}")
print(f"Bg objects dir: {bg_objects_dir} (exists: {bg_objects_dir.exists()})")
print(f"Out base: {out_base}")
print(f"N: {N}")
print(f"Min scale down: {args.min_scale_down}")
print(f"IoU threshold: {args.iou_threshold}")
print(f"Max distractors: {args.max_neg}")

generate_synthetic_dataset(
    crops_dir,
    bg_dir,
    out_base,
    N,
    bg_objects_dir=bg_objects_dir,
    min_scale_down=args.min_scale_down,
    iou_threshold=args.iou_threshold,
    max_neg=args.max_neg
)