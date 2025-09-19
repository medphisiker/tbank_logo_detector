# gen_synth.py — synthetic data generator using crops and backgrounds
import os
import argparse
from pathlib import Path
from synthesis_generator.generator import generate_synthetic_dataset

# Parse arguments
parser = argparse.ArgumentParser(description="Generate synthetic T-Bank logo images")
parser.add_argument("--N", type=int, default=20, help="Number of synthetic images to generate (default: 20)")
args = parser.parse_args()

# Set env var for consistency
os.environ['N'] = str(args.N)

script_dir = Path(__file__).parent

# Paths
crops_dir = script_dir / "crops"
bg_dir = script_dir / "backgrounds"

# Determine output base path (Docker vs local)
if Path("/app/data").exists():
    out_base = Path("/app/data") / "data_synt"
else:
    out_base = script_dir.parent.parent / "data" / "data_synt"

N = int(os.getenv('N', 20))  # Total synthetic images, from arg or env var

# Debug prints
print(f"Script dir: {script_dir}")
print(f"Crops dir: {crops_dir}")
print(f"Bg dir: {bg_dir}")
print(f"Out base: {out_base}")
print(f"N: {N}")

generate_synthetic_dataset(crops_dir, bg_dir, out_base, N)