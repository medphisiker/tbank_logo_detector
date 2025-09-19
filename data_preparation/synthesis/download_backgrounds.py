from synthesis_generator.background_utils import download_backgrounds
from pathlib import Path
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Download background images for synthesis")
parser.add_argument("--num", type=int, default=1000, help="Number of backgrounds to download")
parser.add_argument("--size", type=int, default=1920, help="Image size (default: 1920 for high-res)")
parser.add_argument("--thematic", action="store_true", help="Use thematic backgrounds (requires API key)")
args = parser.parse_args()

# Конфигурация
script_dir = Path(__file__).parent
backgrounds_dir = script_dir / "backgrounds"

print(f"Downloading {args.num} backgrounds of size {args.size}x{args.size}")
print(f"Thematic: {args.thematic}")

download_backgrounds(str(backgrounds_dir), args.num, args.size, args.thematic)