#!/usr/bin/env python3
"""
Prepare background objects (distractors) for synthetic data generation.

This script helps set up distractor objects that will be placed on backgrounds
to create hard negatives for training the logo detector.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

def download_distractors(output_dir: str, num_images: int = 200) -> None:
    """Download distractor images from various sources."""
    os.makedirs(output_dir, exist_ok=True)

    # Sources for distractors (logos of other banks, generic emblems, etc.)
    sources = [
        # Tinkoff variants (for hard negatives)
        "https://logos-world.net/wp-content/uploads/2020/12/Tinkoff-Logo.png",
        # Other bank logos
        "https://logos-world.net/wp-content/uploads/2020/11/Sberbank-Logo.png",
        "https://logos-world.net/wp-content/uploads/2020/11/VTB-Logo.png",
        # Generic emblems - using picsum for now
    ]

    # Add random images from picsum as generic distractors
    for i in range(num_images - len(sources)):
        sources.append(f"https://picsum.photos/200/200?random={i+1000}")

    successful = 0
    for i, url in enumerate(tqdm(sources[:num_images], desc="Downloading distractors")):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                ext = '.png' if 'png' in url else '.jpg'
                filename = f"distractor_{i:03d}{ext}"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                successful += 1
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    print(f"Downloaded {successful}/{num_images} distractor images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare background objects for synthesis")
    parser.add_argument("--output", type=str, default="background_objects", help="Output directory")
    parser.add_argument("--num", type=int, default=200, help="Number of distractor images")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output

    print(f"Preparing {args.num} distractor images in {output_dir}")
    download_distractors(str(output_dir), args.num)