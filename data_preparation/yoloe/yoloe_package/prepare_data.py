import os
import shutil
from glob import glob

from .paths import INPUT_IMAGES_DIR, SUBSET

def prepare_data(subset=None):
    img_dir = INPUT_IMAGES_DIR
    if not os.path.exists(img_dir):
        raise ValueError(f"Input images directory not found: {img_dir}")
    
    all_img_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_img_paths.extend(glob(os.path.join(img_dir, '**', ext), recursive=True))
    
    if subset is not None and subset < len(all_img_paths):
        subset_dir = os.path.join(os.path.dirname(img_dir), 'subset')
        os.makedirs(subset_dir, exist_ok=True)
        selected_paths = all_img_paths[:subset]
        for src_path in selected_paths:
            rel_path = os.path.relpath(src_path, img_dir)
            dst_path = os.path.join(subset_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
        img_dir = subset_dir
        print(f"Created subset with {subset} images in {img_dir}")
    else:
        print("Using full dataset")
    
    print(f"Images prepared in {img_dir}")
    return img_dir