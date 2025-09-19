import os
import shutil
from glob import glob

def prepare_data(input_dir, subset=None):
    print(f"Preparing data from {input_dir}, subset={subset}")
    img_dir = input_dir
    if not os.path.exists(img_dir):
        raise ValueError(f"Input images directory not found: {img_dir}")
    
    all_img_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_img_paths.extend(glob(os.path.join(img_dir, '**', ext), recursive=True))
    print(f"Found {len(all_img_paths)} images")
    
    if subset is not None and subset < len(all_img_paths):
        subset_dir = os.path.join(os.path.dirname(img_dir), 'subset')

        # Clear previous subset files
        if os.path.exists(subset_dir):
            print(f"Clearing previous subset files from {subset_dir}")
            shutil.rmtree(subset_dir)

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