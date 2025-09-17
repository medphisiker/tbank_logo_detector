import os
import zipfile
import shutil
from glob import glob

from .paths import ZIP_FILES, GDRIVE_BASE, DATA_DIR, SUBSET

def prepare_data(gdrive_base, subset=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    for zip_name in ZIP_FILES:
        zip_path = os.path.join(gdrive_base, zip_name)
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print(f"Unzipped {zip_name}")
        else:
            print(f"Warning: {zip_path} not found")
    
    img_dir = DATA_DIR
    all_img_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_img_paths.extend(glob(os.path.join(img_dir, '**', ext), recursive=True))
    
    if subset is not None and subset < len(all_img_paths):
        subset_dir = os.path.join(img_dir, 'subset')
        os.makedirs(subset_dir, exist_ok=True)
        selected_paths = all_img_paths[:subset]
        for src_path in selected_paths:
            rel_path = os.path.relpath(src_path, img_dir)
            dst_path = os.path.join(subset_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
        img_dir = subset_dir
        print(f"Subset to {subset} images in {subset_dir}")
    else:
        print("Using full dataset")
    
    print("Files prepared in /content/data/")
    return img_dir