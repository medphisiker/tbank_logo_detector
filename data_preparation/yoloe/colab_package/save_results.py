import os
import shutil

from .paths import OUTPUT_DIR, PSEUDO_COCO, RUNS_DIR

def save_results(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # PSEUDO_COCO already in OUTPUT_DIR, no copy needed
    # Zip runs
    zip_path = os.path.join(output_dir, 'runs_yoloe.zip')
    shutil.make_archive(zip_path[:-4], 'zip', RUNS_DIR)
    print(f'Saved results to {output_dir}: pseudo_coco.json and runs_yoloe.zip')