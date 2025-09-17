import os
import shutil

def save_results(output_dir, runs_dir):
    os.makedirs(output_dir, exist_ok=True)
    # PSEUDO_COCO already saved to output_dir in export_coco
    # Zip runs
    zip_path = os.path.join(output_dir, 'runs_yoloe.zip')
    shutil.make_archive(zip_path[:-4], 'zip', runs_dir)
    print(f'Saved results to {output_dir}: pseudo_coco.json and runs_yoloe.zip')