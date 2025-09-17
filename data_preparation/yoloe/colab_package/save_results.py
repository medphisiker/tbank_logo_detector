import os
import shutil

from .paths import OUTPUT_DIR, PSEUDO_COCO, RUNS_DIR

def save_results(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(PSEUDO_COCO, output_dir + 'pseudo_coco.json')
    # Zip runs
    shutil.make_archive('/content/runs_colab', 'zip', RUNS_DIR)
    shutil.move('/content/runs_colab.zip', output_dir + 'runs_colab.zip')
    print('Saved to GDrive:', output_dir)