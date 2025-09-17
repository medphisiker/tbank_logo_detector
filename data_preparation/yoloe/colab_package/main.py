from google.colab import drive
from .prepare_data import prepare_data
from .yolo_predict import run_yolo_predict
from .export_coco import export_to_coco
from .save_results import save_results
from .paths import GDRIVE_BASE, OUTPUT_DIR, SUBSET

# Mount drive
drive.mount('/content/drive')

# Prepare data
img_dir = prepare_data(GDRIVE_BASE, SUBSET)

# Run YOLO predict
results = run_yolo_predict(img_dir)

# Export to COCO
export_to_coco(img_dir)

# Save results
save_results(OUTPUT_DIR)