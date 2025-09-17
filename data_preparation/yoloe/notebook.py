# %% [markdown]
# # YOLOE Labeling in Colab with GDrive
# 
# Этот notebook предназначен для автоматической разметки подмножества data_sirius (~1000 изображений) с использованием модели YOLOE в Google Colab.
# 
# Цель: детекция 3 классов логотипов Т-Банка — yellow_shield_black_T, white_shield_black_T, purple_shield_white_T.
# 
# Используются гибридные промпты (визуальные bbox из референсов и текстовые описания).
# 
# Все операции с файлами через Google Drive: mount, copy/unzip в /content/, predict, save results back to GDrive.
# 
# Выход: pseudo_coco.json с псевдо-аннотациями (COCO формат) и ZIP с runs/predict (TXT labels и изображения с bbox).
# 
# **Предварительные требования:**
# 
# - В GDrive в папке tbank_project/ (или измените GDRIVE_BASE) разместите:
#   - data_sirius.zip (~1000 изображений для разметки)
#   - refs_ls_coco.json (COCO экспорт из Label Studio с bbox для референсных изображений, классы 0-2)
#   - Опционально: example_ref.jpg (референсное изображение)
#   - Для optional eval: small_gt_coco.json (маленький GT COCO для mAP)

# %%
!pip install -U ultralytics pycocotools opencv-python pillow numpy google-colab --quiet

# %% [markdown]
# ## Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## Define Paths and Subset (edit these variables)

# %%
# GDrive paths
GDRIVE_BASE = '/content/drive/MyDrive/tbank_project/'  
DATA_ZIP = GDRIVE_BASE + 'data_sirius.zip'
REFS_JSON = GDRIVE_BASE + 'refs_ls_coco.json'
EXAMPLE_REF = GDRIVE_BASE + 'example_ref.jpg'  # Optional
OUTPUT_DIR = GDRIVE_BASE + 'yoloe_results/'
SUBSET = None  # None for full dataset, or int e.g. 10 for first 10 images (for testing)
GT_COCO = GDRIVE_BASE + 'small_gt_coco.json'  # Optional GT for mAP

# Local /content paths
DATA_DIR = '/content/data_sirius_subset'
REFS_LOCAL = '/content/refs_ls_coco.json'
EXAMPLE_LOCAL = '/content/example_ref.jpg'
PSEUDO_COCO = '/content/pseudo_coco.json'
RUNS_DIR = '/content/runs/colab_predict'
LABELS_DIR = RUNS_DIR + '/labels'
GT_COCO_LOCAL = '/content/small_gt_coco.json'

# %% [markdown]
# ## Copy and Prepare Subset from GDrive to /content/

# %%
import os, zipfile, shutil, random
os.makedirs('/content/data_sirius_subset', exist_ok=True)
# Unzip full
with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
    zip_ref.extractall('/content/data_sirius_subset')
# Copy refs
shutil.copy(REFS_JSON, '/content/refs_ls_coco.json')
if os.path.exists(EXAMPLE_REF):
    shutil.copy(EXAMPLE_REF, '/content/example_ref.jpg')
# Subset if needed
img_dir = '/content/data_sirius_subset'
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if SUBSET is not None and SUBSET < len(img_files):
    subset_dir = '/content/data_sirius_subset_subset'
    os.makedirs(subset_dir, exist_ok=True)
    for f in img_files[:SUBSET]:  # First N
        shutil.move(os.path.join(img_dir, f), os.path.join(subset_dir, f))
    img_dir = subset_dir
    print(f"Subset to {SUBSET} images in {subset_dir}")
else:
    print("Using full dataset")
print("Files prepared in /content/")

# %% [markdown]
# ## Load and Prepare YOLOE

# %%
from ultralytics import YOLOE
import numpy as np
import json
model = YOLOE('yoloe-11l-seg.pt')
# Load refs for visual prompts
with open(REFS_LOCAL, 'r') as f:
    refs_data = json.load(f)
# Group bboxes/cls by class (0-2, assume grouped in LS)
grouped_bboxes = {}  # dict class_id: list of [x1,y1,x2,y2]
grouped_cls = np.array([0,1,2])  # For multi-class
# Example grouping code (adapt based on refs_data['annotations'])
for ann in refs_data['annotations']:
    cls_id = ann['category_id']
    bbox = ann['bbox']  # [x,y,w,h] -> [x,y,x+w,y+h]
    x1, y1, w, h = bbox
    if cls_id not in grouped_bboxes:
        grouped_bboxes[cls_id] = []
    grouped_bboxes[cls_id].append([x1, y1, x1+w, y1+h])
visual_prompts = {'bboxes': np.array(grouped_bboxes), 'cls': grouped_cls}
# Text prompts
names = ['yellow_shield_black_T', 'white_shield_black_T', 'purple_shield_white_T']
text_prompts = ['yellow shield with black T logo', 'white shield with black T logo', 'purple shield with white T logo']
model.set_classes(names, text_prompts)  # Hybrid

# %% [markdown]
# ## Run YOLOE Predict

# %%
import torch
results = model.predict(
    source=img_dir,
    visual_prompts=visual_prompts,
    conf=0.5,
    iou=0.7,
    save_txt=True,
    project=RUNS_DIR,
    device=0 if torch.cuda.is_available() else 'cpu'
)
print('Prediction complete. Results in ' + RUNS_DIR + '/')

# %% [markdown]
# ## Export to COCO

# %%
# Script to convert predict txt to pseudo_coco.json
# Assume standard YOLO txt: class cx cy w h conf per line
# Build COCO structure
coco = {
    'info': {'description': 'YOLOE pseudo labels'},
    'licenses': [],
    'images': [],
    'annotations': [],
    'categories': [{'id': i+1, 'name': name} for i, name in enumerate(names)]
}
image_id = 0
ann_id = 0
for img_file in os.listdir(img_dir):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')): continue
    img_path = os.path.join(img_dir, img_file)
    # Add image
    from PIL import Image
    with Image.open(img_path) as img:
        w, h = img.size
    coco['images'].append({'id': image_id, 'file_name': img_file, 'width': w, 'height': h})
    # Load txt if exists
    txt_file = img_file.rsplit('.',1)[0] + '.txt'
    txt_path = os.path.join(LABELS_DIR, txt_file)
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls_id = int(parts[0])  # 0-2
                conf = float(parts[5]) if len(parts) >5 else 1.0
                cx, cy, bw, bh = map(float, parts[1:5])
                x, y = cx - bw/2, cy - bh/2
                w_ann, h_ann = bw * w, bh * h
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cls_id +1,
                    'bbox': [x, y, w_ann, h_ann],
                    'area': w_ann * h_ann,
                    'iscrowd': 0,
                    'score': conf
                })
                ann_id += 1
    image_id += 1
with open(PSEUDO_COCO, 'w') as f:
    json.dump(coco, f)
print('Exported ' + PSEUDO_COCO)

# %% [markdown]
# Subset handled: If SUBSET set, only those images processed/saved.

# %% [markdown]
# ## Save Results to GDrive

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)
shutil.copy(PSEUDO_COCO, OUTPUT_DIR + 'pseudo_coco.json')
# Zip runs
shutil.make_archive('/content/runs_colab', 'zip', RUNS_DIR)
shutil.move('/content/runs_colab.zip', OUTPUT_DIR + 'runs_colab.zip')
print('Saved to GDrive:', OUTPUT_DIR)

# %% [markdown]
# ## Optional: Evaluate mAP (upload small GT COCO to GDrive)

# %%
# If you have GT_COCO, uncomment below
# shutil.copy(GT_COCO, GT_COCO_LOCAL)
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# coco_gt = COCO(GT_COCO_LOCAL)
# coco_dt = coco_gt.loadRes(PSEUDO_COCO)
# coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()
print('Add GT path and run for mAP')


