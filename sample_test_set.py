import os
import json
import random
import numpy as np
import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from pycocotools import mask as cocomask
from tqdm import tqdm

# Конфигурация
MODEL_PATH = 'yoloe-11l-seg.pt'  # YOLOE модель
DATA_DIR = Path('data/data_sirius/images')  # Папка с изображениями data_sirius
REFS_COCO_PATH = 'data/tbank_official_logos/refs_ls_coco.json'  # LS COCO для refs
OUTPUT_DIR = Path('data_preparation/yoloe/test_set')
IMAGES_OUT = OUTPUT_DIR / 'images'
LABELS_OUT = OUTPUT_DIR / 'labels'
COCO_OUT = OUTPUT_DIR / 'annotations_coco.json'
CONF_THRESH = 0.5
IOU_THRESH = 0.7
IMG_SIZE = 640
N_POS = 500
N_NEG = 500

# Классы
CLASS_NAMES = [
    "yellow_shield_black_T: Жёлтый щит с чёрной жирной буквой T, минималистичный, без теней или с глубиной через переходы.",
    "white_shield_black_T: Белый щит с чёрной T, чёткие границы, мягкая тень.",
    "purple_shield_white_T: Фиолетовый щит с белой T, наклонённый, градиентный."
]

REF_BASE_DIR = 'data/tbank_official_logos'
REF_FILENAMES = ['logo0.png', 'logo1.png', 'logo2.png', 'logo3.png', 'logo4.png', 'logo5.png', 'logo6.png', 'logo7.png', 'logo8.png']

def load_refs_from_coco(coco_path):
    """Загрузка references из LS COCO"""
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    img_dict = {img['id']: img for img in coco['images']}
    refer_data = []
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = img_dict[img_id]
        file_name = img_info['file_name']
        if file_name not in REF_FILENAMES:
            continue
        
        ref_path = os.path.join(REF_BASE_DIR, file_name)
        if not os.path.exists(ref_path):
            continue
        
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            continue
        
        x, y, w, h = ann['bbox']
        width, height = img_info['width'], img_info['height']
        x1, y1 = x / width, y / height
        x2, y2 = (x + w) / width, (y + h) / height
        bbox = np.array([x1, y1, x2, y2])
        cls_id = ann['category_id']
        refer_data.append((ref_img, bbox, cls_id))
    
    grouped = {0: [], 1: [], 2: []}
    for img, bbox, cls in refer_data:
        grouped[cls].append((img, bbox))
    
    return grouped

def yolo_to_coco(images_dir, labels_dir, output_json):
    """Конверт YOLO txt to COCO"""
    annotations = []
    images_list = []
    categories = [{"id": i, "name": name.split(':')[0].strip()} for i, name in enumerate(CLASS_NAMES)]
    
    img_id = 0
    ann_id = 0
    
    for img_file in sorted(images_dir.glob('*.jpg')):
        img_name = img_file.stem
        img_path = str(img_file)
        h, w = cv2.imread(img_path).shape[:2]
        
        images_list.append({
            "id": img_id,
            "file_name": img_name + '.jpg',
            "width": w,
            "height": h
        })
        
        txt_file = labels_dir / f"{img_name}.txt"
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, x_center, y_center, width, height = map(float, parts[:5])
                x = (x_center - width / 2) * w
                y = (y_center - height / 2) * h
                bbox_w = width * w
                bbox_h = height * h
                
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [float(x), float(y), float(bbox_w), float(bbox_h)],
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0
                })
                ann_id += 1
        
        img_id += 1
    
    coco = {
        "images": images_list,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)

def main():
    # Создать директории
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.mkdir(parents=True, exist_ok=True)
    
    # Загрузка модели
    model = YOLO(MODEL_PATH)
    text_pe = ops.get_text_pe(CLASS_NAMES)
    model.set_classes(CLASS_NAMES, text_pe)
    
    # Visual prompts
    refer_data = load_refs_from_coco(REFS_COCO_PATH)
    refer_images = []
    bboxes = []
    cls_ids = []
    for cls_id, data_list in refer_data.items():
        for img, bbox in data_list:
            refer_images.append(img)
            bboxes.append(bbox)
            cls_ids.append(cls_id)
    
    bboxes = np.array(bboxes)
    cls_ids = np.array(cls_ids)
    visual_prompts = {
        'refer_images': refer_images,
        'bboxes': bboxes,
        'cls': cls_ids
    }
    
    # Collect images
    images = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png'))
    if len(images) < N_POS + N_NEG:
        raise ValueError(f"Недостаточно изображений в {DATA_DIR}")
    
    # Inference with save_txt
    results = model.predict(
        source=images,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save_txt=True,
        save_conf=True,
        project=str(LABELS_OUT.parent),
        name='temp_inference',
        exist_ok=True,
        visual_prompts=visual_prompts,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=False
    )
    
    # Collect conf per image (temp txt in temp_inference/labels/)
    temp_labels_dir = Path('runs/detect/temp_inference/labels')
    positives = []
    negatives = []
    
    for i, img_path in enumerate(images):
        img_name = Path(img_path).stem
        txt_path = temp_labels_dir / f"{img_name}.txt"
        
        max_conf = 0
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            if lines:
                confs = [float(line.split()[5]) for line in lines if len(line.split()) > 5]
                if confs:
                    max_conf = max(confs)
        
        if max_conf > CONF_THRESH:
            positives.append((img_path, max_conf, img_name))
        else:
            negatives.append((img_path, 0, img_name))
    
    # Select
    positives.sort(key=lambda x: x[1], reverse=True)
    selected_pos = positives[:N_POS]
    selected_neg = random.sample(negatives, min(N_NEG, len(negatives)))
    
    selected = selected_pos + selected_neg
    random.shuffle(selected)
    
    # Copy images and labels
    for img_path, _, img_name in tqdm(selected, desc="Copying"):
        # Copy image
        img = cv2.imread(str(img_path))
        ext = Path(img_path).suffix
        out_img_path = IMAGES_OUT / f"{img_name}{ext}"
        cv2.imwrite(str(out_img_path), img)
        
        # Copy or create txt
        txt_path = temp_labels_dir / f"{img_name}.txt"
        out_txt_path = LABELS_OUT / f"{img_name}.txt"
        if txt_path.exists():
            with open(txt_path, 'r') as f_in, open(out_txt_path, 'w') as f_out:
                f_out.write(f_in.read())
        else:
            out_txt_path.touch()  # Empty for negatives
    
    # Clean temp
    temp_labels_dir.parent.rmdir() if temp_labels_dir.parent.exists() else None
    
    # Convert to COCO
    yolo_to_coco(IMAGES_OUT, LABELS_OUT, COCO_OUT)
    
    print(f"Test set created: {len(selected)} images in {OUTPUT_DIR}")
    print(f"COCO: {COCO_OUT}")

if __name__ == "__main__":
    main()