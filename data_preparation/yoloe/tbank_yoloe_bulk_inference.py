import os
import numpy as np
import torch
import json
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import ops
import cv2
from pycocotools import mask as cocomask  # pip install pycocotools

# Конфигурация
MODEL_PATH = 'yoloe-11l-seg.pt'  # Предобученная YOLOE segmentation модель
DATA_DIR = 'data_sirius'  # Папка с изображениями для inference
REFS_COCO_PATH = 'refs_ls_coco.json'  # Label Studio export COCO для references (9 logos с bbox и cls 0-2)
OUTPUT_COCO = 'tbank_detections_coco_3classes.json'  # Выходной COCO для data_sirius
CONF_THRESH = 0.5
IOU_THRESH = 0.7
IMG_SIZE = 640

# 3 класса с descriptive text prompts
CLASS_NAMES = [
    "yellow_shield_black_T: Жёлтый щит с чёрной жирной буквой T, минималистичный, без теней или с глубиной через переходы.",
    "white_shield_black_T: Белый щит с чёрной T, чёткие границы, мягкая тень.",
    "purple_shield_white_T: Фиолетовый щит с белой T, наклонённый, градиентный."
]

# Маппинг filenames refs to expected paths (относительно workspace)
REF_BASE_DIR = 'tbank_official_logos'
REF_FILENAMES = ['logo0.png', 'logo1.png', 'logo2.png', 'logo3.png', 'logo4.png', 'logo5.png', 'logo6.png', 'logo7.png', 'logo8.png']

def load_refs_from_coco(coco_path):
    """Загрузка references из LS COCO: refer_images, bboxes, cls_ids (grouped by class)"""
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    # Dict image_id -> {'file_name': str, 'width': int, 'height': int, 'path': str}
    img_dict = {img['id']: img for img in coco['images']}
    
    # List of (refer_img, bbox_normalized, cls_id) для всех refs
    refer_data = []
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = img_dict[img_id]
        file_name = img_info['file_name']
        if file_name not in REF_FILENAMES:
            continue  # Skip unexpected
        
        ref_path = os.path.join(REF_BASE_DIR, file_name)
        if not os.path.exists(ref_path):
            raise ValueError(f"Ref path {ref_path} не найден.")
        
        # Загрузка изображения
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            raise ValueError(f"Не удалось загрузить {ref_path}.")
        
        # Bbox from LS: xywh, normalize to x1y1x2y2 [0,1]
        x, y, w, h = ann['bbox']
        width, height = img_info['width'], img_info['height']
        x1, y1 = x / width, y / height
        x2, y2 = (x + w) / width, (y + h) / height
        bbox = np.array([x1, y1, x2, y2])
        
        cls_id = ann['category_id']  # 0,1,2
        
        refer_data.append((ref_img, bbox, cls_id))
    
    if len(refer_data) != 9:
        raise ValueError(f"Ожидалось 9 refs, найдено {len(refer_data)}.")
    
    # Group by cls_id для multi per class
    grouped = {0: [], 1: [], 2: []}
    for img, bbox, cls in refer_data:
        grouped[cls].append((img, bbox))
    
    return grouped

def mask_to_rle(mask):
    """RLE для COCO seg"""
    if mask is None or len(mask) == 0:
        return []
    # Assume mask is binary np.array
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def main():
    # Проверки
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Папка {DATA_DIR} не существует.")
    if not os.path.exists(REFS_COCO_PATH):
        raise ValueError(f"LS COCO export {REFS_COCO_PATH} не найден. Создайте проект в Label Studio для refs и экспортируйте.")
    
    # Загрузка модели
    model = YOLO(MODEL_PATH)
    
    # Text prompts
    text_pe = ops.get_text_pe(CLASS_NAMES)
    model.set_classes(CLASS_NAMES, text_pe)
    
    # Visual prompts: collect all refs with their cls
    refer_data = load_refs_from_coco(REFS_COCO_PATH)
    # Flatten to list for YOLOE: all imgs, all bboxes, all cls (repeat cls for each ref)
    refer_images = []
    bboxes = []
    cls_ids = []
    for cls_id, data_list in refer_data.items():
        for img, bbox in data_list:
            refer_images.append(img)
            bboxes.append(bbox)
            cls_ids.append(cls_id)
    
    bboxes = np.array(bboxes)  # (9,4)
    cls_ids = np.array(cls_ids)  # (9,)
    
    visual_prompts = {
        'refer_images': refer_images,
        'bboxes': bboxes,
        'cls': cls_ids
    }
    
    # Predict
    results = model.predict(
        source=DATA_DIR,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        visual_prompts=visual_prompts,
        save_txt=False,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    # COCO export
    coco_data = {
        'info': {'description': 'T-Bank Logo Detections from YOLOE (3 classes)', 'version': '1.0'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': name.split(':')[0].strip()} for i, name in enumerate(CLASS_NAMES)]
    }
    
    img_id = 0
    ann_id = 0
    for r in results:
        img_path = Path(r.path)
        img_name = img_path.name
        coco_data['images'].append({
            'id': img_id,
            'file_name': img_name,
            'width': r.orig_shape[1],
            'height': r.orig_shape[0]
        })
        
        if r.boxes is not None:
            for i, box in enumerate(r.boxes):
                conf = box.conf[i].cpu().item()
                if conf > CONF_THRESH:
                    xyxy = box.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    
                    cls_id = int(box.cls[i].cpu().numpy())
                    
                    seg = None
                    if r.masks is not None and i < len(r.masks):
                        # Extract mask for this detection
                        mask = r.masks.data[i].cpu().numpy() if hasattr(r.masks, 'data') else None
                        if mask is not None:
                            # Resize mask to orig_shape if needed
                            mask = cv2.resize(mask, (r.orig_shape[1], r.orig_shape[0]))
                            seg = mask_to_rle(mask)
                            seg = [seg] if seg else []
                    
                    coco_data['annotations'].append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': cls_id,
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(w * h),
                        'iscrowd': 0,
                        'score': conf,
                        'segmentation': seg
                    })
                    ann_id += 1
        
        img_id += 1
    
    with open(OUTPUT_COCO, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Инференс завершён. COCO: {OUTPUT_COCO}")
    print(f"Изображения: {len(coco_data['images'])}, Детекции: {len(coco_data['annotations'])}")
    print("Примечание: Для запуска создайте LS проект с 3 классами, импортируйте 9 refs, разметьте bbox, экспортируйте как COCO в refs_ls_coco.json.")

if __name__ == "__main__":
    main()