import os
import cv2
import json
from pathlib import Path
from operator import itemgetter
from yoloe_package.config import load_config

def calculate_iou(box1, box2):
    """Расчет IoU между двумя bounding box'ами.

    Parameters
    ----------
    box1 : list
        [x1, y1, x2, y2]
    box2 : list
        [x1, y1, x2, y2]

    Returns
    -------
    float
        IoU значение (0.0 до 1.0)
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    # Координаты пересечения
    xa = max(x11, x21)
    ya = max(y11, y21)
    xb = min(x12, x22)
    yb = min(y12, y22)

    # Площадь пересечения
    inter_area = max(0, xb - xa) * max(0, yb - ya)

    # Площади боксов
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    # IoU
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area

def extract_background_objects():
    """Извлечение non-overlapping объектов для фона.

    Находит объекты из text_prompt_detections.json, которые не пересекаются
    с T-Bank логотипами по IoU < 0.3, извлекает кропы и сортирует по уверенности.
    """
    # Загрузка конфигурации
    config = load_config()
    output_dir = config.get("output_dir", "/data/yoloe_results")

    # Загрузка результатов text prompt detection (Шаг 1)
    text_detections_file = "text_prompt_detections.json"
    if not os.path.exists(text_detections_file):
        print(f"Text detections file not found: {text_detections_file}")
        print("Please run text_prompt_detection.py first")
        return

    with open(text_detections_file, 'r', encoding='utf-8') as f:
        text_results = json.load(f)

    print(f"Loaded {len(text_results)} images from text detections")

    # Загрузка T-Bank detections из pseudo_coco.json
    tbank_coco_file = os.path.join(output_dir, "pseudo_coco.json")
    tbank_boxes = {}

    if os.path.exists(tbank_coco_file):
        with open(tbank_coco_file, 'r', encoding='utf-8') as f:
            tbank_data = json.load(f)

        # Группировка T-Bank боксов по изображениям
        img_dict = {img['id']: img for img in tbank_data['images']}
        for ann in tbank_data['annotations']:
            img_id = ann['image_id']
            if img_id in img_dict:
                img_file = img_dict[img_id]['file_name']
                if img_file not in tbank_boxes:
                    tbank_boxes[img_file] = []
                bbox = ann['bbox']
                # Конвертация из COCO [x, y, w, h] в [x1, y1, x2, y2]
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                tbank_boxes[img_file].append([x1, y1, x2, y2])

        print(f"Loaded T-Bank detections for {len(tbank_boxes)} images")
    else:
        print(f"T-Bank detections file not found: {tbank_coco_file}")
        print("Using empty T-Bank boxes (all objects will be considered non-overlapping)")

    # Папка для кропов фона
    backgrounds_dir = "backgrounds_obj"
    os.makedirs(backgrounds_dir, exist_ok=True)

    # Сбор non-overlapping детекций
    non_overlapping = []

    for img_result in text_results:
        img_path = img_result['image_path']
        img_name = os.path.basename(img_path)

        # Получение T-Bank боксов для этого изображения
        tbank_img_boxes = tbank_boxes.get(img_name, [])

        # Загрузка изображения для проверки размеров
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        for det in img_result['detections']:
            box = det['box']  # [x1, y1, x2, y2]
            score = det['score']

            # Проверка площади
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / (W * H)
            if area_ratio < 0.005 or area_ratio > 0.6:  # фильтр по площади
                continue

            # Проверка IoU с T-Bank боксами
            is_overlapping = False
            for tbank_box in tbank_img_boxes:
                iou = calculate_iou(box, tbank_box)
                if iou >= 0.3:  # порог IoU
                    is_overlapping = True
                    break

            if not is_overlapping:
                non_overlapping.append({
                    "img_path": img_path,
                    "box": box,
                    "score": score,
                    "class_name": det.get('class_name', 'unknown'),
                    "img_name": img_name
                })

    print(f"Found {len(non_overlapping)} non-overlapping detections")

    # Сортировка по уверенности (по убыванию)
    non_overlapping_sorted = sorted(non_overlapping, key=itemgetter("score"), reverse=True)

    # Извлечение и сохранение кропов
    for i, det in enumerate(non_overlapping_sorted):
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        class_name = det["class_name"]
        img_name = det["img_name"]

        # Загрузка изображения
        img = cv2.imread(det["img_path"])
        if img is None:
            continue

        # Проверка границ
        H, W = img.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # Извлечение кропа
        crop = img[y1:y2, x1:x2]

        # Сохранение кропа
        crop_filename = "03d"
        crop_path = os.path.join(backgrounds_dir, crop_filename)
        cv2.imwrite(crop_path, crop)

        if i < 10 or (i % 100 == 0):  # Логирование первых 10 и каждые 100
            print(".3f")

    print(f"Extracted {len(non_overlapping_sorted)} background object crops to {backgrounds_dir}")
    if non_overlapping_sorted:
        print(f"Top background object score: {non_overlapping_sorted[0]['score']:.3f}")

if __name__ == "__main__":
    extract_background_objects()