import os
import cv2
import json
from pathlib import Path
from operator import itemgetter
from yoloe_package.config import load_config

def extract_tbank_crops():
    """Извлечение кропов логотипов T-Bank из результатов bulk inference.

    Читает результаты из runs_dir, извлекает кропы, сортирует по уверенности
    и сохраняет в папку tbank_crops.
    """
    # Загрузка конфигурации
    config = load_config()
    runs_dir = config.get("runs_dir", "runs/yoloe_predict")
    output_dir = config.get("output_dir", "/data/yoloe_results")

    # Поиск последней папки predict
    predict_dirs = sorted([d for d in os.listdir(runs_dir) if d.startswith('predict')],
                         key=lambda x: int(x.split('predict')[1] or '0'), reverse=True)

    if not predict_dirs:
        print("No predict directories found")
        return

    latest_predict = os.path.join(runs_dir, predict_dirs[0])
    labels_dir = os.path.join(latest_predict, "labels")

    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return

    # Папка для кропов
    crops_dir = "tbank_crops"
    os.makedirs(crops_dir, exist_ok=True)

    # Сбор всех детекций
    detections = []

    # Чтение всех txt файлов с результатами
    for txt_file in Path(labels_dir).glob("*.txt"):
        img_name = txt_file.stem
        img_path = None

        # Поиск соответствующего изображения
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(output_dir, "annotated_images", f"{img_name}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            print(f"Image not found for {img_name}")
            continue

        # Загрузка изображения
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        # Чтение детекций из txt файла
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])

                # YOLO формат: class x_center y_center width height [confidence]
                if len(parts) == 5:  # без confidence
                    x_center, y_center, width, height = map(float, parts[1:])
                    conf = 1.0
                else:  # с confidence
                    x_center, y_center, width, height, conf = map(float, parts[1:])

                # Конвертация в пиксели
                x1 = int((x_center - width/2) * W)
                y1 = int((y_center - height/2) * H)
                x2 = int((x_center + width/2) * W)
                y2 = int((y_center + height/2) * H)

                # Проверка границ
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                # Проверка минимальной площади
                area = (x2 - x1) * (y2 - y1)
                if area < 100:  # слишком маленький кроп
                    continue

                detections.append({
                    "img_path": img_path,
                    "box": [x1, y1, x2, y2],
                    "score": conf,
                    "class_id": cls_id,
                    "img_name": img_name
                })

    print(f"Found {len(detections)} detections")

    # Сортировка по уверенности (по убыванию)
    detections_sorted = sorted(detections, key=itemgetter("score"), reverse=True)

    # Извлечение и сохранение кропов
    for i, det in enumerate(detections_sorted):
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        img_name = det["img_name"]

        # Загрузка изображения
        img = cv2.imread(det["img_path"])
        if img is None:
            continue

        # Извлечение кропа
        crop = img[y1:y2, x1:x2]

        # Сохранение кропа
        crop_filename = "03d"
        crop_path = os.path.join(crops_dir, crop_filename)
        cv2.imwrite(crop_path, crop)

        if i < 10 or (i % 100 == 0):  # Логирование первых 10 и каждые 100
            print(".3f")

    print(f"Extracted {len(detections_sorted)} crops to {crops_dir}")
    print(f"Top crop score: {detections_sorted[0]['score']:.3f}" if detections_sorted else "No detections")

if __name__ == "__main__":
    extract_tbank_crops()