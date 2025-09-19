import os
import json
import cv2
from ultralytics import YOLOE
from yoloe_package.config import load_config
from yoloe_package.prepare_data import prepare_data
import glob

def run_text_prompt_detection():
    """Запуск обнаружения объектов по текстовым промптам.

    Использует YOLOE с текстовыми промптами для обнаружения всех объектов,
    сохраняет результаты в JSON с уверенностью модели.
    """
    # Загрузка конфигурации
    config = load_config()

    # Подготовка данных (используем существующие функции)
    input_dir = config.get("input_dir", "/data/data_sirius/images")
    subset = config.get("subset")
    img_dir = prepare_data(input_dir, subset)

    # Инициализация модели YOLOE
    model = YOLOE("yoloe-11l-seg.pt")

    # Текстовые промпты для обнаружения объектов
    text_prompts = [
        "logo", "brand logo", "shield", "emblem", "text logo",
        "yellow shield with black T", "white shield with black T", "purple shield with white T",
        "object", "item", "symbol", "icon", "badge", "insignia"
    ]

    # Установка классов и текстовых эмбеддингов
    model.set_classes(text_prompts, model.get_text_pe(text_prompts))

    # Получение списка изображений
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    img_paths.sort()

    print(f"Found {len(img_paths)} images for text prompt detection")

    # Параметры из конфига
    conf = config.get("conf", 0.18)  # Низкий порог для обнаружения всего
    iou = config.get("iou", 0.7)
    device = config.get("device", "auto")

    # Выходной файл
    output_file = "text_prompt_detections.json"

    results_all = []

    # Обработка каждого изображения
    for img_path in img_paths:
        print(f"Processing {img_path}")

        # Предсказание
        results = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            device=device,
            save=False  # Не сохраняем визуализации
        )

        # Извлечение результатов
        detections = []
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                scores = result.boxes.conf.cpu().numpy()  # уверенность
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # ID класса

                for box, score, cls_id in zip(boxes, scores, class_ids):
                    detection = {
                        "box": box.tolist(),
                        "score": float(score),
                        "class_id": int(cls_id),
                        "class_name": text_prompts[cls_id] if cls_id < len(text_prompts) else "unknown"
                    }
                    detections.append(detection)

        # Сохранение результатов для изображения
        img_result = {
            "image_path": img_path,
            "detections": detections
        }
        results_all.append(img_result)

    # Сохранение всех результатов в JSON
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)

    print(f"Text prompt detection completed. Results saved to {output_file}")
    print(f"Total images processed: {len(results_all)}")
    total_detections = sum(len(img['detections']) for img in results_all)
    print(f"Total detections: {total_detections}")

if __name__ == "__main__":
    run_text_prompt_detection()