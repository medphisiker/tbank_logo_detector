from ultralytics import YOLOE
import numpy as np
import json
import cv2
import os

def load_model():
    """Загрузка YOLOE модели.

    Returns
    -------
    YOLOE
        Загруженная модель YOLOE.
    """
    model = YOLOE('yoloe-11l-seg.pt')
    print("Model loaded")
    return model


def load_refs_data(refs_images_json):
    """Загрузка данных референсов из JSON файла.

    Parameters
    ----------
    refs_images_json : str
        Путь к JSON файлу с данными референсов.

    Returns
    -------
    dict
        Словарь с данными референсов в формате COCO.
    """
    with open(refs_images_json, 'r') as f:
        refs_data = json.load(f)
    return refs_data


def load_visual_prompts(refs_data, refs_images_dir):
    """Загрузка визуальных промптов из референсных изображений.

    Parameters
    ----------
    refs_data : dict
        Данные референсов в формате COCO.
    refs_images_dir : str
        Путь к директории с референсными изображениями.

    Returns
    -------
    dict
        Словарь с визуальными промптами для модели.
    """
    # Create img_dict for normalization
    img_dict = {img['id']: img for img in refs_data['images']}
    
    # Group refer_images, bboxes/cls by class (0=purple, 1=white, 2=yellow)
    grouped_refs = {0: [], 1: [], 2: []}
    for ann in refs_data['annotations']:
        cls_id = ann['category_id']
        img_id = ann['image_id']
        if img_id not in img_dict:
            continue
        img_info = img_dict[img_id]
        file_name = img_info['file_name']
        ref_path = os.path.join(refs_images_dir, file_name)
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            continue
        width = img_info['width']
        height = img_info['height']
        bbox = ann['bbox']  # [x,y,w,h]
        x1, y1, bw, bh = bbox
        x1_norm = x1 / width
        y1_norm = y1 / height
        x2_norm = (x1 + bw) / width
        y2_norm = (y1 + bh) / height
        bbox_norm = [x1_norm, y1_norm, x2_norm, y2_norm]
        grouped_refs[cls_id].append((ref_img, bbox_norm))
    
    # Flatten for visual prompts
    refer_images = []
    bboxes = []
    cls_ids = []
    for cls_id in sorted(grouped_refs.keys()):
        for ref_img, bbox_norm in grouped_refs[cls_id]:
            refer_images.append(ref_img)
            bboxes.append(bbox_norm)
            cls_ids.append(cls_id)
    
    visual_prompts = {
        'refer_images': refer_images,
        'bboxes': np.array(bboxes, dtype=np.float32),
        'cls': np.array(cls_ids, dtype=np.int64)
    }
    return visual_prompts


def setup_text_prompts(model):
    """Настройка текст промптов и классов.

    Parameters
    ----------
    model : YOLOE
        Модель YOLOE для настройки.

    Returns
    -------
    YOLOE
        Модель с настроенными классами и текст промптами.
    """
    names = ['purple_shield_white_T', 'white_shield_black_T', 'yellow_shield_black_T']
    text_prompts = ['purple shield with white T logo', 'white shield with black T logo', 'yellow shield with black T logo']
    print("Generating text embeddings")
    text_pe = model.get_text_pe(text_prompts)
    print("Setting classes with text embeddings")
    model.set_classes(names, text_pe)  # Hybrid
    return model


def perform_prediction(model, img_dir, visual_prompts, conf, iou, runs_dir, device):
    """Выполнение предсказания на изображениях.

    Parameters
    ----------
    model : YOLOE
        Настроенная модель YOLOE.
    img_dir : str
        Путь к директории с изображениями для предсказания.
    visual_prompts : dict
        Визуальные промпты для модели.
    conf : float
        Порог уверенности для предсказаний.
    iou : float
        Порог IoU для NMS.
    runs_dir : str
        Путь к директории для сохранения результатов.
    device : str
        Устройство для выполнения (cpu/cuda).

    Returns
    -------
    list
        Результаты предсказаний.
    """
    results = model.predict(
        source=img_dir,
        visual_prompts=visual_prompts,
        conf=conf,
        iou=iou,
        save_txt=True,
        project=runs_dir,
        name='predict',
        device=device
    )
    print(f'Prediction complete. Results in {runs_dir}/')
    return results


def run_yolo_predict(img_dir, refs_images_json, runs_dir, conf=0.5, iou=0.7, device='auto', refs_images_dir='/data/tbank_official_logos/images'):
    """Запуск предсказания YOLOE с визуальными промптами.

    Parameters
    ----------
    img_dir : str
        Путь к директории с изображениями для предсказания.
    refs_images_json : str
        Путь к JSON файлу с данными референсов.
    runs_dir : str
        Путь к директории для сохранения результатов.
    conf : float, optional
        Порог уверенности для предсказаний (default: 0.5).
    iou : float, optional
        Порог IoU для NMS (default: 0.7).
    device : str, optional
        Устройство для выполнения (default: 'auto').
    refs_images_dir : str, optional
        Путь к директории с референсными изображениями (default: '/data/tbank_official_logos/images').

    Returns
    -------
    list
        Результаты предсказаний YOLOE.
    """
    print(f"Starting YOLOE predict on {img_dir}, refs={refs_images_json}, runs_dir={runs_dir}, conf={conf}, iou={iou}, device={device}")

    model = load_model()
    refs_data = load_refs_data(refs_images_json)
    visual_prompts = load_visual_prompts(refs_data, refs_images_dir)
    model = setup_text_prompts(model)
    results = perform_prediction(model, img_dir, visual_prompts, conf, iou, runs_dir, device)

    return results