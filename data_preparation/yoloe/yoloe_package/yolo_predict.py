from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
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
    # Create img_dict
    img_dict = {img['id']: img for img in refs_data['images']}
    
    # Use per-image approach for all cases
    img_to_prompts = {}
    for ann in refs_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_dict:
            continue
        if img_id not in img_to_prompts:
            img_info = img_dict[img_id]
            file_name = img_info['file_name']
            if file_name.startswith('images/'):
                file_name = file_name[len('images/'):]
            ref_path = os.path.join(refs_images_dir, "images", file_name)
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                continue
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            img_to_prompts[img_id] = {
                'img': ref_img,
                'width': img_info['width'],
                'height': img_info['height'],
                'bboxes': [],
                'cls': []
            }
        bbox = ann['bbox']
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        pixel_bbox = [x1, y1, x2, y2]
        img_to_prompts[img_id]['bboxes'].append(pixel_bbox)
        img_to_prompts[img_id]['cls'].append(ann['category_id'])

    refer_images = []
    bboxes_list = []
    cls_list_local = []
    for img_id, data in img_to_prompts.items():
        refer_images.append(data['img'])
        bboxes_list.append(np.array(data['bboxes'], dtype=np.float32))
        cls_list_local.append(np.array(data['cls'], dtype=np.int64))
    
    visual_prompts = {
        'refer_images': refer_images,
        'bboxes': bboxes_list,
        'cls': cls_list_local
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


def perform_prediction(model, img_paths, visual_prompts, conf, iou, runs_dir, device):
    """Выполнение предсказания на изображениях (нормализация visual_prompts)."""
    import numpy as np
    import torch

    # --- Prepare visual_prompts for model (keep np.arrays) ---
    prompts = None
    if visual_prompts is not None:
        # Assume it's already correctly structured from load_visual_prompts
        if isinstance(visual_prompts, dict):
            vp = dict(visual_prompts)
            prompts = {}
            # refer_images: keep as-is (list of RGB np.ndarray)
            if "refer_images" in vp and vp["refer_images"] is not None:
                prompts["refer_images"] = vp["refer_images"]

            # bboxes: keep as list of np.array (2D)
            bboxes_input = vp.get("bboxes", None)
            if bboxes_input is not None:
                prompts["bboxes"] = [np.asarray(group, dtype=np.float32) for group in bboxes_input]

            # cls: keep as list of np.array (1D int)
            cls_input = vp.get("cls", None)
            if cls_input is not None:
                prompts["cls"] = [np.asarray(group, dtype=np.int64) for group in cls_input]

            # if prompts ended up empty, set to None
            if not prompts:
                prompts = None
        else:
            print(f"Warning: unexpected visual_prompts type {type(visual_prompts)}, ignoring prompts")
            prompts = None

    # --- Logging for debug (can be removed later) ---
    if prompts is None:
        print("perform_prediction: prompts=None")
    else:
        try:
            print("perform_prediction: prompts keys:", list(prompts.keys()))
            if "bboxes" in prompts:
                total_bboxes = sum(len(group) for group in prompts["bboxes"])
                print("perform_prediction: number of bboxes:", total_bboxes)
        except Exception:
            pass

    # Replicate prompts for each image - use first reference image for all predictions
    if prompts is not None and len(img_paths) > 1:
        # Use only the first reference image for all predictions to avoid complexity
        first_refer = prompts['refer_images'][0] if prompts['refer_images'] else None
        first_bboxes = prompts['bboxes'][0] if prompts['bboxes'] else None
        first_cls = prompts['cls'][0] if prompts['cls'] else None

        if first_refer is not None:
            prompts = {
                'refer_images': [first_refer] * len(img_paths),
                'bboxes': [first_bboxes] * len(img_paths),
                'cls': [first_cls] * len(img_paths)
            }
        else:
            prompts = None

    # --- Call model.predict using the normalized prompts ---
    results = model.predict(
        source=img_paths,
        visual_prompts=prompts,         # <- pass normalized prompts under 'visual_prompts'
        conf=conf,
        iou=iou,
        save_txt=True,
        save=True,
        project=runs_dir,
        name='predict',
        device=device,
        predictor=YOLOEVPSegPredictor
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
    import glob
    import os

    print(f"Starting YOLOE predict on {img_dir}, refs={refs_images_json}, runs_dir={runs_dir}, conf={conf}, iou={iou}, device={device}")

    model = load_model()
    refs_data = load_refs_data(refs_images_json)
    visual_prompts = load_visual_prompts(refs_data, refs_images_dir)
    
    # Logging visual prompts
    if visual_prompts is not None:
        print("visual_prompts keys:", list(visual_prompts.keys()))
        num_refer = len(visual_prompts["refer_images"])
        total_bboxes = sum(len(cls_group) for cls_group in visual_prompts["cls"])
        print(f"Number of refer images: {num_refer}, total bboxes: {total_bboxes}")
    
    model = setup_text_prompts(model)
    
    # Get list of image paths
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    img_paths.sort()
    print(f"Found {len(img_paths)} images to predict")
    
    results = perform_prediction(model, img_paths, visual_prompts, conf, iou, runs_dir, device)

    return results