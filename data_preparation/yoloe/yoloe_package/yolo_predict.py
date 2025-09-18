from ultralytics import YOLOE
import numpy as np
import json

def run_yolo_predict(img_dir, refs_json, runs_dir, conf=0.5, iou=0.7, device='auto'):
    print(f"Starting YOLOE predict on {img_dir}, refs={refs_json}, runs_dir={runs_dir}, conf={conf}, iou={iou}, device={device}")
    model = YOLOE('yoloe-11l-seg.pt')
    print("Model loaded")
    
    # Load refs for visual prompts
    with open(refs_json, 'r') as f:
        refs_data = json.load(f)
    
    # Create img_dict for normalization
    img_dict = {img['id']: img for img in refs_data['images']}
    
    # Group bboxes/cls by class (0=purple, 1=white, 2=yellow)
    grouped_bboxes = {}
    grouped_cls = np.array([0,1,2])  # For multi-class
    for ann in refs_data['annotations']:
        cls_id = ann['category_id']
        img_id = ann['image_id']
        if img_id not in img_dict:
            continue
        img_info = img_dict[img_id]
        width = img_info['width']
        height = img_info['height']
        bbox = ann['bbox']  # [x,y,w,h]
        x1, y1, bw, bh = bbox
        x1_norm = x1 / width
        y1_norm = y1 / height
        x2_norm = (x1 + bw) / width
        y2_norm = (y1 + bh) / height
        if cls_id not in grouped_bboxes:
            grouped_bboxes[cls_id] = []
        grouped_bboxes[cls_id].append([x1_norm, y1_norm, x2_norm, y2_norm])
    
    visual_prompts = {'bboxes': grouped_bboxes, 'cls': grouped_cls}
    
    # Text prompts matching category ids
    names = ['purple_shield_white_T', 'white_shield_black_T', 'yellow_shield_black_T']
    text_prompts = ['purple shield with white T logo', 'white shield with black T logo', 'yellow shield with black T logo']
    print("Generating text embeddings")
    text_pe = model.get_text_pe(text_prompts)
    print("Setting classes with text embeddings")
    model.set_classes(names, text_pe)  # Hybrid
    
    results = model.predict(
        source=img_dir,
        visual_prompts=visual_prompts,
        conf=conf,
        iou=iou,
        save_txt=True,
        project=runs_dir,
        name='predict',
        recursive=True,
        device=device
    )
    print(f'Prediction complete. Results in {runs_dir}/')
    return results