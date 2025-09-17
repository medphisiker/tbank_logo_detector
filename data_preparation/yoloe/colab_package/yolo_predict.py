from ultralytics import YOLOE
import numpy as np
import json
import torch

from .paths import REFS_LOCAL, RUNS_DIR

def run_yolo_predict(img_dir):
    model = YOLOE('yoloe-11l-seg.pt')
    
    # Load refs for visual prompts
    with open(REFS_LOCAL, 'r') as f:
        refs_data = json.load(f)
    
    # Group bboxes/cls by class (0=purple, 1=white, 2=yellow)
    grouped_bboxes = {}
    grouped_cls = np.array([0,1,2])  # For multi-class
    for ann in refs_data['annotations']:
        cls_id = ann['category_id']
        bbox = ann['bbox']  # [x,y,w,h] -> [x,y,x+w,y+h]
        x1, y1, w, h = bbox
        if cls_id not in grouped_bboxes:
            grouped_bboxes[cls_id] = []
        grouped_bboxes[cls_id].append([x1, y1, x1+w, y1+h])
    
    visual_prompts = {'bboxes': np.array(grouped_bboxes), 'cls': grouped_cls}
    
    # Text prompts matching category ids
    names = ['purple_shield_white_T', 'white_shield_black_T', 'yellow_shield_black_T']
    text_prompts = ['purple shield with white T logo', 'white shield with black T logo', 'yellow shield with black T logo']
    model.set_classes(names, text_prompts)  # Hybrid
    
    results = model.predict(
        source=img_dir,
        visual_prompts=visual_prompts,
        conf=0.5,
        iou=0.7,
        save_txt=True,
        project=RUNS_DIR,
        name='predict',
        recursive=True,
        device='auto'  # Auto-detect GPU/CPU
    )
    print('Prediction complete. Results in ' + RUNS_DIR + '/')
    return results