import os
import json
from PIL import Image

from .paths import PSEUDO_COCO, LABELS_DIR

def export_to_coco(img_dir):
    names = ['yellow_shield_black_T', 'white_shield_black_T', 'purple_shield_white_T']
    coco = {
        'info': {'description': 'YOLOE pseudo labels for all datasets'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': i+1, 'name': name} for i, name in enumerate(names)]
    }
    image_id = 0
    ann_id = 0
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, img_dir)
            # Add image
            with Image.open(img_path) as img:
                w, h = img.size
            coco['images'].append({'id': image_id, 'file_name': rel_path, 'width': w, 'height': h})
            # Load txt if exists (mirrors structure)
            txt_file = file.rsplit('.',1)[0] + '.txt'
            rel_txt = os.path.relpath(os.path.join(root, txt_file), img_dir)
            txt_path = os.path.join(LABELS_DIR, rel_txt)
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