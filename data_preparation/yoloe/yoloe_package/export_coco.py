import os
import json
from PIL import Image

def export_to_coco(img_dir, labels_dir, output_path):
    print(f"Exporting COCO from {img_dir}, labels_dir={labels_dir}, output={output_path}")
    names = ['purple_shield_white_T', 'white_shield_black_T', 'yellow_shield_black_T']
    coco = {
        'info': {'description': 'YOLOE pseudo labels for tbank data_sirius'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': name} for i, name in enumerate(names)]
    }
    image_id = 0
    ann_id = 0
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, img_dir)
            with Image.open(img_path) as img:
                w, h = img.size
            coco['images'].append({'id': image_id, 'file_name': rel_path, 'width': w, 'height': h})
            txt_file = file.rsplit('.',1)[0] + '.txt'
            rel_txt = os.path.relpath(os.path.join(root, txt_file), img_dir)
            txt_path = os.path.join(labels_dir, rel_txt)
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cls_id = int(parts[0])
                        points_str = parts[1:]
                        if len(parts) % 2 == 0:  # even total, has conf (class + even coords + conf)
                            conf = float(points_str[-1])
                            points_str = points_str[:-1]
                        else:  # odd total, no conf (class + even coords)
                            conf = 1.0
                        if len(points_str) % 2 != 0 or len(points_str) < 4:
                            continue
                        points = list(map(float, points_str))
                        xs = points[0::2]
                        ys = points[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        x = x_min * w
                        y = y_min * h
                        w_ann = (x_max - x_min) * w
                        h_ann = (y_max - y_min) * h
                        coco['annotations'].append({
                            'id': ann_id,
                            'image_id': image_id,
                            'category_id': cls_id,
                            'bbox': [x, y, w_ann, h_ann],
                            'area': w_ann * h_ann,
                            'iscrowd': 0,
                            'score': conf
                        })
                        ann_id += 1
            image_id += 1
    
    with open(output_path, 'w') as f:
        json.dump(coco, f)
    print(f'Exported to {output_path}')