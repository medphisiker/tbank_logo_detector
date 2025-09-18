import json
import os
from yoloe_package import prepare_data, run_yolo_predict, export_to_coco, save_results

def main():
    config_path = os.getenv('CONFIG_PATH', '/app/yoloe/config.json')
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Config loaded")

    input_dir = config.get('input_dir', '/data/data_sirius/images')
    refs_json = config.get('refs_json', 'data/tbank_official_logos/refs_ls_coco.json')
    output_dir = config.get('output_dir', '/data/yoloe_results')
    subset = config.get('subset')
    conf = config.get('conf', 0.5)
    iou = config.get('iou', 0.7)
    runs_dir = config.get('runs_dir', 'runs/yoloe_predict')
    device = config.get('device', 'auto')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Prepare data
    img_dir = prepare_data(input_dir, subset)

    # Run prediction
    results = run_yolo_predict(img_dir, refs_json, runs_dir, conf, iou, device)

    # Export COCO
    pseudo_coco = os.path.join(output_dir, 'pseudo_coco.json')
    labels_dir = os.path.join(runs_dir, 'predict', 'labels')
    export_to_coco(img_dir, labels_dir, pseudo_coco)

    # Save results
    save_results(output_dir, runs_dir)

    print(f"Bulk inference completed. Results in {output_dir}")

if __name__ == "__main__":
    main()