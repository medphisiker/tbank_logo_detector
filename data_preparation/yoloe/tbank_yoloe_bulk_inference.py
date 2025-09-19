import json
import os
from yoloe_package import prepare_data, run_yolo_predict, export_to_coco, save_results
from ultralytics import settings

def main():
    config_path = os.getenv("CONFIG_PATH", "/app/config.json")
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    print("Config loaded")

    weights_dir = config.get("weights_dir", "./ultralytics_weights")
    
    settings["weights_dir"] = weights_dir
    print(settings)
    print(f"Weights dir set to {weights_dir}")

    input_dir = config.get("input_dir", "/data/data_sirius/images")
    refs_images_json = config.get("refs_images_json", "/data/tbank_official_logos/refs_ls_coco.json")
    output_dir = config.get("output_dir", "/data/yoloe_results")
    subset = config.get("subset")
    conf = config.get("conf", 0.5)
    iou = config.get("iou", 0.7)
    runs_dir = config.get("runs_dir", "runs/yoloe_predict")
    device = config.get("device", "auto")
    refs_images_dir = config.get("refs_images_dir", "/data/tbank_official_logos/images")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Prepare data
    img_dir = prepare_data(input_dir, subset)

    # Run prediction
    results = run_yolo_predict(img_dir, refs_images_json, runs_dir, conf, iou, device, refs_images_dir)

    # Export COCO
    pseudo_coco = os.path.join(output_dir, "pseudo_coco.json")
    labels_dir = os.path.join(runs_dir, "predict", "labels")
    export_to_coco(img_dir, labels_dir, pseudo_coco)

    # Save results
    save_results(output_dir, runs_dir)

    print(f"Bulk inference completed. Results in {output_dir}")


if __name__ == "__main__":
    main()
