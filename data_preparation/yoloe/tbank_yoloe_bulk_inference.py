import argparse
import os
from pathlib import Path
from yoloe_package import prepare_data, run_yolo_predict, export_to_coco, save_results

def main():
    parser = argparse.ArgumentParser(description='YOLOE Bulk Inference for T-Bank Logos')
    parser.add_argument('--input_dir', type=str, default='data/data_sirius/images', help='Input images directory')
    parser.add_argument('--refs_json', type=str, default='data/tbank_official_logos/refs_ls_coco.json', help='Reference COCO JSON')
    parser.add_argument('--output_dir', type=str, default='yoloe_results', help='Output directory')
    parser.add_argument('--subset', type=int, default=None, help='Subset number of images (None for all)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IOU threshold')
    parser.add_argument('--runs_dir', type=str, default='runs/yoloe_predict', help='Runs directory for labels')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, 0)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    # Prepare data
    img_dir = prepare_data(args.input_dir, args.subset)

    # Run prediction
    results = run_yolo_predict(img_dir, args.refs_json, args.runs_dir, args.conf, args.iou, args.device)

    # Export COCO
    export_to_coco(img_dir, args.runs_dir, args.output_dir + '/pseudo_coco.json')

    # Save results
    save_results(args.output_dir, args.runs_dir)

    print(f"Bulk inference completed. Results in {args.output_dir}")

if __name__ == "__main__":
    main()