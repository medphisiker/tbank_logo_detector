import os
from .prepare_data import prepare_data
from .yolo_predict import run_yolo_predict
from .export_coco import export_to_coco
from .save_results import save_results

def run_inference_pipeline(params):
    """Запуск основного пайплайна инференса.

    Parameters
    ----------
    params : dict
        Словарь с параметрами для инференса.

    Returns
    -------
    str
        Путь к выходной директории с результатами.
    """
    # Prepare data
    img_dir = prepare_data(params["input_dir"], params["subset"])

    # Run prediction
    results = run_yolo_predict(
        img_dir,
        params["refs_images_json"],
        params["runs_dir"],
        params["conf"],
        params["iou"],
        params["device"],
        params["refs_images_dir"]
    )

    # Find the latest predict run dir
    predict_dirs = sorted([d for d in os.listdir(params["runs_dir"]) if d.startswith('predict')], key=lambda x: int(x.split('predict')[1] or '0'), reverse=True)
    if predict_dirs:
        latest_predict = os.path.join(params["runs_dir"], predict_dirs[0])
        labels_dir = os.path.join(latest_predict, "labels")
        save_dir = os.path.join(latest_predict, "save")
    else:
        labels_dir = os.path.join(params["runs_dir"], "predict", "labels")
        save_dir = os.path.join(params["runs_dir"], "predict", "save")

    # Export COCO
    pseudo_coco = os.path.join(params["output_dir"], "pseudo_coco.json")
    export_to_coco(img_dir, labels_dir, pseudo_coco)

    # Copy annotated images
    if os.path.exists(save_dir):
        import shutil
        annotated_dir = os.path.join(params["output_dir"], "annotated_images")
        shutil.copytree(save_dir, annotated_dir, dirs_exist_ok=True)
        print(f"Annotated images copied to {annotated_dir}")

    # Save results
    save_results(params["output_dir"], params["runs_dir"])

    return params["output_dir"]