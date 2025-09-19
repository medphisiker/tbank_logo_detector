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

    # Export COCO
    pseudo_coco = os.path.join(params["output_dir"], "pseudo_coco.json")
    labels_dir = os.path.join(params["runs_dir"], "predict", "labels")
    export_to_coco(img_dir, labels_dir, pseudo_coco)

    # Save results
    save_results(params["output_dir"], params["runs_dir"])

    return params["output_dir"]