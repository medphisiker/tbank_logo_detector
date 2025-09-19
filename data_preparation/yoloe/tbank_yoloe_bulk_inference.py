import json
import os
from yoloe_package import prepare_data, run_yolo_predict, export_to_coco, save_results
from ultralytics import settings

def load_config():
    """Загрузка конфигурации из JSON файла.

    Returns
    -------
    dict
        Словарь с конфигурационными параметрами.
    """
    config_path = os.getenv("CONFIG_PATH", "/app/config.json")
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    print("Config loaded")
    return config


def setup_weights_dir(config):
    """Настройка директории для весов модели.

    Parameters
    ----------
    config : dict
        Конфигурационный словарь.

    Returns
    -------
    str
        Путь к директории с весами.
    """
    weights_dir = config.get("weights_dir", "./ultralytics_weights")
    settings["weights_dir"] = weights_dir
    print(settings)
    print(f"Weights dir set to {weights_dir}")
    return weights_dir


def load_parameters(config):
    """Загрузка параметров из конфигурации.

    Parameters
    ----------
    config : dict
        Конфигурационный словарь.

    Returns
    -------
    dict
        Словарь с параметрами для инференса.
    """
    params = {
        "input_dir": config.get("input_dir", "/data/data_sirius/images"),
        "refs_images_json": config.get("refs_images_json", "/data/tbank_official_logos/refs_ls_coco.json"),
        "output_dir": config.get("output_dir", "/data/yoloe_results"),
        "subset": config.get("subset"),
        "conf": config.get("conf", 0.5),
        "iou": config.get("iou", 0.7),
        "runs_dir": config.get("runs_dir", "runs/yoloe_predict"),
        "device": config.get("device", "auto"),
        "refs_images_dir": config.get("refs_images_dir", "/data/tbank_official_logos/images")
    }
    return params


def create_directories(output_dir, runs_dir):
    """Создание необходимых директорий.

    Parameters
    ----------
    output_dir : str
        Путь к выходной директории.
    runs_dir : str
        Путь к директории для результатов прогонов.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)


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


def main():
    """Главная функция для запуска bulk инференса YOLOE.

    Загружает конфигурацию, настраивает параметры,
    создает директории и запускает пайплайн инференса.
    """
    config = load_config()
    setup_weights_dir(config)
    params = load_parameters(config)
    create_directories(params["output_dir"], params["runs_dir"])
    output_dir = run_inference_pipeline(params)
    print(f"Bulk inference completed. Results in {output_dir}")


if __name__ == "__main__":
    main()
