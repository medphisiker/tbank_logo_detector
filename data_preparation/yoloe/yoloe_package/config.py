import json
import os
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
        "refs_images_dir": config.get("refs_images_dir", "/data/tbank_official_logos/images"),
        "batch_size": config.get("batch_size", 1),
        "imgsz": config.get("imgsz", 640),
        "half": config.get("half", False),
        "save_visualizations": config.get("save_visualizations", True),
        "use_sahi": config.get("use_sahi", False),
        "sahi_slice_height": config.get("sahi_slice_height", 512),
        "sahi_slice_width": config.get("sahi_slice_width", 512),
        "sahi_overlap_height_ratio": config.get("sahi_overlap_height_ratio", 0.2),
        "sahi_overlap_width_ratio": config.get("sahi_overlap_width_ratio", 0.2)
    }
    return params