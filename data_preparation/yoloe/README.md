# YOLOE Docker Setup

## Обзор

Этот Dockerfile позволяет запустить пакет YOLOE для автоматической разметки логотипов Т-Банка в изолированной среде. Использует uv для управления зависимостями, основан на nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 с Python 3.13. По умолчанию запускает tbank_yoloe_bulk_inference.py с config.json.

## Требования

- Docker установлен.
- NVIDIA Docker для GPU.
- Данные в `data/data_sirius/images/` и `data/tbank_official_logos/`.

## Сборка образа

Из корня проекта:

```
docker build -f data_preparation/yoloe/ultralytics_dockerfile -t tbank-yoloe-ultralytics data_preparation/yoloe
```

## Запуск контейнера

Из корня проекта:

```
docker run --gpus all -v ./data:/app/data tbank-yoloe
```

- Volume: `./data:/app/data`.
- Результаты в `data/yoloe_results/`.

## Скрипт запуска

Используйте `tbank_yoloe_bulk_inference.py` с config.json для параметров (input_dir, refs_json, output_dir, subset, conf, iou, runs_dir, device).

Пример config.json:
```json
{
  "input_dir": "data/data_sirius/images",
  "refs_json": "data/tbank_official_logos/refs_ls_coco.json",
  "output_dir": "yoloe_results",
  "subset": null,
  "conf": 0.5,
  "iou": 0.7,
  "runs_dir": "runs/yoloe_predict",
  "device": "auto"
}
```

Запуск в контейнере:
```
docker run --gpus all -v ./data:/data -v ./ultralytics_cache/weights:/ultralytics/weights -v ./ultralytics_cache/runs:/ultralytics/runs tbank-yoloe-ultralytics
```

## Настройка

- GPU: --gpus all.
- CPU: docker run -v ./data:/app/data tbank-yoloe.

## Выходные файлы

- `data/yoloe_results/pseudo_coco.json`: Псевдо-аннотации.
- `data/yoloe_results/runs_yoloe.zip`: Labels и визуализации.

## Отладка

Интерактивно:
```
docker run -it --gpus all -v ./data:/app/data tbank-yoloe /bin/bash
```
Затем внутри контейнера:
```
cd /app/yoloe
python tbank_yoloe_bulk_inference.py
```

Пакет yoloe_package в образе, зависимости через uv.