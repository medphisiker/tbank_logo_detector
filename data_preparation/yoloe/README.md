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
docker run -it --gpus all -v ./data:/data -v ./data_preparation/yoloe:/app tbank-yoloe-ultralytics
```

- Volume: `./data:/data` для подключения данных для обработки и сохранения результатов обработки.
- Volume: `./data_preparation/yoloe:/app` для подключения папки содержащей код ля работы `YOLOE`:
1. скрипт `tbank_yoloe_bulk_inference.py` для запуска обработки.
2. python package `yoloe_package` содержаший удобный интерфейс для использования `YOLOE` для few-shot разметки данных.
- Результаты в `data/yoloe_results/`.
3. в эту же папку будут сохраняться веса моделей `model_cache`, необходимые для работы пайплайна.

## Скрипт запуска

Используйте `tbank_yoloe_bulk_inference.py` с config.json для параметров (input_dir, refs_json, output_dir, subset, conf, iou, runs_dir, device).

Пример config.json (для Docker используй абсолютные пути):
```json
{
  "input_dir": "/data/data_sirius/images",
  "refs_json": "/data/tbank_official_logos/refs_ls_coco.json",
  "output_dir": "/data/yoloe_results",
  "subset": 10,
  "conf": 0.5,
  "iou": 0.7,
  "runs_dir": "runs/yoloe_predict",
  "device": "0",
  "weights_dir": "./ultralytics_weights"
}
```

Запуск в контейнере:
```
docker run -it --gpus all -v ./data:/data -v ./data_preparation/yoloe:/app tbank-yoloe-ultralytics python tbank_yoloe_bulk_inference.py
```

## Выходные файлы

- `data/yoloe_results/pseudo_coco.json`: Псевдо-аннотации.
- `data/yoloe_results/runs_yoloe.zip`: Labels и визуализации.

## Отладка

Интерактивно:
```
docker run -it --gpus all -v ./data:/data -v ./data_preparation/yoloe:/app tbank-yoloe-ultralytics /bin/bash
```
Затем внутри контейнера:
```
python tbank_yoloe_bulk_inference.py
```

Пакет yoloe_package в образе, зависимости через uv.