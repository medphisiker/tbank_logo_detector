# YOLOE Docker Setup

## Обзор

Используйте Docker контейнер для запуска YOLOE - модели для автоматической разметки логотипов Т-Банка. Контейнер основан на ultralytics/ultralytics:latest с предустановленными зависимостями Ultralytics.

### Примеры результатов

Ниже приведены примеры изображений с визуализацией найденных логотипов Т-Банка с помощью YOLOE:

![Пример 1: logo3.jpg](../../data/yoloe_results/annotated_images/logo3.jpg)

![Пример 2: logo4.jpg](../../data/yoloe_results/annotated_images/logo4.jpg)

![Пример 3: logo6.jpg](../../data/yoloe_results/annotated_images/logo6.jpg)

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
docker run -it --gpus all -v ./data:/data -v ./data_preparation/yoloe:/app tbank-yoloe-ultralytics python tbank_yoloe_bulk_inference.py
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

**Примечание по weights_dir**: Из-за бага в ultralytics, веса моделей скачиваются в /app (в Docker-контейнере), независимо от указанного weights_dir. В локальной среде — в рабочую директорию (data_preparation/yoloe). При монтировании -v ./data_preparation/yoloe:/app веса сохраняются локально в data_preparation/yoloe.

**device**: "0" для GPU 0 (рекомендуется в Docker с --gpus all), "cpu" для CPU. "auto" может не работать — используйте явный номер GPU или "cpu".

**subset**: null для полного датасета или число (e.g. 10) для подмножества первых N изображений (копируются в поддиректорию 'subset' внутри input_dir).

## Выходные файлы

- `data/yoloe_results/pseudo_coco.json`: Псевдо-аннотации в COCO формате.
- `data/yoloe_results/annotated_images/`: Изображения с визуализацией предсказаний (bbox и маски).

## Отладка

Интерактивно:
```
docker run -it --gpus all -v ./data:/data -v ./data_preparation/yoloe:/app tbank-yoloe-ultralytics /bin/bash
```
Затем внутри контейнера:
```
python tbank_yoloe_bulk_inference.py
```