# YOLOE Docker Setup

## Обзор

Этот Dockerfile позволяет запустить пакет YOLOE для автоматической разметки логотипов Т-Банка в изолированной среде. Использует uv для управления зависимостями, основан на python:3.10-slim.

## Требования

- Docker установлен.
- Данные в `data/data_sirius/images/` (изображения для разметки) и `data/tbank_official_logos/` (рефы).

## Сборка образа

Из корня проекта (d:/tinkoff/tbank_logo_detector):

```
docker build -f data_preparation/yoloe/Dockerfile -t tbank-yoloe data_preparation/yoloe
```

## Запуск контейнера

Из корня проекта, примонтируйте директорию `data` в /app/data внутри контейнера:

```
docker run -v ./data:/app/data tbank-yoloe
```

- Volume mount: `./data:/app/data`.
- Результаты сохраняются в `data/yoloe_results/` (pseudo_coco.json, runs_yoloe.zip).

## Настройка

- В `yoloe_package/paths.py`: SUBSET для теста (e.g. 10 изображений).
- GPU: Добавьте `--gpus all` в docker run, если нужно (требует NVIDIA Docker).

## Выходные файлы

- `data/yoloe_results/pseudo_coco.json`: COCO с псевдо-аннотациями.
- `data/yoloe_results/runs_yoloe.zip`: Labels и визуализации.

## Отладка

Для интерактивного режима из корня:

```
docker run -it -v ./data:/app/data tbank-yoloe /bin/bash
cd /app/yoloe && python yoloe_package/main.py
```

Пакет yoloe_package включен в образ, зависимости установлены через uv.