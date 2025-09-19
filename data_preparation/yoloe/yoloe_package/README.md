# YOLOE Package

Этот пакет предназначен для автоматической разметки данных логотипов Т-Банка с использованием модели YOLOE в локальной среде (Windows/Linux/Mac). Интегрирован с tbank_yoloe_bulk_inference.py для конфигурации через config.json.

## Структура пакета

- `prepare_data.py`: Подготовка изображений из директории (с опциональным subset).
- `yolo_predict.py`: Загрузка модели YOLOE, подготовка визуальных промптов из референсных изображений и выполнение предикта.
- `export_coco.py`: Экспорт результатов предикта в COCO формат (pseudo_coco.json).
- `pipeline.py`: Основной пайплайн инференса (подготовка данных, предикт, экспорт, сохранение результатов).
- `config.py`: Загрузка конфигурации из config.json.
- `directories.py`: Создание необходимых директорий.
- `__init__.py`: Экспорт функций пакета.

## Установка и использование

1. **Убедитесь в наличии данных**:
   - `data/data_sirius/images/` - изображения для разметки (поддерживает рекурсивную структуру).
   - `data/tbank_official_logos/refs_ls_coco.json` - COCO разметка референсных логотипов.
   - `data/tbank_official_logos/images/` - референсные изображения (используются косвенно через bbox в JSON).

2. **Установите зависимости** (рекомендуется в виртуальном окружении):
   ```
   pip install ultralytics torch torchvision pillow numpy opencv-python pycocotools
   ```
   - Для GPU: убедитесь, что CUDA установлен и torch поддерживает GPU.

3. **Настройте config.json** (в корне yoloe):
```json
{
  "input_dir": "/data/data_sirius/images",
  "refs_images_json": "/data/tbank_official_logos/refs_ls_coco.json",
  "refs_images_dir": "/data/tbank_official_logos",
  "output_dir": "/data/yoloe_results",
  "subset": null,
  "conf": 0.5,
  "iou": 0.7,
  "runs_dir": "runs/yoloe_predict",
  "device": "0",
  "weights_dir": "./ultralytics_weights",
  "batch_size": 1,
  "imgsz": 480,
  "half": true
}
```

**Примечание по weights_dir**: Из-за бага в ultralytics, веса моделей скачиваются в /app (в Docker-контейнере), независимо от указанного weights_dir. В локальной среде — в рабочую директорию (data_preparation/yoloe). При монтировании -v ./data_preparation/yoloe:/app веса сохраняются локально в data_preparation/yoloe.

**device**: "0" для GPU 0 (рекомендуется в Docker с --gpus all), "cpu" для CPU. "auto" может не работать в некоторых setup — используйте явный параметр.

Для локального запуска используйте относительные пути (e.g. "input_dir": "data/data_sirius/images"), но в Docker — абсолютные как выше.

**subset**: null для полного датасета или число (e.g. 10) для подмножества первых N изображений (копируются в поддиректорию 'subset' внутри input_dir).

**batch_size**: размер батча для обработки изображений (по умолчанию 1, уменьшите при нехватке памяти GPU).

**imgsz**: размер входного изображения для модели (по умолчанию 640, уменьшите для снижения потребления памяти).

**half**: включить FP16 для снижения потребления памяти (по умолчанию false, установите true для GPU с поддержкой FP16).

4. **Запустите пакет**:
   ```
   cd data_preparation/yoloe
   python tbank_yoloe_bulk_inference.py
   ```
   Или из другого места с CONFIG_PATH:
   ```
   python data_preparation/yoloe/tbank_yoloe_bulk_inference.py
   ```

## Настройка

- В `config.json`:
  - `subset`: null для полного датасета или число (e.g. 10) для подмножества.
  - `output_dir`: директория для результатов (e.g. "yoloe_results").
  - `conf`, `iou`: пороги уверенности и IoU.
  - `device`: "auto", "cpu" или "cuda".

- Модель: 'yoloe-11l-seg.pt' (скачивается автоматически при первом запуске).

## Выходные файлы

- `yoloe_results/pseudo_coco.json`: Псевдо-аннотации в COCO формате.
- `yoloe_results/annotated_images/`: Изображения с визуализацией предсказаний (bbox и маски), скопированные из runs/yoloe_predict/predict*/.

Пакет готов к использованию без дополнительной установки.