# YOLOE Package

Этот пакет предназначен для автоматической разметки данных логотипов Т-Банка с использованием модели YOLOE в локальной среде (Windows/Linux/Mac).

## Структура пакета

- `paths.py`: Константы и пути к файлам (INPUT_IMAGES_DIR, REFS_LOCAL и т.д.).
- `prepare_data.py`: Подготовка изображений из директории (с опциональным subset).
- `yolo_predict.py`: Загрузка модели YOLOE, подготовка промптов и выполнение предикта (recursive=True для поддиректорий).
- `export_coco.py`: Экспорт результатов предикта в COCO формат (pseudo_coco.json) с os.walk для рекурсивной структуры.
- `save_results.py`: Сохранение результатов (JSON и ZIP архива runs) в OUTPUT_DIR.
- `main.py`: Основной скрипт для оркестрации всего процесса.
- `__init__.py`: Пустой файл для Python пакета.

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

3. **Добавьте путь к пакету** (если запускаете из другого места):
   ```
   import sys
   sys.path.append('data_preparation/yoloe/colab_package')
   ```

4. **Запустите пакет**:
   ```
   from colab_package.main import main
   main()
   ```
   Или напрямую:
   ```
   cd data_preparation/yoloe/colab_package
   python main.py
   ```

## Настройка

- В `paths.py`:
  - `SUBSET = None` для полного датасета или число (e.g. 10) для подмножества.
  - `OUTPUT_DIR = 'yoloe_results/'` - директория для результатов.

- Модель: 'yoloe-l-seg.pt' (скачивается автоматически при первом запуске).

## Выходные файлы

- `yoloe_results/pseudo_coco.json`: Псевдо-аннотации в COCO формате для data_sirius.
- `yoloe_results/runs_yoloe.zip`: Архив с TXT labels и изображениями с bbox.

## Оценка mAP (опционально)

Добавьте GT COCO (e.g. small_gt_coco.json) и используйте pycocotools для оценки в save_results.py.

Пакет готов к использованию без дополнительной установки.