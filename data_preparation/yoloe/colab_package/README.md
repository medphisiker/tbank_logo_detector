# YOLOE Colab Package

Этот пакет предназначен для автоматической разметки данных логотипов Т-Банка с использованием модели YOLOE в Google Colab.

## Структура пакета

- `paths.py`: Константы и пути к файлам (GDRIVE_BASE, ZIP_FILES и т.д.).
- `prepare_data.py`: Разархивирование архивов из Google Drive и создание subset изображений (рекурсивно).
- `yolo_predict.py`: Загрузка модели YOLOE, подготовка промптов и выполнение предикта (с recursive=True для поддиректорий).
- `export_coco.py`: Экспорт результатов предикта в COCO формат (pseudo_coco.json) с os.walk для рекурсивной структуры.
- `save_results.py`: Сохранение результатов в Google Drive (копирование JSON и ZIP архива runs).
- `main.py`: Основной скрипт для оркестрации всего процесса, с функцией main().
- `__init__.py`: Пустой файл для Python пакета.

## Установка и использование в Google Colab

1. **Подключите Google Drive вручную**:
   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Скопируйте пакет в Colab** (через файлы или git):
   - Загрузите папку `colab_package` в /content/colab_package.

3. **Добавьте путь к пакету**:
   ```
   import sys
   sys.path.append('/content/colab_package')
   ```

4. **Установите зависимости**:
   ```
   !pip install -U ultralytics pycocotools opencv-python pillow numpy google-colab --quiet
   ```

5. **Запустите пакет**:
   ```
   from colab_package.main import main
   main()
   ```
   Или запустите main.py напрямую:
   ```
   %run colab_package/main.py
   ```

## Предварительные требования

- В Google Drive по пути `/content/drive/MyDrive/tbank_logo_detector_data/` разместите архивы:
  - `data_sirius.zip`
  - `data_synt.zip`
  - `tbank_official_logos.zip` (с refs_ls_coco.json внутри)

- Опционально: `small_gt_coco.json` для оценки mAP (раскомментируйте в save_results.py или добавьте отдельный скрипт).

## Настройка

- В `paths.py` измените `SUBSET = None` на число для подмножества изображений (e.g. 100).
- `GDRIVE_BASE` - путь к данным в Drive.

## Выходные файлы

- `pseudo_coco.json`: Псевдо-аннотации в COCO формате.
- `runs_colab.zip`: Архив с TXT labels и изображениями с bbox (в OUTPUT_DIR).

## Оценка mAP (опционально)

В save_results.py раскомментируйте секцию для COCOeval, если есть GT COCO.

Пакет готов к использованию без установки - просто импортируйте и запустите.