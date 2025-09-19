# Synthesis Generator

Модуль для генерации синтетических данных логотипов Т-Банка для задач детекции объектов. Создает разнообразные изображения с наложением логотипов на случайные фоны с применением аугментаций.

## Описание

Пакет `synthesis_generator` предоставляет инструменты для создания синтетического датасета изображений с логотипами Т-Банка. Поддерживает 3 класса логотипов (purple, white, yellow) и генерирует данные в формате YOLO для обучения моделей детекции.

Основные возможности:
- Обрезка логотипов из COCO-аннотаций
- Скачивание фоновых изображений
- Генерация синтетических изображений с аугментациями
- Автоматическое определение путей для Docker и локального запуска
- Поддержка балансировки по классам

## Структура модуля

```
synthesis_generator/
├── __init__.py           # Инициализация пакета
├── crop_utils.py         # Функции для обрезки логотипов из COCO
├── background_utils.py   # Функции для скачивания фоновых изображений
├── augmentations.py      # Конфигурация аугментаций Albumentations
├── generator.py          # Основные функции генерации синтетических данных
└── README.md             # Эта документация
```

## Установка

### Зависимости

- Python >= 3.10
- albumentations
- pillow
- numpy
- tqdm
- requests

### Установка пакета

```bash
cd data_preparation/synthesis
pip install -e .
```

Или через uv:
```bash
cd data_preparation/synthesis
uv sync
```

## Использование

### Программное использование

```python
from synthesis_generator.crop_utils import crop_logos
from synthesis_generator.background_utils import download_backgrounds
from synthesis_generator.generator import generate_synthetic_dataset

# 1. Обрезка логотипов из официальных изображений
crop_logos(
    coco_path="data/tbank_official_logos/refs_ls_coco.json",
    images_dir="data/tbank_official_logos/images",
    crops_dir="data_preparation/synthesis/crops"
)

# 2. Скачивание фоновых изображений
download_backgrounds(
    backgrounds_dir="data_preparation/synthesis/backgrounds",
    num_backgrounds=500,
    img_size=640
)

# 3. Генерация синтетического датасета
generate_synthetic_dataset(
    crops_dir="data_preparation/synthesis/crops",
    bg_dir="data_preparation/synthesis/backgrounds",
    out_base="data/data_synt",
    N=1000
)
```

### Через скрипты

```bash
# Из директории data_preparation/synthesis
python crop_logos.py
python download_backgrounds.py
python gen_synth.py --N 1000
```

### Через Docker

```bash
# Сборка образа
docker build -t tbank-synth -f Dockerfile .

# Запуск пайплайна
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" \
           -v "$(pwd)/data:/app/data" \
           --rm tbank-synth python gen_synth.py --N 1000
```

## API Reference

### crop_utils.py

#### `load_coco_annotations(coco_path: str) -> dict`
Загружает COCO-аннотации из JSON-файла.

#### `crop_logos_from_annotations(coco: dict, images_dir: str, crops_dir: str) -> None`
Обрезает логотипы из изображений на основе COCO-аннотаций.

#### `crop_logos(coco_path: str, images_dir: str, crops_dir: str) -> None`
Основная функция для обрезки логотипов.

### background_utils.py

#### `download_backgrounds(backgrounds_dir: str, num_backgrounds: int = 500, img_size: int = 640) -> None`
Скачивает случайные фоновые изображения с Picsum.

### augmentations.py

#### `get_augmentation_pipeline() -> A.Compose`
Возвращает пайплайн аугментаций на основе Albumentations.

### generator.py

#### `setup_output_dirs(out_base: Path, splits: list = ['train', 'val', 'test']) -> None`
Создает директории для изображений и лейблов.

#### `load_crops_by_class(crops_dir: Path) -> dict`
Загружает обрезанные логотипы, организованные по классам.

#### `load_backgrounds(bg_dir: Path) -> list`
Загружает список путей к фоновым изображениям.

#### `generate_synthetic_image(bg_path: str, crop_path: str, aug_pipeline) -> tuple`
Генерирует одно синтетическое изображение с наложением логотипа.

#### `save_yolo_label(lbl_path: Path, cls: int, bbox: tuple) -> None`
Сохраняет лейбл в формате YOLO.

#### `generate_synthetic_dataset(crops_dir: Path, bg_dir: Path, out_base: Path, N: int, aug_pipeline=None) -> None`
Генерирует полный синтетический датасет.

## Формат данных

### Классы логотипов
- 0: purple (фиолетовый)
- 1: white (белый)
- 2: yellow (желтый)

### Структура выходных данных
```
data/data_synt/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Формат лейблов YOLO
```
<class_id> <x_center> <y_center> <width> <height>
```

## Аугментации

Применяются следующие аугментации:
- RandomBrightnessContrast (яркость/контраст)
- GaussNoise (гауссов шум)
- MotionBlur (размытие движения)
- HueSaturationValue (оттенок/насыщенность/яркость)

## Конфигурация

Модуль автоматически определяет окружение:
- Docker: использует пути `/app/synthesis` и `/app/data`
- Локально: использует относительные пути от корня проекта

## Примеры

### Генерация 2000 изображений

```bash
cd data_preparation/synthesis
python gen_synth.py --N 2000
```

### Использование готового Docker-образа

```bash
docker pull medphisiker/tbank-synth:latest
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" \
           -v "$(pwd)/data:/app/data" \
           --rm medphisiker/tbank-synth python gen_synth.py --N 1000
```

## Разработка и расширение

Для расширения функциональности:
1. Добавьте новые аугментации в `augmentations.py`
2. Реализуйте новые функции генерации в `generator.py`
3. Обновите зависимости в `pyproject.toml`