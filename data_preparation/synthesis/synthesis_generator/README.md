# Synthesis Generator

Модуль для генерации синтетических данных логотипов Т-Банка для задач детекции объектов. Создает разнообразные изображения с наложением логотипов на случайные фоны с применением аугментаций.

## Описание

Пакет `synthesis_generator` предоставляет инструменты для создания синтетического датасета изображений с логотипами Т-Банка. Поддерживает 3 класса логотипов (purple, white, yellow) и генерирует данные в формате YOLO для обучения моделей детекции.

Основные возможности:
- Обрезка логотипов из COCO-аннотаций
- Скачивание высококачественных фоновых изображений (1920x1920) с тематическими опциями
- Генерация синтетических изображений с расширенными аугментациями
- Размещение hard-negatives (distractors) для улучшения обучения
- Multi-logo размещение с контролем IoU-перекрытий
- Балансировка по классам (purple/white/yellow)
- Случайное позиционирование с контролем видимости
- Автоматическое определение путей для Docker и локального запуска

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
cd docker && docker build -t tbank-synth -f Dockerfile .

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

#### `download_backgrounds(backgrounds_dir: str, num_backgrounds: int = 1000, img_size: int = 1920, thematic: bool = False) -> None`
Скачивает фоновые изображения с Picsum или Unsplash API.

**Новые параметры:**
- `img_size`: размер изображений (default: 1920 для high-res)
- `thematic`: использовать тематические фоны (требует API key)

### augmentations.py

#### `get_augmentation_pipeline() -> A.Compose`
Возвращает legacy пайплайн аугментаций.

#### `get_background_aug_pipeline() -> A.Compose`
Возвращает расширенный пайплайн аугментаций для фоновых изображений.

#### `get_neg_aug_pipeline() -> A.Compose`
Возвращает пайплайн аугментаций для distractor объектов.

#### `get_logo_aug_pipeline() -> A.Compose`
Возвращает пайплайн аугментаций для логотипов.

### generator.py

#### `calculate_iou(bbox1: tuple, bbox2: tuple) -> float`
Вычисляет IoU между двумя bounding boxes в YOLO-формате.

#### `load_background_objects(bg_objects_dir: Path) -> list`
Загружает список путей к distractor изображениям.

#### `setup_output_dirs(out_base: Path, splits: list = ['train', 'val', 'test']) -> None`
Создает директории для изображений и лейблов.

#### `load_crops_by_class(crops_dir: Path) -> dict`
Загружает обрезанные логотипы, организованные по классам.

#### `load_backgrounds(bg_dir: Path) -> list`
Загружает список путей к фоновым изображениям.

#### `place_distractors(bg: Image.Image, bg_objects: list, neg_aug_pipeline, max_neg: int = 15) -> Image.Image`
Размещает distractor объекты на фоне без лейблов.

#### `place_multi_logos(bg: Image.Image, crops_by_class: dict, logo_aug_pipeline, iou_threshold: float = 0.4, max_logos: int = 10) -> tuple`
Размещает несколько логотипов с контролем IoU.

#### `generate_synthetic_image(bg_path: str, crop_path: str, aug_pipeline, min_scale_down: float = 0.5) -> tuple`
Генерирует одно синтетическое изображение с наложением логотипа и случайным ресайзом фона.

#### `save_yolo_label(lbl_path: Path, cls: int, bbox: tuple) -> None`
Сохраняет лейбл в формате YOLO.

#### `generate_synthetic_dataset(crops_dir: Path, bg_dir: Path, out_base: Path, N: int, aug_pipeline=None, bg_objects_dir: Path = None, min_scale_down: float = 0.5, iou_threshold: float = 0.4, max_neg: int = 15) -> None`
Генерирует полный синтетический датасет с расширенными возможностями.

**Новые параметры:**
- `bg_objects_dir`: директория с distractor объектами (hard-negatives)
- `min_scale_down`: минимальный коэффициент уменьшения фона (0.5-1.0)
- `iou_threshold`: порог IoU для размещения логотипов (default: 0.4)
- `max_neg`: максимальное количество distractors на изображение (default: 15)


## Разработка и расширение

Для расширения функциональности:
1. Добавьте новые аугментации в `augmentations.py`
2. Реализуйте новые функции генерации в `generator.py`
3. Обновите зависимости в `pyproject.toml`