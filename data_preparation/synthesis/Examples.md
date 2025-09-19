# Примеры использования генератора синтетических данных

## Генерация с hard-negatives и multi-logo

### Linux/Mac - полный пайплайн с новыми возможностями
```bash
cd data_preparation/synthesis
python crop_logos.py
python download_backgrounds.py --num 1000 --size 1920
python prepare_background_objects.py --num 200
python gen_synth.py --N 5000 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15
```

### Windows PowerShell - полный пайплайн с новыми возможностями
```powershell
cd data_preparation/synthesis
python crop_logos.py
python download_backgrounds.py --num 1000 --size 1920
python prepare_background_objects.py --num 200
python gen_synth.py --N 5000 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15
```

### Через Docker - только генерация с кастомными параметрами (Linux/Mac)
```bash
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 2000 --min_scale_down 0.7 --iou_threshold 0.3 --max_neg 20
```

### Через Docker - только генерация с кастомными параметрами (Windows PowerShell)
```powershell
docker run -v "${PWD}/data_preparation/synthesis:/app/synthesis" -v "${PWD}/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 2000 --min_scale_down 0.7 --iou_threshold 0.3 --max_neg 20
```

## Кастомная настройка для разных сценариев

### Для dense сцен (много объектов)
```bash
# Linux/Mac
python gen_synth.py --N 1000 --iou_threshold 0.2 --max_neg 25

# Windows PowerShell
python gen_synth.py --N 1000 --iou_threshold 0.2 --max_neg 25
```

### Для clean сцен (минимальные distractors)
```bash
# Linux/Mac
python gen_synth.py --N 1000 --max_neg 5

# Windows PowerShell
python gen_synth.py --N 1000 --max_neg 5
```

### Для high-res backgrounds
```bash
# Linux/Mac
python download_backgrounds.py --num 2000 --size 2560

# Windows PowerShell
python download_backgrounds.py --num 2000 --size 2560
```

## Программное использование

### Базовый пример генерации
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

### Продвинутый пример с distractors
```python
from pathlib import Path
from synthesis_generator.generator import generate_synthetic_dataset

# Пути
crops_dir = Path("data_preparation/synthesis/crops")
bg_dir = Path("data_preparation/synthesis/backgrounds")
bg_objects_dir = Path("data_preparation/synthesis/background_objects")
out_base = Path("data/data_synt")

# Генерация с расширенными параметрами
generate_synthetic_dataset(
    crops_dir=crops_dir,
    bg_dir=bg_dir,
    out_base=out_base,
    N=2000,
    bg_objects_dir=bg_objects_dir,  # Distractors
    min_scale_down=0.5,             # Random background scaling
    iou_threshold=0.4,              # IoU control for multi-logo
    max_neg=15                     # Max distractors per image
)
```

## Сценарии использования

### Сценарий 1: Быстрый старт (минимальная настройка)
```bash
# Скачать готовый образ
docker pull medphisiker/tbank-synth:latest

# Запустить полный пайплайн
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 1000
```

### Сценарий 2: Кастомизация для специфических требований
```bash
# Подготовка данных
python crop_logos.py
python download_backgrounds.py --num 2000 --size 1920
python prepare_background_objects.py --num 300

# Генерация с кастомными параметрами
python gen_synth.py --N 5000 --min_scale_down 0.3 --iou_threshold 0.5 --max_neg 20
```

### Сценарий 3: Интеграция в ML пайплайн
```python
# В вашем скрипте обучения
from synthesis_generator.generator import generate_synthetic_dataset

def prepare_training_data():
    # Генерация синтетических данных
    generate_synthetic_dataset(
        crops_dir="crops/",
        bg_dir="backgrounds/",
        out_base="data/train_synth",
        N=10000,
        bg_objects_dir="background_objects/",
        min_scale_down=0.5,
        iou_threshold=0.4,
        max_neg=15
    )

    # Загрузка и merge с real данными
    # ... ваш код обучения
```

## Troubleshooting

### Проблема: Недостаточно разнообразия в фоне
**Решение**: Увеличьте количество фонов и размер изображений
```bash
python download_backgrounds.py --num 2000 --size 2560
```

### Проблема: Слишком много false positives
**Решение**: Добавьте больше distractors и уменьшите IoU threshold
```bash
python prepare_background_objects.py --num 500
python gen_synth.py --iou_threshold 0.3 --max_neg 25
```

### Проблема: Низкая точность на small objects
**Решение**: Уменьшите min_scale_down для большего разнообразия размеров
```bash
python gen_synth.py --min_scale_down 0.2