# Скрипты для генерации синтетических данных

## Обзор

В директории `data_preparation/synthesis/` содержатся скрипты для автоматизации пайплайна генерации синтетических данных. Скрипты разделены на BAT-файлы для Windows, shell-скрипты для Linux/Mac и Python-скрипты для непосредственного выполнения.

## Отдельные скрипты для каждого этапа

### run_docker_crop_logos.bat

**Расположение**: `data_preparation/synthesis/run_docker_crop_logos.bat`

**Назначение**: Обрезка логотипов из официальных изображений.

**Функциональность**:
- Запускает Docker-контейнер для обрезки логотипов
- Использует COCO-аннотации из `data/tbank_official_logos/refs_ls_coco.json`
- Сохраняет обрезанные логотипы в `crops/` по классам

**Запуск**:
```cmd
# Windows CMD
cd data_preparation/synthesis
run_docker_crop_logos.bat

# Windows PowerShell
cd data_preparation/synthesis
.\run_docker_crop_logos.bat
```

### run_docker_backgrounds_download.bat

**Расположение**: `data_preparation/synthesis/run_docker_backgrounds_download.bat`

**Назначение**: Скачивание фоновых изображений.

**Функциональность**:
- Скачивает высококачественные фоновые изображения (1920x1920)
- Поддержка тематических фонов
- Сохраняет в `backgrounds/` директорию

**Запуск**:
```cmd
# Windows CMD
cd data_preparation/synthesis
run_docker_backgrounds_download.bat 1000 1920 false

# Windows PowerShell
cd data_preparation/synthesis
.\run_docker_backgrounds_download.bat 1000 1920 false
```

**Параметры**: `[num] [size] [thematic]`

### run_docker_prepare_background_objects.bat

**Расположение**: `data_preparation/synthesis/run_docker_prepare_background_objects.bat`

**Назначение**: Подготовка объектов-distractors для hard-negatives.

**Функциональность**:
- Скачивает и подготавливает distractor изображения
- Включает Tinkoff-варианты и другие похожие объекты
- Сохраняет в `background_objects/` директорию

**Запуск**:
```cmd
# Windows CMD
cd data_preparation/synthesis
run_docker_prepare_background_objects.bat 200

# Windows PowerShell
cd data_preparation/synthesis
.\run_docker_prepare_background_objects.bat 200
```

**Параметры**: `[num]`

### run_docker_synth_gen.bat

**Расположение**: `data_preparation/synthesis/run_docker_synth_gen.bat`

**Назначение**: Генерация синтетических данных с расширенными возможностями.

**Функциональность**:
- Multi-logo размещение с IoU-контролем
- Добавление distractors
- Расширенные аугментации
- Балансировка по классам
- Сохраняет результат в `data/data_synt/`

**Запуск**:
```cmd
# Windows CMD
cd data_preparation/synthesis
run_docker_synth_gen.bat 5000 0.5 0.4 15

# Windows PowerShell
cd data_preparation/synthesis
.\run_docker_synth_gen.bat 5000 0.5 0.4 15
```

**Параметры**: `[N] [min_scale_down] [iou_threshold] [max_neg]`

## Shell-скрипты (Linux/Mac)

### docker-build-run.sh

**Расположение**: `data_preparation/synthesis/docker-build-run.sh`

**Назначение**: Сборка и запуск Docker-контейнера для Unix-систем.

**Функциональность**:
- Собирает образ `tbank-synth` локально
- Монтирует проект и запускает генерацию

**Запуск**:
```bash
cd data_preparation/synthesis
./docker-build-run.sh
```

**Примечание**: Для поэтапного запуска используйте прямые команды Docker аналогично Windows-версии, заменяя `%CD%` на `$(pwd)` и `docker run` команды.

## Python-скрипты

### crop_logos.py

**Назначение**: Обрезка логотипов из официальных референсных изображений по аннотациям COCO.

**Функциональность**:
- Загружает COCO-аннотации из `data/tbank_official_logos/refs_ls_coco.json`
- Обрабатывает изображения из `data/tbank_official_logos/images/`
- Сохраняет обрезанные логотипы в `data_preparation/synthesis/crops/` с именами по классам (purple_XX.png, white_XX.png, yellow_XX.png)
- Использует функции из пакета `synthesis_generator.crop_utils`

**Запуск**:
```bash
cd data_preparation/synthesis
python crop_logos.py
```

**Особенности**:
- Автоматически определяет пути в зависимости от среды (Docker vs локально)
- В Docker использует `/app/data/`, локально - относительные пути от корня проекта

### download_backgrounds.py

**Назначение**: Скачивание высококачественных фоновых изображений для синтеза.

**Функциональность**:
- Скачивает случайные фотографии с сервиса Picsum (https://picsum.photos)
- Размер изображений: 1920x1920 пикселей (по умолчанию, для high-res)
- Количество: 1000 изображений (по умолчанию)
- Поддержка тематических фонов (требует Unsplash API key)
- Сохраняет в `data_preparation/synthesis/backgrounds/bg_XXXX.jpg`
- Использует функции из пакета `synthesis_generator.background_utils`

**Запуск**:
```bash
cd data_preparation/synthesis
python download_backgrounds.py --num 1000 --size 1920
```

**Аргументы**:
- `--num`: количество изображений (int, default: 1000)
- `--size`: размер изображений (int, default: 1920)
- `--thematic`: использовать тематические фоны (flag)

**Особенности**:
- Не требует VPN или API-ключей для базового режима
- Показывает прогресс загрузки с помощью tqdm
- Обрабатывает ошибки сети и продолжает загрузку
- High-res изображения позволяют случайный downscale для реализма

### prepare_background_objects.py

**Назначение**: Подготовка объектов-distractors (hard-negatives) для синтетической генерации.

**Функциональность**:
- Скачивает изображения похожих объектов (логотипы других банков, generic emblems)
- Создает коллекцию distractors для размещения на фонах без лейблов
- Помогает модели игнорировать похожие, но не целевые объекты
- Сохраняет в `data_preparation/synthesis/background_objects/`

**Запуск**:
```bash
cd data_preparation/synthesis
python prepare_background_objects.py --num 200
```

**Аргументы**:
- `--num`: количество distractor изображений (int, default: 200)
- `--output`: директория для сохранения (str, default: background_objects)

**Особенности**:
- Автоматически скачивает различные типы distractors
- Включает Tinkoff-варианты для hard-negative mining
- Создает разнообразную коллекцию для realistic сцен

### gen_synth.py

**Назначение**: Основной скрипт генерации синтетических данных с расширенными возможностями.

**Функциональность**:
- Принимает аргумент `--N` для количества генерируемых изображений (default: 1000)
- Поддержка переменных окружения для всех параметров
- Multi-logo размещение (1-10 логотипов на изображение)
- Hard-negatives (distractors) для улучшения обучения
- Случайный ресайз фонов (0.5-1.0 от оригинала)
- Контроль IoU-перекрытий между логотипами
- Балансировка по классам (purple/white/yellow)
- Расширенные аугментации (фон + объекты + логотипы)
- Генерирует изображения и лейблы в формате YOLO
- Автоматически распределяет по сплитам train/val/test

**Запуск**:
```bash
cd data_preparation/synthesis
python gen_synth.py --N 5000 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15
```

**Аргументы**:
- `--N`: количество синтетических изображений (int, default: 1000)
- `--min_scale_down`: минимальный коэффициент уменьшения фона (float, default: 0.5)
- `--iou_threshold`: порог IoU для размещения логотипов (float, default: 0.4)
- `--max_neg`: максимальное количество distractors на изображение (int, default: 15)

**Переменные окружения**:
- `N`, `MIN_SCALE_DOWN`, `IOU_THRESHOLD`, `MAX_NEG`

**Особенности**:
- Автоматическое определение путей (Docker: `/app/data/`, локально: `../../data/`)
- Логирование процесса генерации с балансом классов
- Проверка наличия файлов: crops/, backgrounds/, background_objects/
- Поддержка как single-logo, так и multi-logo сцен