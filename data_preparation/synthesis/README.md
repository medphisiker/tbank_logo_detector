# Data Preparation — Synthesis Scripts

## Обзор

В директории `data_preparation/synthesis/` реализован пайплайн генерации синтетических данных для детекции логотипов Т-Банка. Основан на разработанном нами Python-пакете `synthesis_generator`, который предоставляет модульную архитектуру для создания разнообразных синтетических изображений.

За подробностями о пакете `synthesis_generator` обращайтесь к [`data_preparation/synthesis/synthesis_generator/README.md`](synthesis_generator/README.md).

## Быстрый старт

Для запуска пайплайна синтеза данных используйте отдельные скрипты для каждого этапа. Это позволяет анализировать результаты каждого шага и при необходимости корректировать параметры:

### Полный пайплайн (последовательное выполнение):

```cmd
# Windows CMD
cd data_preparation/synthesis
run_docker_crop_logos.bat
run_docker_backgrounds_download.bat 1000 1920
run_docker_prepare_background_objects.bat 200
run_docker_synth_gen.bat 5000 0.5 0.4 15

# Windows PowerShell
cd data_preparation/synthesis
.\run_docker_crop_logos.bat
.\run_docker_backgrounds_download.bat 1000 1920
.\run_docker_prepare_background_objects.bat 200
.\run_docker_synth_gen.bat 5000 0.5 0.4 15
```

Эта последовательность выполнит:
1. **Обрезку логотипов** из официальных изображений
2. **Скачивание фонов** (1920x1920, 1000 изображений)
3. **Подготовку distractors** (200 объектов)
4. **Генерацию синтетики** с расширенными возможностями

Результат сохраняется в `data/data_synt/`.

**Новые возможности:**
- Multi-logo размещение (1-10 логотипов на изображение)
- Hard-negatives (distractors) для улучшения обучения
- Случайный ресайз фонов (0.5-1.0 от оригинала)
- Расширенные аугментации (ElasticTransform, JPEG compression, shadows)
- Балансировка по классам (purple/white/yellow)
- Контроль IoU-перекрытий

## Использование готового Docker-образа

Рекомендуется использовать опубликованный Docker-образ без локальной сборки:

```bash
docker pull medphisiker/tbank-synth:latest
```

**Запуск с готовым образом**:
```bash
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 2000
```

**Через отдельные скрипты**:
```cmd
# Windows CMD
cd data_preparation/synthesis
run_docker_crop_logos.bat
run_docker_backgrounds_download.bat 1000 1920
run_docker_prepare_background_objects.bat 200
run_docker_synth_gen.bat 2000 0.5 0.4 15

# Windows PowerShell
cd data_preparation/synthesis
.\run_docker_crop_logos.bat
.\run_docker_backgrounds_download.bat 1000 1920
.\run_docker_prepare_background_objects.bat 200
.\run_docker_synth_gen.bat 2000 0.5 0.4 15
```

### Docker Hub

[Ссылка на Docker Hub](https://hub.docker.com/r/medphisiker/tbank-synth)

**Описание**: Docker-образ для генерации синтетических данных логотипов Т-Банка с расширенными возможностями. Включает пакет `synthesis_generator` с модульной архитектурой, `gen_synth.py` с multi-logo размещением, hard-negatives, расширенными albumentations augmentations, uv для deps (albumentations, pillow, numpy, tqdm, requests). Поддерживает параметры: `--N`, `--min_scale_down`, `--iou_threshold`, `--max_neg`. Используйте `docker run` с mounts для `/app/synthesis` и `/app/data`.

## Отдельные команды Docker-контейнера

Для запуска отдельных этапов пайплайна используйте прямые команды Docker:

### Windows CMD:
```cmd
# Обрезка логотипов
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm medphisiker/tbank-synth:latest python crop_logos.py

# Скачивание высококачественных фонов (1920x1920)
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm medphisiker/tbank-synth:latest python download_backgrounds.py --num 1000 --size 1920

# Подготовка distractors
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm medphisiker/tbank-synth:latest python prepare_background_objects.py --num 200

# Генерация синтетики с расширенными параметрами
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 2000 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15
```

### Windows PowerShell:
```powershell
# Обрезка логотипов
docker run -v "${PWD}/data_preparation/synthesis:/app/synthesis" -v "${PWD}/data:/app/data" --rm medphisiker/tbank-synth:latest python crop_logos.py

# Скачивание высококачественных фонов (1920x1920)
docker run -v "${PWD}/data_preparation/synthesis:/app/synthesis" -v "${PWD}/data:/app/data" --rm medphisiker/tbank-synth:latest python download_backgrounds.py --num 1000 --size 1920

# Подготовка distractors
docker run -v "${PWD}/data_preparation/synthesis:/app/synthesis" -v "${PWD}/data:/app/data" --rm medphisiker/tbank-synth:latest python prepare_background_objects.py --num 200

# Генерация синтетики с расширенными параметрами
docker run -v "${PWD}/data_preparation/synthesis:/app/synthesis" -v "${PWD}/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 2000 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15
```

**С переменными окружения** (работает в обоих CMD и PowerShell):
```cmd
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm medphisiker/tbank-synth:latest -e N=2000 -e MIN_SCALE_DOWN=0.5 -e IOU_THRESHOLD=0.4 -e MAX_NEG=15 python gen_synth.py
```

## Сборка собственного Docker-образа

**Примечание**: Все BAT и SH скрипты в проекте используют готовый образ с DockerHub (`medphisiker/tbank-synth:latest`). Сборка локального образа требуется только для разработки или кастомизации.

Если необходимо собрать образ локально:

### Ручная сборка
```bash
cd data_preparation/synthesis/docker
docker build -t tbank-synth -f Dockerfile .
```

### Через скрипты
```cmd
# Windows CMD
cd data_preparation/synthesis
run_synth.bat 2000

# Windows PowerShell
cd data_preparation/synthesis
.\run_synth.bat 2000

# Linux/Mac
cd data_preparation/synthesis
./docker-build-run.sh
```

### Структура образа

- **Базовый образ**: `python:3.10-slim`
- **Рабочая директория**: `/app`
- **Mount точки**:
  - `/app/synthesis`: директория с кодом и промежуточными данными
  - `/app/data`: директория с входными и выходными данными
- **CMD**: `python gen_synth.py`
- **Поддержка**: аргументы `--N`, `--min_scale_down`, `--iou_threshold`, `--max_neg` и соответствующие переменные окружения
- **Новые возможности**: multi-logo, distractors, advanced augmentations, random background scaling

### Публикация образа
```bash
docker login
docker tag tbank-synth medphisiker/tbank-synth:latest
docker push medphisiker/tbank-synth:latest
```

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


## Примеры использования новых возможностей

### Генерация с hard-negatives и multi-logo
```bash
# Linux/Mac - полный пайплайн с новыми возможностями
cd data_preparation/synthesis
python crop_logos.py
python download_backgrounds.py --num 1000 --size 1920
python prepare_background_objects.py --num 200
python gen_synth.py --N 5000 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15

# Windows PowerShell - полный пайплайн с новыми возможностями
cd data_preparation/synthesis
python crop_logos.py
python download_backgrounds.py --num 1000 --size 1920
python prepare_background_objects.py --num 200
python gen_synth.py --N 5000 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15

# Через Docker - только генерация с кастомными параметрами (Linux/Mac)
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 2000 --min_scale_down 0.7 --iou_threshold 0.3 --max_neg 20

# Через Docker - только генерация с кастомными параметрами (Windows PowerShell)
docker run -v "${PWD}/data_preparation/synthesis:/app/synthesis" -v "${PWD}/data:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N 2000 --min_scale_down 0.7 --iou_threshold 0.3 --max_neg 20
```

### Кастомная настройка для разных сценариев
```bash
# Linux/Mac - для dense сцен (много объектов)
python gen_synth.py --N 1000 --iou_threshold 0.2 --max_neg 25

# Linux/Mac - для clean сцен (минимальные distractors)
python gen_synth.py --N 1000 --max_neg 5

# Linux/Mac - для high-res backgrounds
python download_backgrounds.py --num 2000 --size 2560

# Windows PowerShell - для dense сцен (много объектов)
python gen_synth.py --N 1000 --iou_threshold 0.2 --max_neg 25

# Windows PowerShell - для clean сцен (минимальные distractors)
python gen_synth.py --N 1000 --max_neg 5

# Windows PowerShell - для high-res backgrounds
python download_backgrounds.py --num 2000 --size 2560
```

## Closing notes

* ✅ **Реализованы все ключевые улучшения** из плана: high-res фоны с random scaling, hard-negatives, multi-logo с IoU-контролем, расширенные аугментации, балансировка классов, позиционирование с контролем видимости
* Цель — максимально автоматизировать и документировать происхождение аннотаций (provenance), чтобы при обнаружении проблем можно было быстро проследить, откуда пришло неверное правило и исправить pipeline.
* На верхнем уровне мы используем современную связку VLM + SAM + verification (VLM/ML classifier/OCR) + эмбеддинг-кластеризацию для масштабируемой и дешёвой по времени разметки.
* Следующий шаг: по этому документу сгенерировать runnable notebook / script в `annotation/` с примером запуска Grounding DINO → SAM → ensemble и с шаблоном промптов. Готов приступить и сгенерировать `annotation`-notebook прямо сейчас, если нужно.

Для планов и идей см. [3.data_preparation_plan.md](3.data_preparation_plan.md).