# Docker для Synthesis Generator

## Обзор

В директории `docker/` содержится все необходимое для запуска пайплайна генерации синтетических данных в Docker-контейнере. Рекомендуется использовать готовый образ с Docker Hub для простоты.

## Docker Hub образ

🚀 **Рекомендуемый способ использования** — готовый образ на Docker Hub:

```bash
docker pull medphisiker/tbank-synth:latest
```

**Преимущества использования готового образа:**
- ✅ Готов к использованию без сборки
- ✅ Все зависимости предустановлены
- ✅ Оптимизирован для синтеза данных Т-Банка
- ✅ Регулярные обновления и исправления
- ✅ Консистентная среда выполнения

[Ссылка на Docker Hub](https://hub.docker.com/r/medphisiker/tbank-synth)

**Описание образа:**
Docker-образ для генерации синтетических данных логотипов Т-Банка с расширенными возможностями. Включает пакет `synthesis_generator` с модульной архитектурой, `gen_synth.py` с multi-logo размещением, hard-negatives, расширенными albumentations augmentations, uv для deps (albumentations, pillow, numpy, tqdm, requests). Поддерживает параметры: `--N`, `--min_scale_down`, `--iou_threshold`, `--max_neg`. Используйте `docker run` с mounts для `/app/synthesis` и `/app/data`.

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


## Файлы в директории docker/

- `Dockerfile`: инструкции для сборки образа
- `pyproject.toml`: зависимости Python-пакета
- `README.md`: эта документация