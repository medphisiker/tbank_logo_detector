# Настройка Label Studio в Docker для разметки логотипов T-Bank

## Предварительные требования
- Установленный Docker Desktop на Windows 10 (скачайте с https://www.docker.com/products/docker-desktop/, если не установлен).
- PowerShell (доступен по умолчанию в Windows 10).
- Папка проекта: `d:/tinkoff/tbank_logo_detector`.
- Существующая папка: `tbank_official_logos/` с PNG-файлами (logo0.png - logo8.png).
- Для `data_sirius/`: Создайте папку вручную, если нужно (например, `mkdir data_sirius`), и поместите туда файлы для разметки.

Label Studio будет запущен в Docker-контейнере с монтированием локальных папок для импорта задач. Порт: 8080. Логин/пароль: admin/admin.

## Шаг 1: Запуск Label Studio в Docker (PowerShell)
Откройте PowerShell в директории проекта (`d:/tinkoff/tbank_logo_detector`) и выполните команды:

1. Убедитесь, что Docker запущен: `docker --version` (должен показать версию).

2. Запустите контейнер Label Studio с монтированием volumes и переменной окружения для локального хранения:
   ```
   docker run -it -p 8080:8080 -v "${PWD}/data:/label-studio/data/local" -v label_studio_data:/label-studio/data --env LOCAL_FILES_SERVING_ENABLED=true heartexlabs/label-studio:1.20.0
   ```
   - `-p 8080:8080`: Прокидывает порт для доступа к UI.
   - `-v "${PWD}/data:/label-studio/data/local"`: Монтирует папку `data` в контейнер как `/label-studio/data/local` (системные файлы LS остаются в контейнере).
     - Картинки из датасета: `/label-studio/data/local/data_sirius/images/`
     - Референсные логотипы: `/label-studio/data/local/tbank_official_logos/images/`
   - `-v label_studio_data:/label-studio/data`: Создаёт volume для хранения проектов LS (данные сохраняются между запусками).
   - `--env LOCAL_FILES_SERVING_ENABLED=true`: Включает возможность подключения локальных файлов (решает ошибку "Serving local files can be dangerous").
   - Если нужно остановить: `Ctrl+C`, контейнер остановится. Для фонового режима: добавьте `-d` (detached).

   Альтернатива: Создайте PowerShell-скрипт `start_ls_docker.ps1` для автоматизации (см. ниже).

3. После запуска откройте браузер: http://localhost:8080.

   **Первый вход (регистрация):**
   - При первом запуске LS покажет форму создания аккаунта.
   - Свой Email: например, admin@example.com.
   - Пароль: your_password.
   - Подтвердите пароль и создайте аккаунт.
   - После регистрации войдите с этими данными.

   **Последующие входы:** Используйте созданные email/пароль (admin@example.com / your_password).

## Шаг 2: Создание проекта для Object Detection
1. В UI Label Studio нажмите **Create Project**.
2. Название: "TBank Logo Detection" (или любое).
3. Тип: **Object Detection with Bounding Boxes**.
4. Конфигурация (config.xml): Используйте предустановленную для bbox или кастомизируйте:
   ```
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="yellow_shield_black_T" background="yellow"/>
       <Label value="white_shield_black_T" background="white"/>
       <Label value="purple_shield_black_T" background="purple"/>
     </RectangleLabels>
   </View>
   ```
   - Классы: yellow_shield_black_T (для жёлтых логотипов: logo0-3,6,8), white_shield_black_T (белые: logo1,4,5), purple_shield_black_T (фиолетовый: logo7).
   - Сохраните проект.

## Шаг 3: Импорт задач (файлов для разметки)

### Вариант 1: Быстрый импорт для небольших датасетов (до 1000 файлов)
1. В проекте: **Import** > **Local Files**.
2. Выберите папку: `/label-studio/data/local/tbank_official_logos/images` (для официальных логотипов) или `/label-studio/data/local/data_sirius/images` (для других данных).
3. Загрузите PNG-файлы (LS автоматически создаст задачи из изображений).
4. Для tbank_official_logos: Импортируйте logo0.png - logo8.png (9 файлов).
5. Подтвердите импорт. Задачи появятся в **Tasks**.

### Вариант 2: Рекомендуемый способ для больших датасетов (рекомендуется для 30 000+ файлов)
Используйте **Source Cloud Storage** в виде Local files для эффективной работы с большими объемами данных:

1. Перейдите в **Settings** проекта.
2. Выберите вкладку **Cloud Storage**.
3. Нажмите **Add Source Storage**.
4. Выберите тип **Local files**.
5. Укажите параметры:
   - **Storage title**: "Local Images Storage"
   - **Absolute local path**: `/label-studio/data/local/tbank_official_logos/images` или `/label-studio/data/local/data_sirius/images`
   - **Import method**: Выберите **Files** (это активирует опцию "Treat every bucket object as a source file")
6. Нажмите **Check Connection** для проверки.
7. Нажмите **Add Storage** для сохранения.
8. После создания хранилища нажмите **Sync Storage** - это импортирует все файлы из указанной папки в проект.

**Преимущества Source Cloud Storage:**
- Работает значительно быстрее для больших датасетов (30 000+ файлов)
- Позволяет динамически добавлять новые файлы в папку и синхронизировать их кнопкой **Sync Storage**
- Не требует перезапуска проекта при добавлении новых файлов
- Поддерживает автоматическое обнаружение новых файлов
- Эффективнее использует ресурсы по сравнению с UI upload

**Важно для больших объемов данных:**
- Label Studio не предназначен для хостинга больших объемов данных через UI upload
- Для датасетов более 1000 файлов всегда используйте Source Cloud Storage
- Рекомендуется использовать cloud storage (S3, GCS, Azure) для production проектов

**Примечание:** Source Cloud Storage с типом Local files - это оптимальный способ работы с большими датасетами в Label Studio. Он обеспечивает быструю синхронизацию, автоматическое обнаружение новых файлов и эффективное использование ресурсов.

## Шаг 4: Разметка (Annotation)
1. Перейдите в **Annotate**.
2. Для каждого изображения (logo0-8):
   - Выделите bbox вокруг логотипа.
   - Присвойте класс: 
     - yellow_shield_black_T для logo0,1,2,3,6,8 (жёлтые/оранжевые щиты с чёрной T).
     - white_shield_black_T для logo1,4,5 (белые щиты с чёрной T; уточните по визуалу).
     - purple_shield_black_T для logo7 (фиолетовый).
   - Сохраните аннотацию (Ctrl+Enter).
3. Разметка manual: Выполните для всех 9 изображений. Группируйте по классам для эффективности.

## Шаг 5: Экспорт данных
1. В проекте: **Export** > **COCO**.
2. Скачайте файл: `tbank_logo_detection_coco.zip` (или аналогичный).
3. Разархивируйте архив и переименуйте json файл в `refs_ls_coco.json`.
4. Сохраните в корень проекта или в `data/tbank_official_logos/annotations`.

## Шаг 6: Импорт существующих COCO аннотаций (Опционально)
Если у вас уже есть размеченные данные в COCO формате, вы можете импортировать их в Label Studio:

### Вариант 1: Использование Label Studio Converter
```bash
# Установите Label Studio Converter
uv sync
# или если нужно добавить пакет:
# uv add label-studio-converter

# Конвертируйте COCO в Label Studio JSON формат
label-studio-converter import coco \
  -i data/tbank_official_logos/refs_ls_coco.json \
  -o data/tbank_official_logos/label_studio_annotations.json
```

### Вариант 2: Использование готового скрипта
```bash
# Запустите готовый скрипт конвертации
python convert_scrypt.py
```

**Скрипт автоматически:**
- Проверит наличие входного COCO файла
- Конвертирует его в Label Studio JSON формат
- Покажет инструкции по импорту в Label Studio
- Обработает возможные ошибки

### Вариант 3: Импорт через UI Label Studio
1. В проекте Label Studio нажмите **Import**
2. Выберите **Upload Files**
3. Загрузите конвертированный JSON файл `label_studio_annotations.json`
4. Аннотации будут импортированы вместе с изображениями

**Важно:** Убедитесь, что пути к изображениям в COCO файле соответствуют структуре монтирования Docker (`/label-studio/data/local/...`).

## Шаг 7: Исправление путей в COCO файле (Опционально)
Если пути к изображениям в экспортированном COCO файле указывают на Docker-контейнер Label Studio, используйте скрипт `fix_coco_paths.py` для преобразования путей в относительные:

```bash
# Исправление путей в COCO файле
uv run python fix_coco_paths.py data/tbank_official_logos/refs_ls_coco.json --images-dir images

# Скрипт создаст файл refs_ls_coco_fixed.json с исправленными путями
```

**Что делает скрипт:**
- Преобразует пути типа `../../label-studio/data/local/tbank_official_logos/logo0.png` в `images/logo0.png`
- Сохраняет исправленный файл как `refs_ls_coco_fixed.json` в той же директории
- Сохраняет все аннотации и метаданные без изменений

## Автоматизация запуска (Опционально: PowerShell-скрипт)
Создайте файл `start_ls_docker.ps1` в корне:
```
# start_ls_docker.ps1
$projectDir = Get-Location
docker run -it -p 8080:8080 `
  -v "${projectDir}/data:/label-studio/data/local" `
  -v label_studio_data:/label-studio/data `
  --env LOCAL_FILES_SERVING_ENABLED=true `
  heartexlabs/label-studio:1.20.0
```
Запуск: `.\start_ls_docker.ps1` в PowerShell (разрешите выполнение: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, если нужно).

## Устранение неисправностей
- Docker не запускается: Проверьте, что Docker Desktop работает (иконка в трее).
- Папки не видны: Убедитесь в правильных путях volumes (используйте абсолютные, если relative не работает: `-v "d:/tinkoff/tbank_logo_detector/data:/label-studio/data/local"`).
- Порт занят: Измените на `-p 8081:8080` и откройте localhost:8081.
- data_sirius пустая: Создайте папку и добавьте изображения.
- Ошибка "Serving local files can be dangerous": Убедитесь, что установлена переменная окружения `LOCAL_FILES_SERVING_ENABLED=true` при запуске Docker контейнера.
- Проблемы с импортом больших объемов данных: Для датасетов более 1000 файлов используйте Source Cloud Storage вместо UI upload.
- Source Cloud Storage не синхронизируется: Проверьте, что выбрана опция "Import method: Files" (активирует "Treat every bucket object as a source file") при настройке хранилища.
- Новые файлы не появляются: После добавления файлов в папку нажмите "Sync Storage" для их обнаружения и импорта.

После разметки, экспорта и исправления путей файл `refs_ls_coco_fixed.json` готов для импорта в Colab/YOLO.

**Примечание:** Если у вас уже есть готовые COCO аннотации, используйте **Шаг 6** для их импорта в Label Studio перед началом новой разметки.

## Шаг 8: Подготовка данных для Google Colab (Опционально)
Для удобной загрузки данных на Google Drive и работы в Google Colab используйте скрипт `prepare_data_for_colab.py`:

```bash
# Подготовка архивов для Google Drive
uv run python prepare_data_for_colab.py

# Скрипт создаст:
# - Папку tbank_logo_detector_data/
# - Архив tbank_official_logos_[timestamp].zip
# - Архив data_sirius_[timestamp].zip (если папка существует)
```

**Что делает скрипт:**
- Создает папку `tbank_logo_detector_data` в корне проекта
- Создает ZIP архив для всей папки `data` (включая `tbank_official_logos` и `data_sirius` с подпапками `images`)
- Перемещает архив в папку `tbank_logo_detector_data`
- Показывает размеры созданных архивов

**Результат:**
```
==================================================
ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА
==================================================

Созданные архивы в папке tbank_logo_detector_data:
  - data_[timestamp].zip: ~1.6 ГБ

📁 Загрузите папку tbank_logo_detector_data на Google Drive
📓 Затем подключите ее к Google Colab для работы с YOLOE и GROUNDING DINO
```

После загрузки на Google Drive папку `tbank_logo_detector_data` можно подключить к Google Colab для дальнейшей обработки с помощью YOLOE и GROUNDING DINO.

Теперь у вас есть полный пайплайн от разметки в Label Studio до подготовки данных для машинного обучения в Google Colab!