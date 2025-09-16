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
   docker run -it -p 8080:8080 -v "${PWD}/tbank_official_logos:/label-studio/data/tbank_official_logos" -v "${PWD}/data_sirius:/label-studio/data/data_sirius" -v label_studio_data:/label-studio/data --env LOCAL_FILES_SERVING_ENABLED=true heartexlabs/label-studio:1.20.0
   ```
   - `-p 8080:8080`: Прокидывает порт для доступа к UI.
   - `-v ...:/label-studio/data/...`: Монтирует локальные папки в контейнер (доступны как `/label-studio/data/tbank_official_logos` и `/label-studio/data/data_sirius` внутри контейнера).
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
1. В проекте: **Import** > **Local Files**.
2. Выберите папку: `/label-studio/data/tbank_official_logos` (для официальных логотипов) или `/label-studio/data/data_sirius` (для других данных).
   - Загрузите PNG-файлы (LS автоматически создаст задачи из изображений).
   - Для tbank_official_logos: Импортируйте logo0.png - logo8.png (9 файлов).
3. Подтвердите импорт. Задачи появятся в **Tasks**.

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
1. В проекте: **Export** > **COCO 1.0**.
2. Скачайте файл: `tbank_logo_detection_coco.json` (или аналогичный).
3. Сохраните в корень проекта или в `dataset/refs/` (создайте папку: `mkdir dataset/refs` в PowerShell).
   - Переименуйте в `refs_ls_coco.json` для совместимости с Colab.

## Автоматизация запуска (Опционально: PowerShell-скрипт)
Создайте файл `start_ls_docker.ps1` в корне:
```
# start_ls_docker.ps1
$projectDir = Get-Location
docker run -it -p 8080:8080 `
  -v "${projectDir}/tbank_official_logos:/label-studio/data/tbank_official_logos" `
  -v "${projectDir}/data_sirius:/label-studio/data/data_sirius" `
  -v label_studio_data:/label-studio/data `
  heartexlabs/label-studio:1.20.0
```
Запуск: `.\start_ls_docker.ps1` в PowerShell (разрешите выполнение: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, если нужно).

## Устранение неисправностей
- Docker не запускается: Проверьте, что Docker Desktop работает (иконка в трее).
- Папки не видны: Убедитесь в правильных путях volumes (используйте абсолютные, если relative не работает: `-v "d:/tinkoff/tbank_logo_detector/tbank_official_logos:/label-studio/data/tbank_official_logos"`).
- Порт занят: Измените на `-p 8081:8080` и откройте localhost:8081.
- data_sirius пустая: Создайте папку и добавьте изображения.

После разметки и экспорта файл `refs_ls_coco.json` готов для импорта в Colab/YOLO.