# Data Preparation — Synthesis Scripts

## Обзор

В директории `data_preparation/synthesis/` реализован пайплайн генерации синтетических данных для детекции логотипов Т-Банка. Основан на разработанном нами Python-пакете `synthesis_generator`, который предоставляет модульную архитектуру для создания разнообразных синтетических изображений.

## Быстрый старт

Для запуска пайплайна синтеза данных используйте отдельные скрипты для каждого этапа. Все скрипты используют готовый Docker-образ `medphisiker/tbank-synth:latest` для обеспечения консистентности среды выполнения.

**Подробная документация по Docker:** [docker/README.md](docker/README.md)

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

**Возможности пайплайна:**
- Multi-logo размещение (1-10 логотипов на изображение)
- Hard-negatives (distractors) для улучшения обучения
- Случайный ресайз фонов (0.5-1.0 от оригинала)
- Расширенные аугментации (ElasticTransform, JPEG compression, shadows)
- Балансировка по классам (purple/white/yellow)
- Контроль IoU-перекрытий

## Дополнительная документация

- **[Docker](docker/README.md)** — Подробная документация по использованию Docker-образа
- **[Scripts](Scripts.md)** — Описание всех скриптов (BAT, Python, Shell)
- **[Examples](Examples.md)** — Примеры использования и кастомные сценарии
- **[API Reference](synthesis_generator/README.md)** — Техническая документация Python-пакета

## Closing notes

Для планов и идей см. [3.data_preparation_plan.md](3.data_preparation_plan.md).