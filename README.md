README.md с инструкциями: как собрать/запустить Docker (GPU), как скачать веса (ссылка), примеры curl запросов, объяснение pipeline и валидации.

Dockerfile:

Базовый: nvidia/cuda:11.8-runtime или nvcr.io образ; установить Python deps, ultralytics, torch; копировать weights.

CMD запускает uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1.

Пример запуска (GPU):