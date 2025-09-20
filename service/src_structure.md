# Структура файлов сервиса

## Обзор структуры

```
service/
├── src/                          # Исходный код Python
│   ├── __init__.py              # Пакет service
│   ├── main.py                  # FastAPI приложение
│   ├── models.py                # Pydantic модели
│   ├── config.py                # Конфигурация
│   ├── image_processor.py       # Обработка изображений
│   ├── detector.py              # Интерфейс к модели YOLOv11
│   └── utils.py                 # Вспомогательные функции
├── models/                      # Файлы моделей
│   └── tbank_detector.pt        # Обученная модель YOLOv11
├── requirements.txt             # Зависимости Python
├── Dockerfile                   # Docker образ
├── docker-compose.yml           # Локальная разработка
└── README.md                    # Документация
```

## Описание файлов

### src/main.py
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .models import DetectionResponse, ErrorResponse
from .detector import TBankDetector
from .config import Settings

app = FastAPI(title="T-Bank Logo Detector")

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    """Детекция логотипа Т-Банка на изображении"""
    # Валидация файла
    # Предобработка
    # Детекция
    # Постобработка
    # Возврат результата
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### src/models.py
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")
```

### src/config.py
```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Настройки сервиса"""

    # Модель
    model_path: str = "models/tbank_detector.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    input_size: int = 640

    # Производительность
    device: str = "auto"  # auto, cpu, cuda
    max_image_size_mb: int = 20

    # Сервис
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
```

### src/image_processor.py
```python
from PIL import Image
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    """Обработка изображений для модели"""

    def __init__(self, input_size: int = 640):
        self.input_size = input_size

    def validate_image(self, image: Image.Image) -> bool:
        """Валидация изображения"""
        # Проверка формата, размера, качества
        pass

    def preprocess(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Предобработка изображения"""
        # Resize, normalize, convert to tensor
        pass

    def postprocess_coordinates(self, detections: list, original_size: Tuple[int, int]) -> list:
        """Конвертация координат в абсолютные"""
        # Scale coordinates back to original image size
        pass
```

### src/detector.py
```python
from ultralytics import YOLO
import torch
from typing import List, Dict, Any

class TBankDetector:
    """Детектор логотипов на базе YOLOv11"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """Детекция логотипов"""
        # YOLO inference
        # Return list of detections with bbox, confidence, class
        pass

    def warmup(self, num_iterations: int = 3):
        """Прогрев модели"""
        # Run inference several times to warm up GPU
        pass
```

### src/utils.py
```python
import logging
from functools import wraps
from time import time
from typing import Callable, Any

def timing_decorator(func: Callable) -> Callable:
    """Декоратор для замера времени выполнения"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_file_format(filename: str) -> bool:
    """Проверка поддерживаемого формата файла"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return any(filename.lower().endswith(fmt) for fmt in supported_formats)
```

## Зависимости (requirements.txt)

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
pillow==10.1.0
numpy==1.24.3
torch==2.1.0
ultralytics==8.0.228
python-multipart==0.0.6
```

## Dockerfile

```dockerfile
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY src/ ./src/
COPY models/ ./models/

# Создание непривилегированного пользователя
RUN useradd --create-home --shell /bin/bash app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## docker-compose.yml

```yaml
version: '3.8'

services:
  tbank-detector:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Переменные окружения

```bash
# Модель
MODEL_PATH=models/tbank_detector.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
INPUT_SIZE=640
DEVICE=cuda

# Производительность
MAX_IMAGE_SIZE_MB=20

# Сервис
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Логирование
LOG_LEVEL=INFO
```

## Примеры использования

### curl
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

### Python requests
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )

print(response.json())
```

### Response example
```json
{
  "detections": [
    {
      "bbox": {
        "x_min": 100,
        "y_min": 200,
        "x_max": 300,
        "y_max": 400
      }
    }
  ]
}