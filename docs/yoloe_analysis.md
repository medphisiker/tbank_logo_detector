
# Анализ модели YOLOE для предварительной разметки логотипов Т-Банка

> **Дата**: 16 сентября 2025  
> **Автор**: Sonoma (Kilo Code)  
> **Источник**: Официальная документация Ultralytics [YOLOE](https://docs.ultralytics.com/ru/models/yoloe/). Анализ фокусируется на использовании для предварительной разметки изображений из `data_sirius` на основе reference image (образец логотипа Т-Банка).

## Описание модели
YOLOE (YOLO for Everything) — продвинутая модель на базе YOLOv10/YOLO11 для open-vocabulary detection и instance segmentation. Поддерживает текстовые, визуальные и prompt-free режимы. Ключевой модуль — SAVPE (Semantic-Activated Visual Prompt Encoder), позволяющий one-shot detection: модель извлекает семантические признаки из reference image (логотип Т-Банка), чтобы автоматически детектировать и сегментировать похожие экземпляры на множестве изображений без переобучения (zero-shot) или с минимальной донастройкой (fine-tune). Идеально для задачи: точность ~52.6% mAP на COCO, +3.5 AP над YOLO-World на LVIS в open-vocab режиме.

## Возможности для reference-based разметки
- **One-shot detection**: На основе одного reference image модель находит похожие объекты (логотипы) по визуальному сходству (текстуры, формы, цвета). SAVPE кодирует признаки активации и семантики — полезно для уникального дизайна логотипа Т-Банка (щит с T).
- **Instance segmentation**: Bbox + пиксельные маски для точной разметки, упрощая экспорт в COCO/YOLO для LS/FiftyOne.
- **Zero-shot подход**: Предобучена на LVIS/Objects365 (>1200 классов). Укажите reference с bbox — модель найдет похожие без fine-tune. Для data_sirius (множество фото/скринов): batch-inference, ~130 FPS на GPU T4 (6.2 мс/изображение 640x640).
- **Fine-tune подход**: Если zero-shot недостаточно (вариации освещения/углов в data_sirius), дообучите linear probing (заморозка backbone, head only). Требует ~1 GPU, epochs=80, 100-1000 размеченных изображений. Точность +0.1-5% mAP, время обучения в 4 раза меньше YOLOv8.
- **Применение к data_sirius**: Загрузите reference (чистый логотип), примените на папке — авто-аннотации для быстрой разметки. Обработка тысяч изображений масштабируема.

## Ключевые параметры
- **model**: Веса, напр. `yoloe-11l-seg.pt` (large, 26M params; варианты: s/m/l для nano/medium/large).
- **visual_prompts**: Dict с `bboxes` (numpy array координат [x1,y1,x2,y2] для reference, нормализованные 0-1 или пиксели) и `cls` (array ID классов, начиная с 0; для одного логотипа — [0]).
- **refer_image**: Путь к эталонному изображению (jpg/png; если source — видео, использует первый кадр).
- **source**: Целевые изображения/папка (data_sirius/*.jpg).
- **imgsz**: Размер входа (640 по умолчанию, для логотипов — 416-800 для баланса).
- **conf**: Порог confidence (0.25 по умолчанию; для логотипов — 0.5+ чтобы минимизировать FP).
- **iou**: NMS порог (0.7; для overlapping логотипов — 0.45).
- **predictor**: `YOLOEVPSegPredictor` для visual mode (автоматически для refer_image).
- **device**: '0' для GPU, 'cpu' для CPU.
- **save**: True для сохранения аннотаций (bbox/masks в YOLO формате).

Для fine-tune: `data` (YAML с путями к train/val, классы: ['tbank_logo']), `epochs=50-100`, `freeze` (заморозка слоев для linear probing), `trainer=YOLOEPESegTrainer` (для seg) или `YOLOEPETrainer` (для detect).

## Примеры кода для inference
Установка: `pip install -U ultralytics`

**Zero-shot inference с reference для data_sirius (Python):**
```python
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Инициализация модели (large для точности)
model = YOLOE("yoloe-11l-seg.pt")

# Visual prompts: bbox вокруг логотипа на reference (координаты в пикселях; адаптируйте под ваше reference)
visual_prompts = dict(
    bboxes=np.array([[x1, y1, x2, y2]]),  # Пример: [100, 50, 300, 200] для логотипа
    cls=np.array([0])  # ID класса для логотипа Т-Банка
)

# Batch-inference на data_sirius (папка с изображениями)
results = model.predict(
    source="data_sirius/",  # Папка или список путей
    refer_image="reference_tbank_logo.jpg",  # Эталонное изображение
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    imgsz=640,
    conf=0.5,
    iou=0.7,
    save=True,  # Сохранит аннотации в runs/detect/predict/
    project="tbank_detections"  # Папка для результатов
)

# Просмотр/экспорт: results[0].boxes для bbox, results[0].masks для сегментации
for r in results:
    print(f"Детекций: {len(r.boxes)}, Confidence: {r.boxes.conf}")
    r.show()  # Показать с аннотациями
```

**CLI inference (zero-shot):**
```
yolo predict model=yoloe-11l-seg.pt source=data_sirius/ classes=tbank_logo refer_image=reference_tbank_logo.jpg visual_prompts='{"bboxes": [[100,50,300,200]], "cls": [0]}' conf=0.5 save=True
```
(Для visual prompts CLI ограничен; предпочтите Python.)

**Fine-tune на data_sirius (если нужно дообучить; сначала разметьте ~100 изображений в YOLO формате):**
```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

model = YOLOE("yoloe-11l-seg.pt")
results = model.train(
    data="data_sirius.yaml",  # YAML: path: data_sirius/images, train: labels/train, val: labels/val, nc:1, names: ['tbank_logo']
    epochs=80,
    imgsz=640,
    batch=16,
    trainer=YOLOEPESegTrainer,  # Для segmentation
    freeze=[0,1,2]  # Заморозка backbone для linear probing
)
# После: model.predict(...) как выше, но с повышенной точностью
```

## Интеграция с Label Studio ML Backend
Label Studio (LS) поддерживает ML backends для авто-разметки. Создайте кастомный backend на базе Ultralytics YOLOE для предсказаний на основе reference (из проекта LS). Backend — Python класс, наследующий `LabelStudioMLBase`.

**Пример кода backend (label_studio_ml/yoloe_backend.py):**
```python
import os
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import numpy as np

class YOLOEBBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOEBBackend, self).__init__(**kwargs)
        self.model = YOLOE("yoloe-11l-seg.pt")  # Или fine-tuned модель
        self.classes = ['tbank_logo']  # Классы для LS
        # Reference из LS проекта (загрузите из tasks или fixed)
        self.refer_image = "reference_tbank_logo.jpg"  # Или динамически из config
        self.visual_prompts = dict(
            bboxes=np.array([[100, 50, 300, 200]]),  # Адаптируйте
            cls=np.array([0])
        )

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            img_url = task['data']['image']  # URL изображения из LS
            # Скачайте img_url локально если нужно (self.download_task_data)
            source = self.download_task_data(task, 'image')  # Локальный путь

            results = self.model.predict(
                source=source,
                refer_image=self.refer_image,
                visual_prompts=self.visual_prompts,
                predictor=YOLOEVPSegPredictor,
                conf=0.5,
                save=False
            )

            # Формат LS: from_name='label', to_name='image', type='rectanglelabels' или 'brushlabels' для seg
            ls_preds = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        ls_preds.append({
                            'from_name': 'label',  # Из LS config
                            'to_name': 'image',
                            'type': 'rectanglelabels',  # Или 'polygon' для masks
                            'value': {
                                'x': (x1 / r.orig_shape[1]) * 100,  # % координаты
                                'y': (y1 / r.orig_shape[0]) * 100,
                                'width': ((x2 - x1) / r.orig_shape[1]) * 100,
                                'height': ((y2 - y1) / r.orig_shape[0]) * 100,
                                'rectanglelabels': [self.classes[0]],
                                'score': conf
                            },
                            'score': conf
                        })
                        # Для масок: конвертируйте r.masks в polygons (используйте cv2 или skimage)

            predictions.append({'result': ls_preds, 'score': 0.9})
        return predictions

# Запуск: label-studio-ml start yoloe_backend.py --script yoloe_backend:YOLOEBBackend --port 9090
```
- **Настройка LS проекта**: В XML config добавьте `<View><Image name="image" value="$image"/><RectangleLabels name="label" toName="image"><Label value="tbank_logo"/></></View>`.
- **Использование**: В LS подключите backend (http://localhost:9090), reference загрузите в tasks или config. Backend авто-разметит изображения из data_sirius при импорте.
- **Для seg**: Исп