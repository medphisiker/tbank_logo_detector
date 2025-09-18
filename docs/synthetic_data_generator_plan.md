# План развития генератора синтетических данных для детекции логотипов Т-Банка

## Общий краткий план

Генератор синтетических данных (data_preparation/synthesis) предназначен для создания разнообразного и реалистичного датасета изображений с логотипами Т-Банка (3 класса: yellow, white, purple), чтобы обогатить малый набор реальных референсов (9 изображений) и улучшить обучение детектора (YOLOv8/YOLOE). Текущая версия генерирует простые изображения с одним логотипом на случайном фоне, но для достижения цели проекта (F1-score >0.85 на валидационном сете, robustness к реальным фото из data_sirius) нужно добавить реализм, hard-negatives и контролируемые взаимодействия.

**Цель улучшений**: Создать датасет размером 5000–10000 изображений (1500–3000 на класс, с multi-instance), имитирующий реальные сценарии (разные фоны, окклюзии, distractors как Tinkoff-логотипы). Это повысит recall/precision на small objects, снизит FP на похожих брендах и ускорит пайплайн (pretrain synthetic → pseudo-labeling → fine-tune).

**Ключевые улучшения**:
1. Высококачественные фоны с ресайзом для вариативности разрешений.
2. Hard-negatives (distractors) без разметки для обучения игнорированию.
3. Multi-logo размещение с контролем IoU для окклюзии.
4. Расширенные аугментации для realism.
5. Балансировка по классам.
6. Случайные позиции с защитой от обрезки.

**Ожидаемый эффект**: +15–25% mAP@0.5 на dev-set (на основе Ultralytics/arXiv рекомендаций). Датасет в YOLO-формате, с метаданными (JSON: weights=0.7 для synthetic в обучении). Интеграция: merge с real COCO для full train-set.

Разработка: Обновить gen_synth.py (основной), download_backgrounds.py, добавить папки (background_objects). Docker-совместимо. Тестирование: FiftyOne для кластеризации, mAP на 100 real-like.

## 1. Высококачественные фоны с случайным ресайзом

### Что делает, цель и для чего
Генератор фонов скачивает изображения в высоком разрешении (1920x1920, в 3 раза больше стандартного 640x640 для YOLO), а при сборке синтетики случайно уменьшает их размер (scale_down 0.5–1.0). Это создает фоны с детализированными текстурами (шум, паттерны), но адаптированные к типичным разрешениям реальных фото.

**Цель**: Имитировать domain gap между high-res источниками (интернет-фото) и low-res входами (мобильные снимки, веб). Разнообразие фонов (офис, документы, улица, банк) отражает data_sirius — случайный набор из интернета.

**Для чего**: Повышает generalization модели: детектор научится работать на размытых/сжатых изображениях, улучшая recall на small logos в noisy backgrounds. Без этого synthetic будет "слишком чистым", приводя к overfitting на идеальные фоны.

### Как реализовать
- **download_backgrounds.py**: Изменить img_size=1920. Добавить опцию --thematic (default=False): использовать Unsplash API (бесплатный ключ) с queries=["office document", "street sign", "bank interior", "random texture"] для 1000+ фонов (mix random + thematic, 70/30). Сохранять в backgrounds/ с поддиректориями (e.g., office/, street/). Limit=1000 total.
- **gen_synth.py**: Перед наложением: W_orig, H_orig = bg.size; scale_down = random.uniform(0.5, 1.0); bg = bg.resize((int(W_orig * scale_down), int(H_orig * scale_down))). Затем размещать logos относительно resized. Параметр --min_scale_down=0.5 (для контроля).
- **Дополнения**: Лог: "Background resized to {new_size} from {orig_size}". Тестировать: 30% с scale_down <0.7 для "мобильных" случаев. Deps: requests для API (добавить в pyproject.toml если нужно).

## 2. Hard-negatives: Объекты фона (distractors)

### Что делает, цель и для чего
Добавляется папка background_objects/ с 200–500 PNG-объектами (похожие на logos, но не целевые: Tinkoff-логотипы, щиты других банков (Sber, VTB), generic emblems, текст "T"). На каждый фон случайно накидывается 0–15 таких объектов (без разметки, как background), с аугментациями и случайным размещением (возможны overlaps между negs).

**Цель**: Создать distractors, которые модель должна игнорировать, фокусируясь на целевых logos. Tinkoff как ключевой negative (похожий щит с "T").

**Для чего**: Снижает false positives (FP) на похожих брендах — частая проблема в zero-shot (docs: hard-negative mining в labeling_strategy.md). Без negatives модель переобучится на "чистых" фонах, путая Tinkoff с Т-Банком (precision <0.85). Это учит background suppression для crowded scenes.

### Как реализовать
- **Подготовка**: Создать data_preparation/synthesis/background_objects/ (скачать manually или скрипт: requests с Freepik/official sites). Классы: 40% Tinkoff-variants, 30% other banks, 30% generic (alpha PNG).
- **gen_synth.py**: После фона: num_neg = random.randint(0,15); for _ in range(num_neg): obj = random.choice(background_objects); obj_aug = apply_neg_aug(obj); pos = random_pos(fg); paste без лейбла (IoU-check только с будущими logos, <0.5). 30% изображений без negs.
- **Дополнения**: Аугментации для negs (см. п.4). Параметр --neg_max=15. Лог: "Added {num_neg} negatives". Интеграция: В YOLO train — blank labels для negatives.

## 3. Multi-logo размещение с контролем IoU-перекрытия

### Что делает, цель и для чего
На фон (после negs) размещается 1–10 целевых логотипов (по классам). Для каждого: случайная позиция/scale/rotate; проверка IoU с предыдущими logos (< threshold=0.4); max 5 попыток relocation если overlap. Это контролирует partial occlusion (перекрытия).

**Цель**: Имитировать multi-instance сцены (несколько logos на фото, e.g., коллаж/реклама) и легкие окклюзии (друг друг друга или negs).

**Для чего**: Улучшает multi-object detection и robustness к overlaps (реалистично для data_sirius: случайные вставки). Без контроля — слишком сильные окклюзии (low recall); с контролем — модель научится partial views, повышая F1 на crowded images (план: occlusions для "наклейки").

### Как реализовать
- **gen_synth.py**: После negs: num_logos = random.randint(1,10); placed_bboxes = []; for _ in range(num_logos): attempts=0; while attempts<5: pos,scale,rot = random; bbox_new = calc_bbox(pos,scale,rot); if all(iou(bbox_new, prev) < 0.4 for prev in placed_bboxes): paste; placed_bboxes.append(bbox_new); break; attempts+=1; else: skip. Threshold --iou_threshold=0.4.
- **Дополнения**: 20% случаев allow higher IoU (0.5–0.7) для strong occlusion (метка "occluded" в JSON). IoU-функция: simple intersection over union. Лог: "Placed {num_logos} logos, {skipped} due to overlap".

## 4. Расширенные аугментации и трансформации

### Что делает, цель и для чего
Аугментации применяются поэтапно (фон → negs → logos → final image) с расширенным набором (albumentations). Вероятности 0.3–0.7, чтобы 20–30% без aug. Это добавляет деградации: освещение, шум, distortion, сжатие.

**Цель**: Имитировать реальные условия (тени, блики, JPEG-артефакты, деформации) для domain randomization.

**Для чего**: Делает synthetic похожим на data_sirius (internet-mix), снижая overfitting. Улучшает robustness: модель детектирует logos в low-quality фото (mAP +10–15%, как в Ultralytics guides).

### Как реализовать
- **Compose в gen_synth.py**: Отдельные pipelines:
  - **Фон** (p=0.5–0.7): RandomBrightnessContrast(0.3), GaussNoise(10-50), MotionBlur(3), GaussianBlur(3), Solarize(128), RandomShadow(1), JPEGCompression(70-95).
  - **Объекты фона (negs)** (p=0.6): HueSaturationValue(10/20/20), RandomRotate90, GaussNoise(5-30), Perspective(0.05-0.1), RandomErasing(0.02-0.1), GaussianBlur(3).
  - **Логотипы** (p=0.7): RandomBrightnessContrast(0.2), Rotate(30), ShiftScaleRotate(0.1/0.2/15), Perspective(0.05-0.15), HueSaturationValue(5/10/15), GaussNoise(5-20), ElasticTransform(1/50).
- **Дополнения**: Применять после всех pastes (final aug). Параметр --aug_level=medium (low/medium/high для p). Метаданные: "aug_types": ["blur", "shadow"] для weighted loss.

## 5. Балансировка по классам

### Что делает, цель и для чего
Генерация поровну по 3 классам (yellow/white/purple: N//3 на класс). В multi-logo — uniform комбо (e.g., random mix без bias).

**Цель**: Обеспечить равное представление variants (tbank_official_logos.md: yellow dominant, но balance нужен).

**Для чего**: Избежать class imbalance (per-class recall >0.85, цель в task.md). Помогает в fine-tune: модель не "забывает" rare классы (purple).

### Как реализовать
- **gen_synth.py**: targets = {0: N//3, 1: N//3, 2: N//3}; while sum(targets.values())>0: cls = random_class_with_remaining(targets); generate_one_logo(cls); targets[cls]-=1. Для multi: sample без replacement до num_logos.
- **Дополнения**: Лог: "Balanced: {counts}". Если N%3 !=0 — distribute remainder.

## 6. Случайный диапазон позиций с контролем обрезки

### Что делает, цель и для чего
Позиции логотипов/neg полностью случайные (uniform по canvas), но с check: IoU bbox с full-image >0.8 (≥80% visible). Max 10 attempts, else center или skip.

**Цель**: Разместить объекты anywhere (край, центр), имитируя random вставки в data_sirius.

**Для чего**: Улучшает boundary detection (logos на краю фото), без "обрезанных" (low-quality labels). Повышает adaptability к compositions.

### Как реализовать
- **gen_synth.py**: В placement: attempts=0; while attempts<10: x = random(0, W-w); y=random(0,H-h); bbox = [x/W, y/H, w/W, h/H]; if iou(bbox, [0,0,1,1]) >0.8: place; break; attempts+=1; else: center_pos.
- **Дополнения**: Threshold --clip_threshold=0.8. Для negs — looser (0.6, allow partial off-edge).

## Размер датасета и интеграция
- **Рекомендация**: 5000–10000 изображений (1500–3000/класс, 80/10/10 splits). На основе Ultralytics/arXiv: sufficient для pretrain small objects, с multi-instance (>10K instances).
- **Интеграция**: Новый скрипт merge_synth_real.py (COCO/YOLO merge, weights=0.7 для synthetic). Docker: mounts для new folders. Тестирование: gen 1000 → FiftyOne clusters → mAP на dev-100.