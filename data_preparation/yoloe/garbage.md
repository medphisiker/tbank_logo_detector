Отличный рабочий план — реально минималистичный и одновременно разумный: вы используете YOLOE как «распознаватель по тексту» и как «визуальный фильтр по эталонам», затем превращаете уверенные кропы в реалистичную синтетику с «мусором» (hard negatives) и получаете сильный стартовый датасет для YOLOv12. Ниже — упакованный, практический чек-лист с конкретными порогами, скриптами/псевдокодом и примечаниями по тонкостям, чтобы вы могли действовать быстро и безопасно.

# Краткое резюме вашего workflow

1. YOLOE text-pass: найти *всё подряд* (prompts: logo, shield, emblem, brand, text logo) — low threshold.
2. YOLOE reference-pass: найти *всё похожее на эталоны* — получать confidence.
3. Отобрать top-N (100–500) самых уверенных детекций по score → быстро проверить вручную.
4. Среди остальных детекций взять non-overlapping (IoU<0.3) объекты — просмотреть 100–500 как «мусор» (hard negatives).
5. Генерация синтетики: сначала «мусор» на случайные фоны, потом наложение правильных логотипов (ref + YOLOE-crops) поверх — добавить realistic blending, shadows, occlusion, jpeg noise.
6. Тренировка YOLOv12 на синтетике (и refs).
7. Инференс на всем датасете → собрать pseudo labels.
8. Финальное обучение YOLOv12 на полном наборе (refs + synth + pseudo + вручную отредактированные).

---

# Конкретные параметры / пороги (рекомендации на старт)

* YOLOE discovery pass conf threshold: `conf >= 0.18–0.25` (low) — union всех промптов.
* YOLOE reference pass conf threshold: **выберите топ** по `yoloe_conf` (начать с `>= 0.7`), но сортируйте по убыванию и берите top-N = 100–500.
* IoU для non-overlap: `IoU < 0.3` (чтобы быть уверенным, что «мусор» — не частично перекрывает T-bank логотип).
* Area filter: отбросьте bbox с area < `0.5%` изображения или > `60%` изображения.
* CLIP (опционально) similarity cutoff для автоматического отбора: `clip_sim >= 0.65` (если есть время для проверки).
* Синтетика: сначала 500–2000 изображений (500 — быстрый старт).
* Label weight для синтетики при первом full-train: `0.5–0.7` (не давать 100% доверия синтетике сразу).

---

# Пошаговая инструкция с примерами кода (копипастить и запускать)

> Предполагаю, что у вас есть установленный ultralytics/YOLOE и python environment. Код — минимальный, адаптируйте пути.

### 1) YOLOE — multi text-prompt pass и сохранение детекций

```python
# yoloe_text_pass.py (пример; API может отличаться по версии)
from ultralytics import YOLO
import json, os

model = YOLO("yoloe-small.pt")  # поменяйте на вашу модель
prompts = [
    "logo", "brand logo", "shield", "emblem", "text logo",
    "yellow shield with black T", "white shield with black T", "purple shield with white T"
]

out_file = "yoloe_text_detections.json"
results_all = []

for img_path in list_of_image_paths:  # составьте list_of_image_paths
    res = model.predict(source=img_path, text_prompts=prompts, conf=0.18, iou=0.7)
    # res может содержать boxes, scores, labels; адаптируйте под вашу версию
    # Преобразуем к простому json
    items = []
    for box, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        items.append({"box": box.tolist(), "score": float(score)})
    results_all.append({"image": img_path, "detections": items})

with open(out_file, "w") as f:
    json.dump(results_all, f)
```

(Если API возвращает другую структуру — приведите к вашему формату.)

### 2) Извлечь топ-N confident reference-like кропов

```python
# extract_top_crops.py
import json, cv2, os
from pathlib import Path
from operator import itemgetter

in_json = "yoloe_text_detections.json"
out_dir = "yoloe_confident_crops"
os.makedirs(out_dir, exist_ok=True)

with open(in_json) as f:
    dets = json.load(f)

crops = []
for d in dets:
    img = d['image']
    img_cv = cv2.imread(img)
    H,W = img_cv.shape[:2]
    for det in d['detections']:
        x1,y1,x2,y2 = map(int, det['box'])
        score = det['score']
        area = (x2-x1)*(y2-y1)/(W*H)
        if 0.005 <= area <= 0.6:
            crops.append({"img": img, "box":[x1,y1,x2,y2], "score": score})

# сортируем и берем top N
topN = 300
crops_sorted = sorted(crops, key=itemgetter("score"), reverse=True)[:topN]

for i,c in enumerate(crops_sorted):
    x1,y1,x2,y2 = c['box']
    img_cv = cv2.imread(c['img'])
    crop = img_cv[y1:y2, x1:x2]
    cv2.imwrite(f"{out_dir}/crop_{i:04d}_s{c['score']:.3f}.png", crop)
```

### 3) Non-overlap selection (IoU < 0.3) — получить «мусор»

```python
# get_non_overlap.py
import json, cv2, os
from shapely.geometry import box as shapebox
os.makedirs("yoloe_nonoverlap_crops", exist_ok=True)

# load T-bank detections (you can run YOLOE reference pass that outputs only T-bank candidates)
# Suppose tbank_boxes_dict = {image_path: [ [x1,y1,x2,y2], ... ]}

def iou(b1, b2):
    x11,y11,x12,y12=b1; x21,y21,x22,y22=b2
    xa = max(x11,x21); ya=max(y11,y21); xb=min(x12,x22); yb=min(y12,y22)
    inter = max(0, xb-xa) * max(0, yb-ya)
    a1 = (x12-x11)*(y12-y11); a2=(x22-x21)*(y22-y21)
    denom = a1 + a2 - inter
    return inter / denom if denom>0 else 0

# Iterate through unified detections (from step 1)
with open("yoloe_text_detections.json") as f:
    all_dets=json.load(f)

nonover = []
for d in all_dets:
    img = d['image']
    tboxes = tbank_boxes_dict.get(img, [])  # полученные T-bank bboxes (если есть)
    for det in d['detections']:
        b = det['box']; score = det['score']
        keep = True
        for tb in tboxes:
            if iou(b, tb) >= 0.3:
                keep=False
                break
        if keep and score >= 0.25:
            nonover.append({"img":img, "box":b, "score":score})

# сортируем и сохраняем top M nonover crops
nonover_sorted = sorted(nonover, key=lambda x: x['score'], reverse=True)[:300]
for i,c in enumerate(nonover_sorted):
    x1,y1,x2,y2 = map(int,c['box'])
    img_cv=cv2.imread(c['img'])
    cv2.imwrite(f"yoloe_nonoverlap_crops/non_{i:04d}_s{c['score']:.3f}.png", img_cv[y1:y2, x1:x2])
```

### 4) Быстрый визуальный обзор

* Откройте `yoloe_confident_crops` и `yoloe_nonoverlap_crops` в любой viewer (FiftyOne/VSCode image preview) и удалите очевидные ошибки. Это займёт не много времени, т.к. вы смотрите 200–600 картинок, а не весь датасет.

---

### 5) Синтетический генератор (paste мусор + логотип)

Минимальная логика: для каждого синтета сначала вставляем несколько случайных «мусорных» кропов (чтобы создать clutter), затем поверх вставляем корректный логотип (ref или confident crop).

```python
# synth_from_crops.py (минималистично)
import random, os, cv2, numpy as np
from PIL import Image, ImageFilter, ImageOps

refs = list(Path("dataset/refs").glob("*.png"))
positives = list(Path("yoloe_confident_crops").glob("*.png"))
negatives = list(Path("yoloe_nonoverlap_crops").glob("*.png"))
bgs = list(Path("dataset/backgrounds").glob("*.jpg"))

OUT_IMG = "synth/generated"
os.makedirs(OUT_IMG, exist_ok=True)

def paste_with_shadow(bg, crop, x, y):
    # простой shadow: черное смещённое размытие
    img = bg.copy()
    crop_im = Image.open(crop).convert("RGBA")
    w,h = crop_im.size
    # shadow
    shadow = Image.new('RGBA', crop_im.size, (0,0,0,120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    img.paste(shadow, (x+int(w*0.03), y+int(h*0.05)), shadow)
    img.paste(crop_im, (x,y), crop_im)
    return img

for i in range(1000):  # сколько генерируем
    bg = Image.open(random.choice(bgs)).convert("RGB")
    W,H = bg.size
    # paste several negatives
    num_neg = random.randint(1,4)
    for _ in range(num_neg):
        neg = Image.open(random.choice(negatives)).convert("RGBA")
        scale = random.uniform(0.05,0.3)
        nw=int(W*scale); nh=int(neg.height* nw/neg.width)
        negr=neg.resize((nw,nh))
        x=random.randint(0, max(0, W-nw)); y=random.randint(0, max(0, H-nh))
        bg = paste_with_shadow(bg, negr, x, y)
    # paste positive (logo)
    pos = Image.open(random.choice(positives + refs)).convert("RGBA")
    scale = random.uniform(0.08,0.4)
    nw=int(W*scale); nh=int(pos.height* nw/pos.width)
    posr=pos.resize((nw,nh))
    x=random.randint(0, max(0, W-nw)); y=random.randint(0, max(0, H-nh))
    final = paste_with_shadow(bg, posr, x, y)
    fname=f"{OUT_IMG}/synth_{i:05d}.jpg"
    final.convert("RGB").save(fname, quality=90)
    # write label (YOLO txt) — class index нужно назначить (если pos из refs->extract class)
    # Доп: сохранить provenance: which positive used, which negatives used, coords
```

> Для реалистичности в production используйте Poisson blending (opencv seamlessClone) или готовые библиотеки, но для базовой работы простой paste+shadow уже помогает.

---

### 6) Тренировка YOLOv12 (быстрый baseline)

Сформируйте `data.yaml` с train=папка synth + refs, val=dev100.

Команда (ultralytics CLI, пример):

```bash
yolo task=detect mode=train model=yolov12n.pt data=data.yaml epochs=20 imgsz=640 batch=16 workers=4
```

* Используйте меньший модельный файл (`n` или `s`) для скорости, потом переключайтесь на `m`/`l` если ресурс позволяет.
* При использовании weighted labels — понадобится кастомный даталоадер; сначала можно просто тренировать с uniform labels, затем улучшать.

---

### 7) Полный inference → pseudo labels → финальная тренировка

1. `yolo predict model=best.pt source=/path/to/data_sirius --save-txt --save-conf --conf 0.25`
2. Конвертируйте txt → COCO JSON (есть простые скрипты)
3. Объедините набор: `refs + synth + pseudo_high_quality` (где pseudo отбраны по score/IoU/ensemble)
4. Финальное обучение: epochs 30–50, larger model.

---

# Важные operational детали и советы

* **Provenance:** для каждой синтетической картинки храните meta: `positive_source`, `negative_sources`, `coords`, `yoloe_score`, `clip_sim` (если считаете). Это позволит быстро устранять проблемные подмножества.
* **Dev100 неизменяемый:** держите 100 реальных изображений с ручной разметкой в стороне, используйте их как «единственную правду» для подбора порогов.
* **Weighting:** сначала давайте синтетике и pseudo средний вес (0.5), чтобы модель не «переобучилась» на синтетическом шуме.
* **Hard negatives:** специально въедайте negative примеры в отдельный класс / в background, чтобы уменьшить FP.
* **Кластеризация на кропах:** если много похожих кропов — кластеризуйте и валидация по representatives (быстрее).
* **Оценка:** метрики per-class precision/recall, mAP\@0.5. Сначала следите за precision (чтобы сдаваемая модель не была спамной).

---

# Потенциальные риски и как их избегать

* **Confirmation bias:** если вы только берёте то, что YOLOE считает похожим, модель может «усилить» свои ошибки. Смягчение: CLIP-валидация, human-sanity-check top clusters, добавление hard negatives.
* **Noise accumulation:** большое количество низкокачественной синтетики ухудшит финальную модель — используйте weighting и контрольный dev100.
* **Маскирование реальных вариаций:** не удаляйте разнообразие — сохраняйте и refs, и реальные кропы, и случайные позиции/сложные фоны.

---

# Что делать прямо сейчас (конкретные action items, минимум)

1. Запустить YOLOE text-pass и reference-pass на небольшой подвыборке (1k) — экспорт JSON.
2. Вытянуть top-300 confident crops → быстрый просмотр (30–60 минут максимум).
3. Вытянуть top-300 non-overlap crops → быстрый просмотр (30–60 минут).
4. Сгенерировать 500–1000 синтетических изображений по шаблону выше.
5. Быстрая тренировка YOLOv12 nano 10–20 epoch → получить baseline weights.
6. Сохранить веса и README — можно сдавать.

---

Если хотите, могу сейчас:

* сформировать готовые скрипты `yoloe_text_pass.py`, `extract_top_crops.py`, `get_non_overlap.py`, `synth_from_crops.py`, и `train_baseline.sh` полностью (под ваш FS/пути), или
* помочь подобрать точные prompts/пороги для YOLOE и CLIP на dev100.

Что предпочитаете — прислать готовые файлы или только подсказки по настройке?
