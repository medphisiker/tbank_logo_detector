Отлично — сделаем **минималистичный, приоритетный план** чтобы как можно быстрее получить **рабочий baseline YOLOv12** (веса, которые можно сдать), при этом оставив дорожную карту для улучшений позже. Ниже — пошаговый план «сделай и сдавай», с конкретными командами и минимальными скриптами, чтобы вы могли запустить всё быстро.

# Итоговая цель (минимум, который надо сдать)

* Набор файлов: `best_yolov12_baseline.pt` (веса), `data.yaml` (описание датасета), `README_baseline.md`, `val_predictions.json` (результаты на валидации) и \~10–20 visual examples (predictions).
* Модель должна детектить три класса: `yellow_shield_black_T`, `white_shield_black_T`, `purple_shield_white_T` с приемлемым baseline-качЕством (не обязательно финальное).

# Приоритеты (что делать в первую очередь)

1. **Подготовить минимальный тренировочный набор**: референсы (9 logos) + быстро сгенерированная синтетика (\~1–3k изображений).
2. **Быстрая тренировка**: train YOLOv12 (малый вариант: `yolov12n`/nano) на synthetic+refs 10 эпох — это ваш стартовый baseline.
3. **Быстрая валидация**: отложить 100 реальных изображений (dev100) ручной разметки → оценить и сохранить predictions.
4. **Экспорт весов**: отдать `best_yolov12_baseline.pt` + data yaml + README.
5. (Опционально, если время) запустить inference на full dataset и экспортировать pseudo-labels.

---

# Шаги с командами и скриптами

## 0) Окружение — что установить

(используйте virtualenv / conda)

```bash
pip install -U ultralytics fiftyone opencv-python pillow albumentations tqdm
# если YOLOv12 в вашей установке — используйте соответствующий пакет/репозиторий
```

Если в вашей инфраструктуре YOLOv12 недоступен, используйте `yolov8n` как fallback — план и команды те же.

---

## 1) Подготовка refs (Label Studio → COCO/Yolo)

* Экспортируйте 9 ref-логотипов как отдельные изображения + bbox (Label Studio → COCO).
* Создайте структуру проекта:

```
dataset/
  images/
  labels/         # YOLO txt annotations (one .txt per image) или COCO JSON
  refs/           # 9 ref logos (png)
  backgrounds/    # 200–1000 фонов для синтетики (можно взять с openimages/cc0)
```

---

## 2) Быстрая генерация синтетики (paste logos на фоны)

Ниже — быстрый скрипт (Python) для генерации YOLO-format датасета. Сохраните как `gen_synth.py` и запустите.

```python
# gen_synth.py — minimal synthetic paste generator
import os, random, math
from PIL import Image, ImageEnhance
import albumentations as A

refs_dir = "dataset/refs"
bg_dir = "dataset/backgrounds"
out_img = "dataset/images"
out_lbl = "dataset/labels"
N = 2000  # быстро: 2k картинок

os.makedirs(out_img, exist_ok=True)
os.makedirs(out_lbl, exist_ok=True)

aug = A.Compose([
    A.RandomBrightnessContrast(p=0.7, brightness_limit=0.3, contrast_limit=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.HueSaturationValue(p=0.6, hue_shift_limit=15),
])

refs = [os.path.join(refs_dir,f) for f in os.listdir(refs_dir) if f.lower().endswith(('.png','.jpg'))]
bgs = [os.path.join(bg_dir,f) for f in os.listdir(bg_dir) if f.lower().endswith(('.png','.jpg'))]
classes_map = {0:'yellow_shield_black_T', 1:'white_shield_black_T', 2:'purple_shield_white_T'}
ref_files_by_class = {0: refs[:4], 1: refs[4:7], 2: refs[7:]}  # пример деления, подстройте по вашим refs

for i in range(N):
    bg = Image.open(random.choice(bgs)).convert("RGB")
    W,H = bg.size
    cls = random.choice([0,1,2])
    ref = Image.open(random.choice(ref_files_by_class[cls])).convert("RGBA")

    # random scale+rotate
    scale = random.uniform(0.15, 0.45)
    nw = int(W * scale)
    aspect = ref.width / ref.height
    nh = int(nw / aspect)
    ref_t = ref.resize((nw, nh), Image.LANCZOS).rotate(random.uniform(-25,25), expand=True)

    x = random.randint(0, max(0, W - ref_t.width))
    y = random.randint(0, max(0, H - ref_t.height))

    out = bg.copy()
    out.paste(ref_t, (x,y), ref_t)

    # convert to numpy for augmentations
    import numpy as np
    arr = np.array(out)
    auged = aug(image=arr)['image']
    out = Image.fromarray(auged)

    fname = f"synth_{i:05d}.jpg"
    out.save(os.path.join(out_img, fname), quality=90)

    # write YOLO txt (class x_center y_center w h normalized)
    cx = (x + ref_t.width/2)/W
    cy = (y + ref_t.height/2)/H
    bw = ref_t.width / W
    bh = ref_t.height / H
    with open(os.path.join(out_lbl, fname.replace('.jpg','.txt')), 'w') as f:
        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
```

Запустите:

```bash
python gen_synth.py
```

(Если мало фонов — можно быстро скачивать 200–500 изображений с Pixabay/Unsplash/COCO subset.)

---

## 3) Собрать `data.yaml` для тренировки

Файл `data.yaml` (Ultralytics format):

```yaml
train: /path/to/dataset/images  # папка с изображениями
val: /path/to/your/dev100/images
nc: 3
names: ['yellow_shield_black_T','white_shield_black_T','purple_shield_white_T']
```

Если используете COCO — укажите путь к COCO JSON.

---

## 4) Быстрая тренировка YOLOv12 — baseline

Используем малую конфигурацию (nano) и 10 эпох — быстро получить веса.

```bash
# предполагаем, что ultralytics yolo CLI поддерживает yolov12n.pt
yolo task=detect mode=train model=yolov12n.pt data=data.yaml epochs=10 imgsz=640 batch=16 workers=4
```

Если `yolov12n.pt` недоступен — `model=yolov8n.pt` как fallback.

**Рекомендация:** используйте `--cache` (если есть) и `mixed precision` для ускорения.

---

## 5) Быстрая валидация (dev100)

* Отложите вручную 100 реальных изображений (`val`), размеченных вручную.
* После обучения запустить валидацию:

```bash
yolo val model=runs/detect/exp/weights/best.pt data=data.yaml
```

* Сохраните `val_predictions.json` (Ultralytics может сохранять предсказания; иначе запустите `yolo predict` и конвертируйте в COCO json).

---

## 6) Быстрый inference на part/full dataset (если хотите pseudo)

```bash
yolo predict model=runs/detect/exp/weights/best.pt source=/path/to/data_sirius --save-txt --save-conf --conf 0.25
```

* Это сохранит txt per image с bbox и conf → можно конвертировать в COCO.

Простой скрипт-конвертер txt→COCO можно приложить при необходимости.

---

## 7) Что сдавать (минимальный набор)

* `best_yolov12_baseline.pt` — файл весов (в `runs/detect/.../weights/`).
* `data.yaml` — описание dataset.
* `README_baseline.md` — коротко: как воспроизвести (команды), что внутри (количество синтетики, dev100), какие пороги использовались.
* `val_predictions.json` (или zip с `preds/` txt) + 10–20 PNG с отбражёнными предсказаниями.
* (Опционально) `train_logs` (tensorboard/exp folder).

---

# Быстрые улучшения (если остаётся 1–2 часа)

* **Увеличить epochs до 30** (если GPU доступен) — улучшает качество.
* **Добавьте hard negatives** в синтетику (похожий shield без T) — быстро уменьшит FP.
* **Запустить YOLOE hybrid pass** на dev100 и объединить detections: если YOLOE говорит «это ref-like», повысить доверие к метке. (не обязательный шаг для baseline, но полезен).

---

# Что делать если очень мало времени (свободно заменить шаги)

Если времени вообще мало — минимально:

1. Сгенерить 500–1000 synthetic изображений (генерация займёт минуты–час).
2. Train `yolov12n` 5–10 ep.
3. Сохранить веса и README — это ваш baseline.
   (Лучше иметь плохо работающую модель, чем ничего.)

---

# Полезные советы по reproducibility (чтобы можно было «сдать»)

* Зафиксируйте seed в скриптах (numpy, random, torch).
* В README укажите: GPU (тип), команда установки, команда тренировки, куда выгружены веса.
* Добавьте 10–20 примеров ошибок (FP/FN) — покажет зоны для будущих улучшений.

---

# Что я могу прислать прямо сейчас (если хотите)

1. `gen_synth.py` — готовый и чуть более продвинутый (occlusion, shadows).
2. `train.sh` — готовая команда для ultralytics/YOLOv12.
3. `convert_preds_to_coco.py` — скрипт, чтобы быстро превратить `--save-txt` в COCO JSON.

Скажите, хотите ли получить эти файлы прямо сейчас — пришлю полный код для быстрой копипасты.

---

Если кратко:

* **Сделайте синтетику + refs → тренируйте yolov12n на 10 эпох → валидируйте dev100 → сохраните best.pt** — это и будет ваш работоспособный baseline, который вы сможете сдавать и расширять дальше. Хотите, чтобы я прислал `gen_synth.py` + `train.sh` + `convert_preds_to_coco.py` сейчас?
