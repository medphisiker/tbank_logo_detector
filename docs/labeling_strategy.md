# Краткий итог (в одно предложение)

Ваш подход — **комбинация high-recall zero-shot discovery → reference-guided YOLOE → CLIP-эмбеддинги + кластеризация (FiftyOne) → ручная валидация + псевдоразметка → дообучение детектора** — это правильный и масштабируемый semi-/self-supervised пайплайн для быстрой bulk-разметки трёх классов логотипов.

Ниже — формулировка подхода, конкретный пошаговый пайплайн, предложения по улучшениям и детальный анализ плюсов/минусов популярных zero-shot моделей (и как их лучше использовать в вашей задаче).

---

# 1) Формулировка подхода (сжатая)

1. **Discovery (high-recall):** запустить zero-shot детектор (например Grounding DINO / OWL-ViT / SAM+proposal) по `data_sirius` с промптами типа `"logo", "shield", "brand symbol", "text logo"` — собрать *все* потенциальные объекты (bbox / маски) — цель: максимальный recall.
2. **Reference filtering (YOLOE):** прогнать те же изображения через YOLOE с вашими 9 reference (Label Studio → COCO refs), получить detection scores для целевых 3 классов.
3. **Embeddings & similarity:** в FiftyOne получить CLIP-эмбеддинги всех кропов; посчитать cosine-similarity к эталонам (или к centroid’ам class refs). Отметить детекции с similarity >0.65 (или другой порог) как «похожие».
4. **Кластеризация:** кластеризовать эмбеддинги (DBSCAN / HDBSCAN для непредсказуемого числа кластеров; KMeans если ожидаете конкретное число групп) → получить группы: T-Bank like, other banks, garbage.
5. **Псевдоразметка:** собрать bbox’ы с высокой уверенность (YOLOE\_conf \* clip\_sim > threshold) → экспорт в COCO (псевдо-labels).
6. **Fine-tune:** дообучить YOLO (YOLOv8/11/12 или YOLOE) на комбинированном наборе: ручные refs (9), псевдоразметка (с весами качества), и небольшая валидация (100 вручную промаркированных изображений).
7. **Iterate:** реинференс моделью на всем датасете, повторная кластеризация и подправка (human-in-the-loop только на спорных кластерах).

---

# 2) Пошаговый практический пайплайн (конкретно для вас)

(включаю гиперпараметры, скрипты/метрики и правила принятия решений)

**A. Подготовка refs**

* Label Studio → экспорт `refs_ls_coco.json` (xywh, category\_id 0..2).
* Сохранить эталонные кропы (crop каждого logo) в `refs_crops/` для CLIP.

**B. Bulk discovery (high-recall)**

* Модель: Grounding DINO или OWL-ViT / SAM proposals.
* Prompts: `["logo", "brand symbol", "emblem", "shield", "text logo"]`.
* NMS: iou=0.7, keep low conf threshold (e.g., conf>0.2) — цель recall.
* Output: `discovery_detections_coco.json` (bbox + score + image\_id).

**C. YOLOE reference pass (hybrid)**

* Запустить YOLOE hybrid predict с `refs_ls_coco.json` как visual prompts.
* Keep only detections with conf > 0.5 → `yoloe_tbank_detections.json`.

**D. Embeddings в FiftyOne**

* Для всех кропов (discovery ∪ yoloe): получить CLIP-векторы (рекомендация: CLIP ViT-L/14 если есть GPU, иначе ViT-B/16).
* Нормализовать векторы и считать cosine similarity к каждому class-ref centroid.
* Метрика похожести: `cos_sim > 0.65` маркируем «likely class».

**E. Кластеризация**

* Алгоритмы: HDBSCAN (параметры min\_cluster\_size=10) или DBSCAN (eps=0.2–0.35, min\_samples=5).
* Дополнительно: кластеризовать только top-K by area или по score чтобы уменьшить шум.
* Визуализация: FiftyOne grid по кластерам, top-5 representatives per cluster.

**F. Правила выбора псевдо-линий**

* Критерии high-quality pseudo-label:

  * `YOLOE_conf >= 0.6 AND clip_sim >= 0.7` → label weight = 1.0 (реально используем в train).
  * `YOLOE_conf >= 0.5 AND clip_sim >= 0.65` → weight = 0.7 (использовать с осторожностью).
  * Только discovery без YOLOE или низкая clip\_sim → отправить в review / reject.
* Применять фильтры по размеру: area >= 0.5% и <= 80% изображения. Удалять слишком маленькие/слишком большие боксы.

**G. Fine-tune loop (self-training)**

* Итерации: 3–5 циклов:

  1. Train YOLOv8 on `manual_refs + high_quality_pseudo` (use loss weighting using label weight).
  2. Run inference on `data_sirius`.
  3. Recompute CLIP features & clusters; extend pseudo set using improved confidences; human review of new/ambiguous clusters.
* Loss weighting: weight = `clip_sim * yolo_conf` (но нормализовать в \[0.3,1.0]) или weight = `(yolo_conf^alpha)` с alpha≈1.5 для подавления низкокачественных меток.
* Augmentations: Mosaic, MixUp, color jitter, random brightness/contrast — важны из-за вариаций цвета/теней.
* Validation: держите 100–300 вручную промаркированных изображений (разнообразных) для оценки mAP.

**H. QC / Human-in-the-loop**

* Human review только на: кластеры с смешанным составом, кластеры с низким avg\_sim, и картинки с conflict labels (несколько классов overlapping).
* Интерфейс: FiftyOne + Label Studio для массового исправления; экспорт правок → новый COCO.

**I. Метрики и критерии окончания**

* Цель: per-class precision/recall > 0.85, overall mAP\@0.5 > 0.75 на тест-подборке.
* Останов: если после 2 итераций mAP растёт <1% и ручная корректировка <3% от датасета → можно завершать.

---

# 3) Улучшения / рекомендации (конкретно для вашей задачи)

1. **Ensemble discovery:** объединяйте результаты Grounding DINO + SAM proposals + YOLO (open-vocab) — union для recall, пересечение для high-precision.
2. **Mask refinement:** если вам важна инстанс-сегментация щита (T внутри), используйте SAM (segment anything) для улучшения масок, затем конвертируйте mask->bbox. SAM хорошо дополняет bbox-детекторы.
3. **Color/shape heuristics:** щиты имеют характерную форму/цвет — добавить HSV-histogram similarity или простую сверку доминирующего цвета (например purple vs yellow vs white) как дополнительный фильтр при кластеризации. Это уменьшит путаницу из-за теней/градиентов.
4. **OCR step:** для логотипов, где T — текстовый элемент, прогоните кропы через OCR (Tesseract) и проверьте наличие буквы “T”/латинских символов — дополнительный признак.
5. **Adaptive thresholds:** начните с conservative thresholds (clip>0.8, yolo>0.7) для seed набора, потом расширяйте порог при итерациях.
6. **Hard negative mining:** собрать representative «похожие но чужие» (Sberbank и др.) и включить как negative class / background во время finetune, чтобы уменьшить false positives.
7. **Quality weighting:** используйте непрямую шкалу качества для псевдо-аннотаций (score→weight) при обучении, а не binary accept/reject.
8. **Cluster labelling automation:** для явно однородных кластеров автоматически присваивайте label, а для смешанных отправляйте в Label Studio.
9. **Profiling:** измерьте throughput (imgs/s) и время извлечения CLIP в вашем HW; если слишком медленно — батчируйте CLIP inference, используйте mixed precision.

---

# 4) Анализ zero-shot моделей — плюсы/минусы и рекомендации

Ниже — сводная таблица и рекомендации, ориентированные на вашу задачу (логотипы-щит с вариациями цвета/теней).

### Grounding DINO

* **Плюсы:** Отличный open-vocab обнаружитель; высокий recall на новых классах; prompt-based (можно указать «logo», «shield»); быстро интегрируется.
* **Минусы:** bbox порой грубее для очень мелких объектов; может давать много низко-классных кандидатов (шум).
* **Когда использовать:** как основной discovery-детектор для high-recall шага.
* **Best practice:** низкий conf порог (0.2–0.3) для discovery, затем фильтрация CLIP.

### OWL-ViT

* **Плюсы:** Open-vocab локализация через текстовые промпты, часто точен для стилистических совпадений.
* **Минусы:** bbox менее «нативные» (иногда bounding heatmap→bbox), медленнее/требует больше ресурсов.
* **Когда использовать:** как альтернатива Grounding DINO если DINO плохо ловит определённый тип логотипов. Хорош для image-level queries.

### SAM (Segment Anything Model) — как дополнение

* **Плюсы:** супер для получения масок и точных границ; prompt-free proposals (или bounding-box prompt).
* **Минусы:** не даёт классы — нужен CLIP/other для классификации; генерирует много proposals (нужна агрегация).
* **Когда использовать:** рефайн масок для selected bbox'ов (например, чтобы отделить щит от фона / текста). Идеально в связке: discovery bbox → SAM refine mask → clip sim + cluster.

### YOLOE / YOLO-World (visual prompts)

* **Плюсы:** быстрый inference, может принимать референсы (visual prompts) — удобен для hybrid inference. Хорош для real-time/large-scale.
* **Минусы:** open-vocab способности зависят от обучения/архитектуры; recall на неожиданных формах хуже, чем у Grounding DINO.
* **Когда использовать:** как фильтр/оценщик (вы уже используете правильно): YOLOE для confident detections целевых классов.

### DETR / RT-DETR (transformer-based)

* **Плюсы:** хорошая общая точность, устойчив в сложных сценах.
* **Минусы:** медленнее; open-vocab требует дополнительных механизмов.
* **Когда использовать:** если требуется высокое качество bbox в сложных композициях и GPU-ресурсы позволяют.

### CLIP (для retrieval / similarity)

* **Плюсы:** отличный визуально-семантический эмбеддинг; очень полезно для similarity+clustering.
* **Минусы:** может путать стилизованный текст/шрифты и изображения; чувствителен к сильным стилевым трансформациям.
* **Когда использовать:** всегда для потомков/кластера в FiftyOne; рекомендую ViT-L/14 для лучшей дискриминации.

---

# 5) Рекомендации по кластеризации и порогам (практика)

* **CLIP threshold:** 0.65–0.75 как старт; 0.7 рабочее значение. Тестировать на валидации.
* **DBSCAN/HDBSCAN:** eps начальный \~0.25 (после L2 нормировки), min\_samples 5–10; HDBSCAN лучше при переменной плотности.
* **Cluster size handling:** автоматически помечать «малые» кластеры (size < 10) как кандидаты для human review.
* **Similarity aggregation:** вместо сравнения к отдельному ref, сравнивайте к centroid refs per class (mean embedding) — более устойчиво.
* **Feature fusion:** можно объединить CLIP embedding + цвет histogram + aspect ratio (concatenate) перед кластеризацией — улучшает разделение классов, чувствительных к цвету.

---

# 6) Как строить цикл обучения / кластеризации / доразметки (конкретный loop)

1. **Init:** refs (9) → seed model YOLOE fine-tune (very short) на refs только (или use visual prompts).
2. **Discovery:** Grounding DINO run → collect candidates.
3. **Score & Embed:** YOLOE\_conf, CLIP\_sim → assign pseudo labels (using rules выше).
4. **Cluster & Review:** cluster pseudo labels, auto-label homogeneous clusters, manual review of mixed/outliers.
5. **Train:** train YOLOv8 on (refs + accepted pseudo) with weighted labels.
6. **Evaluate & Expand:** run trained model over dataset → new pseudo candidates (lower threshold than prior iterations allowed) → go to step 3.
7. **Stop:** when validation mAP stabilizes and manual corrections small.

Рекомендуемые итерации: 3–5. На каждой итерации расширяйте порог допустимости (т.е. сначала strict, потом softer).

---

# 7) Потенциальные риски и как с ними бороться

* **Скопление ошибочных псевдо-лейблов (confirmation bias):** используйте строгие пороги на первых итерациях + weight-scaling.
* **Избыточный шум от discovery:** применить heuristics (area, aspect ratio, color) и кластеризацию перед human review.
* **Класс-имбаланс:** компенсировать oversampling/weighted losses при тренинге.
* **Ошибки CLIP на стилизованном шрифте:** добавить OCR-проверку как вспомогательный признак.

---

# 8) Полезные утилиты / команды (шаблон)

* **Export COCO pseudo:** создайте `pseudo_coco.json` с поля `score` и `quality_weight`.
* **Train YOLOv8 (пример):**

```bash
# pseudocode
yolo task=detect mode=train data=combined_dataset.yaml model=yolov8n.pt epochs=50 \
    cache=True imgsz=640 batch=16 augment=True label_smoothing=0.0
# при weights: можно передавать sample_weights через custom dataloader (используйте weight = quality_weight)
```

* **FiftyOne:** загрузить detections и embeddings, использовать `.compute_embeddings()` и `view.cluster_by()`.
  (Вам хватит стандартных FiftyOne API — вы уже упоминали FiftyOne и CLIP.)

---

# 9) Итог — что делать прямо сейчас (actionable to-do)

1. **Запустить Grounding DINO** на небольшом подмножестве (\~1000 imgs) → оценить recall.
2. **Получить CLIP-эмбеддинги** для cropped candidates + refs → смотреть распределение cosine similarity.
3. **Подобрать thresholds (clip, yolo\_conf)** на dev-подборке (100 ручных).
4. **Собрать high-quality pseudo (строго по правилам)** → дообучить YOLOv8 быстро (1–3 эпохи) и посмотреть прирост.
5. **Организовать кластерную ревизию в FiftyOne** и настроить Label Studio для ручной правки спорных кластеров.
6. **Итеративно расширять pseudo set** и повторять.

---

Если хотите — могу:

* написать конкретные **скрипты** (Python) для: bulk discovery → CLIP эмбеддинги → DBSCAN кластеризация → экспорт COCO pseudo;
* или сгенерировать ready-to-run **notebook** с примером на 100 изображений (включая параметры CLIP, DBSCAN и правила фильтрации).

Скажите, какой из вариантов предпочитаете — дам код и шаблоны прямо сейчас.

Отлично — давайте по пунктам, прямо и конструктивно.

Короткий ответ на ваш вопрос «Ты убрал YOLOE полностью. Она не нужна?»
**Нет — не нужно удалять YOLOE.** YOLOE — полезный инструмент в вашем пайплайне как *reference-guided / hybrid* фильтр. Я убрал явную обязательность YOLOE в предыдущем кратком плане, потому что предложил основной discovery через Grounding DINO + CLIP, но это не значит, что YOLOE бесполезна. Наоборот — её стоит сохранить и умело объединять с synthetic-YOLO. Ниже — как именно её интегрировать, и как ввести синтетические данные и обучение YOLOv8 в общий loop.

---

# Резюме: где что ставить (архитектура пайплайна)

1. **Discovery (high-recall):** Grounding DINO (низкий порог) → собираем *все* кандидаты.
2. **Synthetic-YOLO (prelabel):** сгенерировать синтетический датасет (paste-логотипы на много фоновых изображений, domain randomization) → обучить YOLOv8 на этом синтетике → быстрый проход по data\_sirius для предразметки.
3. **YOLOE (reference/hybrid):** запустить hybrid inference с вашими референсами — даёт strong signal по соответствию визуальным примерам.
4. **CLIP (retrieval/verification):** считать эмбеддинги кропов и refs, получить cosine similarity.
5. **Ensemble & rules:** объединить сигналы (Grounding DINO, synthetic-YOLO, YOLOE, CLIP) → принять pseudo-labels по жёстким правилам / весам качества.
6. **Human-in-the-loop & cluster review (FiftyOne + Label Studio).**
7. **Fine-tune real YOLO (YOLOv8/11/12):** на комбинированном наборе (refs + high-quality pseudo + вручную отредактированные).
8. **Итерация.**

---

# Почему оставлять YOLOE (и когда она решает проблему)

* **Сильный for visual-prompting:** если у вас есть реальные reference-изображения (9 logos), YOLOE может искать объекты «по визуальному примеру» — это ровно ваше требование.
* **Высокая точность на похожих примерах:** YOLOE хороша в отсеивании похожих, но не ваших логотипов (меньше false positives), в то время как Grounding DINO даёт recall, но может захватывать шум.
* **Быстрая фильтрация:** после discovery YOLOE быстро скажет «это похоже на refs» или «нет» (hybrid).
* **Итого:** YOLOE — идеальный *filtering/verification* модуль в середине пайплайна.

Когда можно временно не использовать YOLOE: если synthetic-YOLO + CLIP дают уже очень высокий precision и вы хотите упростить pipeline на раннем этапе. Но оптимально — оставить.

---

# Как использовать синтетические данные (практика)

Цель: дать YOLOv8 представление о ваших 3 классах в разных условиях, чтобы он мог делать хорошие initial predictions и сократить нагрузку на discovery.

**Генерация синтетики — принципы**

* **Метод:** copy-paste логотипов (ref crops) на разнообразные фоны (urban photos, indoor, textures), с random affine (scale 0.3–1.5, rotation ±25°), perspective warp, blur, noise, brightness/contrast, hue jitter, shadow overlays, occlusion (частично закрытые логотипы), Gaussian noise, JPEG compression.
* **Количество:** старт 5k — 20k изображений (зависит от времени/ресурсов). Для трёх классов целевой balanced набор (примерно равное количество).
* **Разметка:** автоматическое generation COCO: bbox (xywh) и masks если хотите сегментацию.
* **Domain randomization:** много разных фонов + color augmentation, чтобы уменьшить domain gap к реальным фото.
* **Hard negatives:** добавьте «ложные» примеры — похожие щиты других банков (без T или с другим цветом) как negative class / background (или отдельный класс `other_shield`) — поможет снизить FP.

**Простой псевдокод генератора (python / PIL + albumentations):**

```python
# pseudocode
for i in range(N):
    bg = random.choice(backgrounds).copy()
    logo = random.choice(ref_logos).apply_random_transform()
    x,y = random_position(bg, logo.size)
    composite = paste_with_alpha(bg, logo, (x,y), blend=True)
    composite = apply_augmentations(composite)  # blur, brightness, shadow...
    save_image_and_bbox(composite, bbox_for_logo)
```

---

# Обучение synthetic-YOLO (практическая настройка)

* **Model:** YOLOv8 (n/m/s/l — в зависимости от GPU). Start with `yolov8n`/`yolov8s` for speed.
* **Hyperparams рекомендованные:**

  * epochs: 30–50 (synthetic pretrain)
  * imgsz: 640
  * batch: максимальный, который влазит в GPU (16–64)
  * optimizer: SGD/AdamW (в ultralytics default ok)
  * lr: 1e-3 initial, cosine schedule or step down; warmup 3–5 epochs
  * augment: mosaic, mixup, hue/contrast, random crop
* **Checkpointing:** сохраняйте best by mAP\@0.5 на валидации (synthetic\_dev \~500–1000 реальных-like synthetic images).

**Что делать после тренировки synthetic-YOLO:**

* Прогнать всю `data_sirius` → получить `synthetic_yolo_detections.json`. Это даёт *второй* сигнал (помимо DINO). Synthetic-YOLO обычно ловит хорошо те случаи, где логотип близок по стилю к training synth.

---

# Правила объединения сигналов (ensemble rules)

Используем 4 сигнала: `dino_conf`, `synth_yolo_conf`, `yoloe_conf`, `clip_sim`.

Нормализуем все в \[0,1] (clip уже cos in \[-1,1] → map to \[0,1] via (cos+1)/2 or just use cos directly if thresholding on cos). Простой scoring:

```
ensemble_score = w1*clip_sim + w2*yoloe_conf + w3*synth_yolo_conf + w4*dino_conf
```

Рекомендации по весам (start):

* w1 (CLIP) = 0.4
* w2 (YOLOE) = 0.25
* w3 (synthetic-YOLO) = 0.2
* w4 (DINO) = 0.15

**Decision rules (исходный набор):**

* **High-quality pseudo (accept):** ensemble\_score >= 0.75 AND bbox area between 0.5%–60% of image.
* **Medium-quality (accept with weight):** 0.6 <= ensemble\_score < 0.75 → accept but label\_weight = 0.5.
* **Flag for review:** ensemble\_score in \[0.45,0.6) OR cluster of mixed labels.
* **Reject:** ensemble\_score < 0.45.

(Эти пороги настраиваемы; на dev-100 подберите порог, который даёт target precision.)

Также полезно правило *OR*:

* Если `yoloe_conf >= 0.7 AND clip_sim >= 0.65` → accept immediately (хорошая комбинация visual prompts + similarity).

---

# Как подбирать thresholds на dev-подборке (100 ручных)

1. Сделать выводы всех моделей (DINO, synth-YOLO, YOLOE, CLIP).
2. Для каждой детекции в dev100 посчитать TP/FP w\.r.t. ручным аннотациям.
3. Sweep thresholds (grid) для clip\_sim ∈ \[0.5..0.85], yolo\_conf ∈ \[0.4..0.8], ensemble\_score cutoff ∈ \[0.45..0.8].
4. Целевая метрика на dev: precision >=0.85 при приемлемом recall (зависит от вашей цели). Выберите пороги по ROC/PR-кривой: обычно вы выбираете порог, где precision уже \~0.85 и recall maximal.
5. Зафиксировать пороги и применять к whole dataset.

---

# Fine-tune loop (конкретика)

1. **Pretrain** on synthetic (30–50 ep). Save model S.
2. **Infer** S on data\_sirius → candidates S\_detections.
3. **Ensemble + Filter** (rules выше) → build pseudo\_train\_set (HQ + MQ with weights).
4. **Train** real YOLO (larger model yolov8m/large) on (refs + pseudo\_train) for 15–30 epochs, with label\_weight for MQ. Use stronger augmentations. Early stop on val.
5. **Evaluate** on dev100; compare mAP, per-class precision/recall.
6. **If good but not enough:** expand pseudo set with next-tier detections (lower ensemble cutoff), re-train. Repeat 2–4 cycles.
7. **Final model:** optionally combine S and fine-tuned model into an ensemble for inference (non-max suppression across models).

---

# Практические хитрости

* **Hard negative mining:** на synthetic добавьте «похожие, но чужие» логотипы и пометьте их как background; это сильно снижает FP.
* **Area filter:** удаляйте bbox меньшие, чем 0.5% площади или слишком вытянутые (aspect ratio > 6).
* **Color check:** для классов с цветовой разницей (желтый/фиолетовый/белый) добавьте HSV-доп-правило: если средний hue далеко от class\_color → downgrade score. Это поможет при тенях/градиентах.
* **OCR:** если T — четко текстовый элемент, применяйте OCR как дополнительную проверку (на наличие буквы T).
* **Weighted loss:** используйте `instance_weight = ensemble_score` (clipped to \[0.3,1.0]) в даталоадере/CustomLoss.

---

# Оценка ожидаемого эффекта (что вы получите)

* **Синтетический pretrain** обычно даёт значительный стартовый recall и дозволяет synthetic-YOLO ловить «типичные» размещения логотипов.
* **YOLOE** хорошо отрезает похожие, но неверные объекты — снижает FP.
* **Ensemble+CLIP** даёт лучший precision для pseudo-лейблов, что уменьшает риск обучения на мусоре.
* В сумме: быстрее достижение целевого mAP и меньше ручной правки.

---

Если хотите, могу прямо сейчас:

* сгенерировать для вас **шаблонный python-скрипт** для создания синтетики (paste+augment),
* или **скрипт для объединения детекций** (DINO, synth-YOLO, YOLOE, CLIP) и вычисления `ensemble_score` и экспорт COCO pseudo.

Напишите: хотите ли **код генератора синтетики** или **код объединения/фильтрации детекций** — и я дам готовый, запускаемый пример.

