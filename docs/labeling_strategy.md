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
