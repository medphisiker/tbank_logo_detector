# Ускоренный план для baseline прототипа детектора логотипов Т-Банка

> **Дата**: 16 сентября 2025  
> **Автор**: Sonoma (Kilo Code)  
> **Описание**: Ускоренная версия плана на основе feedback для минимального viable baseline. Фокус на синтетике как основе train/val, confidence-based sampling для реальной тестовой выборки (500 positives + 500 negatives), YOLOv8n (fallback от YOLOv12). Интеграция с fast_solution.md (gen_synth.py, train 10 epochs), yoloe_9logos_pipeline_plan.md (YOLOE улучшение), service.md (FastAPI + Docker). 3 класса: yellow_shield_black_T, white_shield_black_T, purple_shield_white_T. Цель: F1@0.5 >0.8 на synth, >0.6 на test. Timeline: 4-6 часов core + 3 часа доп. Ресурсы: GPU T4. Сдача: GitHub repo с Docker, README, весами.

## Введение
Подход ускорен: LS только для refs (manual bbox 9 PNG в 3 класса), synth для initial train/val (domain randomization), inference synth-модели на data_sirius для sampling high-conf test_set (positives conf>0.7, negatives no det). Сервис: FastAPI /detect с ONNX, OCR filter (Tinkoff reject). Доп: YOLOE для pseudo-labels, DINO+CLIP ensemble. Риски: Synth-reality gap (решаем итерацией 7), noisy sampling (spot-check 10% в LS).

## Основной план (шаги 1-6)

### 1. Поднять Label Studio и разметить refs (30-45 мин)
- Установка: `pip install label-studio` (если нет). Запуск: `label-studio start` (localhost:8080).
- Проект: Object Detection. Config.xml:
  ```
  <View>
    <Image name="image" value="$image"/>
    <RectangleLabels name="label" toName="image">
      <Label value="yellow_shield_black_T" background="yellow"/>
      <Label value="white_shield_black_T" background="white"/>
      <Label value="purple_shield_white_T" background="purple"/>
    </RectangleLabels>
  </View>
  ```
- Импорт: tbank_official_logos/ (9 PNG как изображения).
- Разметка: Bbox щит+T (yellow: logo0-3,6,8 class0; white:1,4,5 class1; purple:7 class2). ~5-10 мин.
- Экспорт: COCO (dataset/refs/refs_ls_coco.json: bbox xywh, category_id 0-2).
- Commit: `git add dataset/refs/ && git commit -m "LS refs annotated"`.

### 2. Синтетика + train/val на синтетике (1-2 часа, GPU)
- Backgrounds: Скачать 200-500 (Unsplash/Pixabay CC0; optional script: requests.get для batch).
- Gen synth: Адаптировать gen_synth.py (fast_solution.md):
  - N=2000 (balanced ~667/class), refs_dir=dataset/refs/, classes_map={0:'yellow...',1:'white...',2:'purple...'}, ref_files_by_class из refs_ls_coco (group bbox).
  - Aug: Random scale 0.15-0.45, rotate -25+25, albumentations (brightness/contrast/hue/GaussNoise/MotionBlur).
  - Output: dataset/images_synth/ (jpg), labels_synth/ (YOLO txt: class cx cy w h).
  - Split: 80/20 train/val_synth (random_split.py).
- data.yaml:
  ```
  train: dataset/images_synth/train
  val: dataset/images_synth/val
  nc: 3
  names: ['yellow_shield_black_T', 'white_shield_black_T', 'purple_shield_white_T']
  ```
- Train: `yolo detect train model=yolov8n.pt data=data.yaml epochs=10 imgsz=640 batch=16 amp=True cache=True` (YOLOv12n fallback: pip install yolov12 if available).
- Val: `yolo val model=runs/detect/train/weights/best.pt data=data.yaml` (mAP@0.5 >0.8 цель). Сохранить best_synth.pt.
- Commit: `git add dataset/ runs/ data.yaml && git commit -m "Synth gen + initial train"`.

### 3. Confidence-based тестовая выборка из data_sirius (1 час)
- Inference: `yolo predict model=best_synth.pt source=data_sirius/ --save-txt --save-conf --conf=0.25 --project=runs/predict_synth --device=0` (batch GPU).
- Sampling (sample_test_set.py):
  - Positives: 500 imgs где max_conf >0.7 (balanced classes if possible; convert txt to COCO labels).
  - Negatives: 500 imgs где no detections (all conf <0.25).
  - Output: test_set/images/ (1000 jpg), test_set/labels/ (COCO test_coco.json с bbox/cls для positives, empty для negatives).
  - Spot-check: 10% (100 imgs) в LS для sanity (manual verify ~20 мин).
- test.yaml: nc=3, names=..., train/test split (но val на full test_set).
- Val: `yolo val model=best_synth.pt data=test.yaml` (F1@0.5 >0.6 initial). Визуалы: evaluate.py (overlay GT/pred, 10-20 PNG в validation/visuals/).
- Commit: `git add test_set/ test.yaml runs/predict_synth/ && git commit -m "Confidence sampling test_set"`.

### 4. Сервис вокруг модели (1 час)
- Экспорт: `yolo export model=best_synth.pt format=onnx opset=12 imgsz=640` (best_synth.onnx).
- FastAPI (src/main.py):
  - Импорт: fastapi, uvicorn, onnxruntime, opencv-python, easyocr, pydantic.
  - Models: BoundingBox/Detection/DetectionResponse из task.md.
  - /detect POST: UploadFile, preproc (cv2 letterbox RGB 0-1), ONNX run (ort_session.run, output boxes/conf), postproc (NMS iou=0.45 conf>0.25, abs bbox int, EasyOCR on crops: reject if "Тинькофф"/"Tinkoff").
  - ErrorResponse для 400/413/500.
- Тест (test_service.py): requests.post с sample img (data_sirius/test.jpg), assert len(detections), latency <10s.
- Docker: Dockerfile (FROM python:3.10-slim, COPY src/requirements.txt, pip install, CMD uvicorn main:app --host 0.0.0.0 --port 8000).
- Build/run: `docker build -t tbank-service . && docker run -p 8000:8000 tbank-service`. Тест curl: `curl -X POST -F file=@test.jpg http://localhost:8000/detect`.
- Commit: `git add src/ docker/ test_service.py && git commit -m "FastAPI service + Docker"`.

### 5. Описание и подготовка к сдаче (30 мин)
- README_baseline.md (корень):
  - Установка: pip requirements.txt (ultralytics fastapi onnxruntime etc.).
  - Шаги: 1-6 выше (команды), метрики (synth F1, test F1), артефакты (best_synth.pt/onxx, test_coco.json, visuals).
  - API: /detect (Pydantic spec), latency <10s T4.
  - Docker: build/run, test curl.
  - Ссылки: data_sirius (password 7*5\Lq=Oik), docs/fast_solution.md.
- Docs: Обновить fast_solution.md (добавить sampling, synth val).
- Commit: `git add README_baseline.md docs/ && git commit -m "README + prep submit"`.

### 6. Создать готовый репо (15 мин)
- GitHub: New repo tbank_logo_detector_baseline (public).
- Push: `git remote add origin https://github.com/user/tbank_logo_detector_baseline.git && git push -u origin main`.
- Структура: src/ (main.py), train/ (gen_synth.py sample_test_set.py data.yaml), annotation/ (refs_ls_coco.json), validation/ (test_set/ evaluate.py visuals/), docker/ (Dockerfile), docs/ (all MD), README_baseline.md.
- Release v1.0: Attach best_synth.onnx, test_coco.json (zip if large), tag.
- Сдача: README с "docker run ...", API test, метрики. Optional: Docker Hub push tbank-service:v1.

## Дополнительные пункты (если время, 2-4 часа)

### 7. Улучшение разметки с YOLOE + новая итерация (1-2 часа)
- Bulk: tbank_yoloe_bulk_inference.py (yoloe_plan): Load refs_ls_coco bbox/cls, hybrid prompts (visual + text desc), predict data_sirius conf>0.5, export pseudo_coco.json (3 classes).
- Merge: merge_pseudo.py (union synth labels + YOLOE, weight=conf; filter CLIP sim>0.65 in FiftyOne quick view).
- Retrain: data_merged.yaml (train: merged images/labels), `yolo train model=best_synth.pt data=data_merged.yaml epochs=20`.
- Update: Export new.onnx, swap in service, re-val on test (F1 +5-10%).
- Commit: `git add merged/ new_model.onnx && git commit -m "YOLOE iteration"`.

### 8. Разметка с Grounding DINO + CLIP embeddings, объединение (1-2 часа)
- DINO: grounding_dino_bulk.py (prompts: "yellow shield black T logo", etc. per class), output dino_coco.json (bbox conf).
- CLIP: FiftyOne on dino_coco: compute_patch_embeddings (clip-vit-base), sim to class text emb >0.65, filter high-quality.
- Ensemble: merge_all.py (rules labeling_strategy: ensemble_score = 0.4*clip_sim + 0.3*yolo_conf + 0.3*dino_conf >0.7; IoU>0.5 union, hard-neg Tinkoff as background).
- Add hard-neg: Collect 200 Tinkoff imgs (web search), label as negative class or background.
- Final train: On full merged (epochs=30), update service to new_model.onnx.
- Val: Re-run on test, visuals FP/FN analysis.
- Commit: `git add dino/ ensemble/ final_model.onnx && git commit -m "DINO+CLIP ensemble"`.

## Timeline и риски
- Core (1-6): 4-6 часов (GPU для train/predict ускоряет x3).
- Доп: +3 часа.
- Риски: Synth gap → low test F1 (решаем 7); Noisy sampling → manual spot-check; YOLOv12 unavailable → YOLOv8n (mAP -5%). Мониторинг: evaluate.py per step.
- Метрики: Synth mAP>0.8, test F1>0.6 initial, >0.75 after доп.
- Интеграция: Ссылки на docs/fast_solution.md (gen_synth), yoloe_analysis.md (YOLOE), service.md (API/Docker).

Готов к реализации. Версия 1.0.