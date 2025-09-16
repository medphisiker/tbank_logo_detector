"""
Пример: поиск похожих детекций в датасете FiftyOne по reference image.

Что делает скрипт:
 1. Загружает/использует Dataset с детекциями (например, предсказания YOLOE) в поле `DETECTIONS_FIELD`.
 2. Вычисляет эмбеддинги для патчей (object patches) и сохраняет их в атрибуте меток (label attribute) `EMBEDDINGS_FIELD`.
 3. Для заданной reference-картинки вычисляет эмбеддинг тем же моделем.
 4. Перебирает все детекции и считает косинусную схожесть между эмбеддингами патчей и эмбеддингом reference.
 5. Возвращает (и выводит) все детекции с похожестью >= порог `threshold` (динамически меняется).

Примечание: для больших датасетов и/или продакшен-скорости рекомендую построить similarity index (LanceDB / Milvus / Qdrant / Elasticsearch / MongoDB Atlas Vector Search) через
fiftyone.brain.compute_similarity() и затем сортировать/фильтровать через sort_by_similarity(). Примеры и справка в документации FiftyOne.
(ссылки в чате).

Требования: fiftyone, torch, torchvision (или другой бекенд для модели эмбеддингов), pillow

Использование:
  - Укажите путь к reference image (REF_PATH)
  - Установите имя датасета или укажите как загрузить его
  - Подправьте DETECTIONS_FIELD на имя поля с детекциями (по умолчанию "predictions")

"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
import fiftyone as fo
import fiftyone.zoo.models as fozm
import fiftyone.brain as fob
from PIL import Image

# -------------------- Параметры --------------------
DATASET_NAME = "my_dataset"            # замените на имя вашего датасета в FiftyOne
# Если датасета нет в FiftyOne, загрузите/создайте его заранее.
DETECTIONS_FIELD = "predictions"      # поле с детекциями YOLOE (список Detection)
EMBEDDINGS_FIELD = "patch_emb"        # куда будем сохранять эмбеддинги у меток (detections[].patch_emb)
BRAIN_KEY = "patch_similarity_index"  # ключ brain для возможного compute_similarity
MODEL_NAME = "clip-vit-base32-torch"  # модель из Model Zoo для эмбеддингов (можно заменить)

# Порог схожести (cosine similarity). 1.0 = идеально совпадает, 0.0 = ортогональны
DEFAULT_THRESHOLD = 0.65
TOP_K = 200  # сколько топовых результатов вернуть (при необходимости)

# -------------------- Утилиты --------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-нормализация векторов по последней оси."""
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная схожесть для 1D-векторов (предполагаем, что они нормализованы)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# -------------------- Основной workflow --------------------

def ensure_dataset(name: str) -> fo.Dataset:
    """Загружает датасет по имени из FiftyOne. Если нет — упадет с ошибкой; ожидается, что пользователь
    заранее загрузил/импортировал датасет и YOLOE-предсказания.
    """
    if name not in fo.list_datasets():
        raise ValueError(f"Dataset '{name}' не найден в FiftyOne. Пожалуйста, загрузите его заранее.")
    return fo.load_dataset(name)


def load_embedding_model(model_name: str):
    """Загружает модель из FiftyOne Model Zoo (или другую, поддерживаемую API).
    Модель должна поддерживать .embed()/ .embed_all() или быть совместимой с compute_patch_embeddings.
    """
    print(f"Loading model {model_name}...")
    model = fozm.load_zoo_model(model_name)
    return model


def compute_patch_embeddings_for_dataset(dataset: fo.Dataset, model, patches_field: str, embeddings_field: str):
    """Вычисляет и сохраняет эмбеддинги для всех object patches (детекций) в указанном поле.
    Эти эмбеддинги будут сохранены в атрибуте меток (detection.<embeddings_field>).
    """
    print(f"Computing patch embeddings into '{embeddings_field}' for field '{patches_field}'... (may take time)")
    # Метод compute_patch_embeddings принимает модель и имя поля с патчами
    dataset.compute_patch_embeddings(model, patches_field=patches_field, embeddings_field=embeddings_field)
    print("Done computing patch embeddings.")


def compute_query_embedding_for_image(model, ref_image_path: str) -> np.ndarray:
    """Вычисляет эмбеддинг для reference image. Возвращает 1D numpy vector.

    Замечание: разные модели ожидают разные типы входа; для большинства моделей в Model Zoo
    .embed()/.embed_all() поддерживает PIL.Image, путь или raw array.
    """
    img = Image.open(ref_image_path).convert("RGB")
    # try model.embed / embed_all (depend on конкретной реализации модели)
    try:
        vec = model.embed(img)
    except Exception:
        # fallback: batch API
        vec = model.embed_all([img])[0]
    vec = np.asarray(vec, dtype=np.float32)
    vec = l2_normalize(vec)
    return vec


def find_similar_detections(dataset: fo.Dataset,
                            ref_image_path: str,
                            threshold: float = DEFAULT_THRESHOLD,
                            top_k: int = TOP_K,
                            model=None,
                            patches_field: str = DETECTIONS_FIELD,
                            embeddings_field: str = EMBEDDINGS_FIELD) -> List[Dict[str, Any]]:
    """Ищет все детекции в датасете похожие на ref_image_path по порогу similarity >= threshold.

    Возвращает список словарей с информацией о совпадении (sample_id, detection_index, bbox, confidence,
    class, similarity).
    """
    if model is None:
        model = load_embedding_model(MODEL_NAME)

    # 1) Убедимся что эмбеддинги для патчей посчитаны
    # Если эмбеддинги отсутствуют, compute_patch_embeddings создаст их в поле, указанном в embeddings_field.
    # Можно пропустить этот шаг если эмбеддинги уже в датасете.
    compute_patch_embeddings_for_dataset(dataset, model, patches_field, embeddings_field)

    # 2) Эмбеддинг для reference
    query_vec = compute_query_embedding_for_image(model, ref_image_path)

    # 3) Перебираем все детекции и считаем косинусную схожесть
    matches = []
    for sample in dataset.iter_samples(progress=True):
        labels = sample.get(patches_field)
        if labels is None:
            continue

        # some label containers are `Detections` objects with `.detections` list,
        # others might be plain lists depending on импорт/формат — пытаемся поддержать оба
        detections = getattr(labels, "detections", labels) or []

        for i, det in enumerate(detections):
            emb = None
            # эмбеддинг у метки может храниться в det.<embeddings_field>
            if isinstance(det, dict):
                emb = det.get(embeddings_field)
            else:
                # object may be a fiftyone label object
                emb = getattr(det, embeddings_field, None)

            if emb is None:
                # пропускаем, если для этой детекции нет эмбеддинга
                continue

            emb = np.asarray(emb, dtype=np.float32)
            emb = l2_normalize(emb)
            sim = float(np.dot(emb, query_vec))

            if sim >= threshold:
                bbox = None
                label_class = None
                confidence = None
                if isinstance(det, dict):
                    bbox = det.get("bounding_box") or det.get("bbox")
                    label_class = det.get("label")
                    confidence = det.get("confidence")
                else:
                    # fiftyone.labels.Detection
                    bbox = getattr(det, "bounding_box", None) or getattr(det, "bbox", None)
                    label_class = getattr(det, "label", None)
                    confidence = getattr(det, "confidence", None)

                matches.append({
                    "sample_id": sample.id,
                    "filepath": sample.filepath,
                    "detection_index": i,
                    "label": label_class,
                    "confidence": confidence,
                    "bbox": bbox,
                    "similarity": sim,
                })

    # 4) Отсортируем результаты по убыванию схожести и вернем топ-k
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
    return matches[:top_k]


# -------------------- Пример вызова (main) --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find similar detections in a FiftyOne dataset")
    parser.add_argument("ref_image", help="Path to reference image")
    parser.add_argument("--dataset", default=DATASET_NAME, help="FiftyOne dataset name")
    parser.add_argument("--patches_field", default=DETECTIONS_FIELD, help="Field with object detections")
    parser.add_argument("--emb_field", default=EMBEDDINGS_FIELD, help="Label attribute for embeddings")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold (0-1)")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Max results to return")
    args = parser.parse_args()

    ds = ensure_dataset(args.dataset)
    results = find_similar_detections(ds, args.ref_image, threshold=args.threshold, top_k=args.top_k,
                                      patches_field=args.patches_field, embeddings_field=args.emb_field)

    print(f"Found {len(results)} matches (threshold={args.threshold}):")
    for r in results:
        print(f"sample={r['sample_id']} file={os.path.basename(r['filepath'])} label={r['label']} sim={r['similarity']:.4f} bbox={r['bbox']}")

    # (опционально) запустить FiftyOne App чтобы просмотреть результаты
    # matched_sample_ids = list({r['sample_id'] for r in results})
    # if matched_sample_ids:
    #     view = ds.select(matched_sample_ids)
    #     fo.launch_app(view)
