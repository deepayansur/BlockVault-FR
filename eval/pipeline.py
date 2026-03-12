from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from face_service.ml import DEFAULT_EMBEDDINGS_PATH, DEFAULT_FACE_DB_DIR, FaceEngine

DEFAULT_NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_BOOTSTRAP_SAMPLES = 50

_WORKER_ENGINE: Optional[FaceEngine] = None
_THREAD_LOCAL = threading.local()
_THREAD_ENGINE_CONFIG: Dict[str, Any] = {}


def collect_class_images(root: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        for fn in sorted(os.listdir(pdir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                rel = os.path.join(pid, fn)
                items.append((rel, pid))
    return items


def sample_items_balanced(items: List[Tuple[str, str]], max_images: int, seed: int) -> List[Tuple[str, str]]:
    if max_images <= 0 or len(items) <= max_images:
        return list(items)

    rng = np.random.default_rng(seed)
    by_pid: Dict[str, List[Tuple[str, str]]] = {}
    for rel, pid in items:
        by_pid.setdefault(pid, []).append((rel, pid))

    pids = list(by_pid.keys())
    rng.shuffle(pids)
    for pid in pids:
        rng.shuffle(by_pid[pid])

    target_class_count = min(len(pids), max(1, max_images // 2))
    selected_pids = pids[:target_class_count]

    sampled: List[Tuple[str, str]] = []

    for pid in selected_pids:
        bucket = by_pid[pid]
        take = min(2, len(bucket), max_images - len(sampled))
        for _ in range(take):
            sampled.append(bucket.pop())
        if len(sampled) >= max_images:
            return sampled

    while len(sampled) < max_images:
        progressed = False
        for pid in selected_pids:
            bucket = by_pid[pid]
            if not bucket:
                continue
            sampled.append(bucket.pop())
            progressed = True
            if len(sampled) >= max_images:
                break
        if not progressed:
            break
        rng.shuffle(selected_pids)
    return sampled


def build_item_paths(items: Iterable[Tuple[str, str]], root: str) -> List[Tuple[str, str]]:
    return [(key, os.path.join(root, key)) for key, _ in items]


def prepare_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < 112:
        scale = 112.0 / float(min_dim)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return img


def _cache_metadata(
    item_paths: Sequence[Tuple[str, str]],
    det_size: int,
    use_gpu: bool,
    cache_context: str,
    runtime_info: Dict[str, Any],
) -> Dict[str, Any]:
    digest = hashlib.sha1()
    for key, path in item_paths:
        digest.update(key.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(os.path.abspath(path).encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return {
        "cache_context": cache_context,
        "det_size": int(det_size),
        "use_gpu": bool(use_gpu),
        "execution_device": runtime_info.get("execution_device", "unknown"),
        "providers": list(runtime_info.get("providers", [])),
        "item_count": int(len(item_paths)),
        "slice_fingerprint": digest.hexdigest(),
    }


def load_embedding_cache(cache_path: Optional[str], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        cache = np.load(cache_path, allow_pickle=True)
        cached_meta = json.loads(str(cache["metadata_json"].item()))
        if cached_meta != metadata:
            return {}
        keys = cache["keys"]
        embs = cache["embs"]
        return {str(key): emb.astype(np.float32) for key, emb in zip(keys, embs)}
    except Exception:
        return {}


def save_embedding_cache(cache_path: Optional[str], metadata: Dict[str, Any], embeddings: Dict[str, np.ndarray]) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    keys = np.array(list(embeddings.keys()), dtype=object)
    embs = np.stack(list(embeddings.values()), axis=0) if embeddings else np.zeros((0, 512), dtype=np.float32)
    np.savez(
        cache_path,
        metadata_json=np.array(json.dumps(metadata), dtype=object),
        keys=keys,
        embs=embs,
    )


def _probe_runtime(det_size: int, use_gpu: bool) -> Dict[str, Any]:
    engine = FaceEngine(det_size=det_size, use_gpu=use_gpu)
    runtime = engine.runtime_info()
    runtime["det_size"] = int(det_size)
    return runtime


def _init_embed_worker(det_size: int, use_gpu: bool, face_db_dir: str, embeddings_path: str) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = FaceEngine(
        det_size=det_size,
        use_gpu=use_gpu,
        face_db_dir=face_db_dir,
        embeddings_path=embeddings_path,
    )
    _WORKER_ENGINE.init_models()


def _embed_chunk_with_engine(engine: FaceEngine, chunk: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    embeddings: List[Tuple[str, np.ndarray]] = []
    failed_read = 0
    failed_face = 0
    for key, path in chunk:
        img = prepare_image(path)
        if img is None:
            failed_read += 1
            continue
        emb = engine.embed_largest_face(img)
        if emb is None:
            failed_face += 1
            continue
        emb = emb.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        embeddings.append((key, emb))
    return {
        "embeddings": embeddings,
        "failed_read": failed_read,
        "failed_face": failed_face,
    }


def _embed_chunk(chunk: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    if _WORKER_ENGINE is None:
        raise RuntimeError("embed worker is not initialized")
    return _embed_chunk_with_engine(_WORKER_ENGINE, chunk)


def _configure_thread_workers(det_size: int, use_gpu: bool, face_db_dir: str, embeddings_path: str) -> None:
    global _THREAD_ENGINE_CONFIG
    _THREAD_ENGINE_CONFIG = {
        "det_size": det_size,
        "use_gpu": use_gpu,
        "face_db_dir": face_db_dir,
        "embeddings_path": embeddings_path,
    }


def _thread_engine() -> FaceEngine:
    engine = getattr(_THREAD_LOCAL, "engine", None)
    config = getattr(_THREAD_LOCAL, "engine_config", None)
    if engine is None or config != _THREAD_ENGINE_CONFIG:
        engine = FaceEngine(
            det_size=int(_THREAD_ENGINE_CONFIG["det_size"]),
            use_gpu=bool(_THREAD_ENGINE_CONFIG["use_gpu"]),
            face_db_dir=str(_THREAD_ENGINE_CONFIG["face_db_dir"]),
            embeddings_path=str(_THREAD_ENGINE_CONFIG["embeddings_path"]),
        )
        engine.init_models()
        _THREAD_LOCAL.engine = engine
        _THREAD_LOCAL.engine_config = dict(_THREAD_ENGINE_CONFIG)
    return engine


def _embed_chunk_thread(chunk: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    return _embed_chunk_with_engine(_thread_engine(), chunk)


def _chunk_items(item_paths: Sequence[Tuple[str, str]], chunk_size: int) -> List[List[Tuple[str, str]]]:
    return [list(item_paths[i : i + chunk_size]) for i in range(0, len(item_paths), chunk_size)]


def embed_items(
    item_paths: Sequence[Tuple[str, str]],
    det_size: int,
    cache_path: Optional[str] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    use_gpu: bool = False,
    cache_context: str = "",
    face_db_dir: str = DEFAULT_FACE_DB_DIR,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    runtime_info = _probe_runtime(det_size=det_size, use_gpu=use_gpu)
    metadata = _cache_metadata(item_paths, det_size, use_gpu, cache_context, runtime_info)
    embeddings = load_embedding_cache(cache_path, metadata)
    needed = [(key, path) for key, path in item_paths if key not in embeddings]

    failed_read = 0
    failed_face = 0
    parallel_backend = "serial"

    if needed:
        worker_count = max(1, int(num_workers))
        if worker_count == 1 or len(needed) == 1:
            _init_embed_worker(det_size, use_gpu, face_db_dir, embeddings_path)
            for key, path in tqdm(needed, desc="Embedding", total=len(needed)):
                result = _embed_chunk([(key, path)])
                failed_read += int(result["failed_read"])
                failed_face += int(result["failed_face"])
                for emb_key, emb in result["embeddings"]:
                    embeddings[emb_key] = emb
        else:
            chunk_size = max(1, math.ceil(len(needed) / float(worker_count * 4)))
            chunks = _chunk_items(needed, chunk_size)
            try:
                ctx = get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=ctx,
                    initializer=_init_embed_worker,
                    initargs=(det_size, use_gpu, face_db_dir, embeddings_path),
                ) as executor:
                    futures = [executor.submit(_embed_chunk, chunk) for chunk in chunks]
                    parallel_backend = "process"
                    for future in tqdm(as_completed(futures), desc=f"Embedding ({worker_count} workers)", total=len(futures)):
                        result = future.result()
                        failed_read += int(result["failed_read"])
                        failed_face += int(result["failed_face"])
                        for emb_key, emb in result["embeddings"]:
                            embeddings[emb_key] = emb.astype(np.float32)
            except (PermissionError, OSError):
                _configure_thread_workers(det_size, use_gpu, face_db_dir, embeddings_path)
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(_embed_chunk_thread, chunk) for chunk in chunks]
                    parallel_backend = "thread_fallback"
                    for future in tqdm(as_completed(futures), desc=f"Embedding ({worker_count} thread fallback)", total=len(futures)):
                        result = future.result()
                        failed_read += int(result["failed_read"])
                        failed_face += int(result["failed_face"])
                        for emb_key, emb in result["embeddings"]:
                            embeddings[emb_key] = emb.astype(np.float32)

    save_embedding_cache(cache_path, metadata, embeddings)
    stats = {
        "failed_read": int(failed_read),
        "failed_face": int(failed_face),
        "runtime": runtime_info,
        "cache": {
            "path": cache_path,
            "metadata": metadata,
            "hit_count": int(len(item_paths) - len(needed)),
            "miss_count": int(len(needed)),
        },
        "num_workers": int(max(1, num_workers)),
        "parallel_backend": parallel_backend,
    }
    return embeddings, stats


def build_embedding_arrays(
    items: Sequence[Tuple[str, str]],
    embeddings: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, int], List[str]]:
    id_to_int: Dict[str, int] = {}
    labels: List[int] = []
    emb_list: List[np.ndarray] = []
    embedded_keys: List[str] = []
    skipped = 0

    for key, pid in items:
        emb = embeddings.get(key)
        if emb is None:
            skipped += 1
            continue
        if pid not in id_to_int:
            id_to_int[pid] = len(id_to_int)
        emb_list.append(emb.astype(np.float32))
        labels.append(id_to_int[pid])
        embedded_keys.append(key)

    embs_arr = np.stack(emb_list, axis=0).astype(np.float32) if emb_list else np.zeros((0, 512), dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)
    return embs_arr, labels_arr, skipped, id_to_int, embedded_keys


def pick_operating_threshold(verification_summary: Dict[str, Any]) -> float:
    far_targets = verification_summary.get("far_targets") or {}
    for key in ("1e-04", "1e-03", "1e-02"):
        target = far_targets.get(key)
        if target and np.isfinite(target.get("threshold", np.inf)):
            return float(target["threshold"])
    return float((verification_summary.get("eer") or {}).get("threshold", 0.0))


def compute_authorization_metrics(
    embs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    seed: int = 0,
    enrolled_fraction: float = 0.7,
) -> Dict[str, Any]:
    if embs.shape[0] == 0 or labels.size == 0:
        return {
            "gallery_size": 0,
            "enrolled_probe_count": 0,
            "unknown_probe_count": 0,
            "enrolled_identity_count": 0,
            "unknown_identity_count": 0,
            "rank1": None,
            "enrolled_accept_rate": None,
            "unknown_reject_rate": None,
            "threshold": float(threshold),
        }

    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), []).append(idx)

    candidate_labels = [label for label, idxs in by_label.items() if len(idxs) >= 2]
    if len(candidate_labels) < 2:
        return {
            "gallery_size": 0,
            "enrolled_probe_count": 0,
            "unknown_probe_count": 0,
            "enrolled_identity_count": 0,
            "unknown_identity_count": 0,
            "rank1": None,
            "enrolled_accept_rate": None,
            "unknown_reject_rate": None,
            "threshold": float(threshold),
        }

    shuffled = list(candidate_labels)
    rng.shuffle(shuffled)
    enrolled_count = max(1, int(round(len(shuffled) * enrolled_fraction)))
    if enrolled_count >= len(shuffled):
        enrolled_count = len(shuffled) - 1
    enrolled_labels = set(shuffled[:enrolled_count])
    unknown_labels = set(shuffled[enrolled_count:])

    gallery_idx: List[int] = []
    enrolled_probe_idx: List[int] = []
    unknown_probe_idx: List[int] = []

    for label, idxs in by_label.items():
        if label not in enrolled_labels and label not in unknown_labels:
            continue
        ordered = list(idxs)
        rng.shuffle(ordered)
        if label in enrolled_labels:
            gallery_idx.append(ordered[0])
            enrolled_probe_idx.extend(ordered[1:])
        else:
            unknown_probe_idx.extend(ordered)

    if not gallery_idx or not enrolled_probe_idx:
        return {
            "gallery_size": int(len(gallery_idx)),
            "enrolled_probe_count": int(len(enrolled_probe_idx)),
            "unknown_probe_count": int(len(unknown_probe_idx)),
            "enrolled_identity_count": int(len(enrolled_labels)),
            "unknown_identity_count": int(len(unknown_labels)),
            "rank1": None,
            "enrolled_accept_rate": None,
            "unknown_reject_rate": None,
            "threshold": float(threshold),
        }

    gallery_embs = embs[np.array(gallery_idx, dtype=np.int32)]
    gallery_labels = labels[np.array(gallery_idx, dtype=np.int32)]

    enrolled_scores = embs[np.array(enrolled_probe_idx, dtype=np.int32)] @ gallery_embs.T
    enrolled_best_idx = np.argmax(enrolled_scores, axis=1)
    enrolled_best_scores = enrolled_scores[np.arange(enrolled_scores.shape[0]), enrolled_best_idx]
    enrolled_best_labels = gallery_labels[enrolled_best_idx]
    enrolled_truth = labels[np.array(enrolled_probe_idx, dtype=np.int32)]

    rank1 = float(np.mean(enrolled_best_labels == enrolled_truth))
    enrolled_accept_rate = float(np.mean((enrolled_best_labels == enrolled_truth) & (enrolled_best_scores >= threshold)))

    unknown_reject_rate: Optional[float]
    if unknown_probe_idx:
        unknown_scores = embs[np.array(unknown_probe_idx, dtype=np.int32)] @ gallery_embs.T
        unknown_best_scores = np.max(unknown_scores, axis=1)
        unknown_reject_rate = float(np.mean(unknown_best_scores < threshold))
    else:
        unknown_reject_rate = None

    return {
        "gallery_size": int(len(gallery_idx)),
        "enrolled_probe_count": int(len(enrolled_probe_idx)),
        "unknown_probe_count": int(len(unknown_probe_idx)),
        "enrolled_identity_count": int(len(enrolled_labels)),
        "unknown_identity_count": int(len(unknown_labels)),
        "rank1": rank1,
        "enrolled_accept_rate": enrolled_accept_rate,
        "unknown_reject_rate": unknown_reject_rate,
        "threshold": float(threshold),
    }

