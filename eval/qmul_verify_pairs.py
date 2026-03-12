import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from eval.metrics import summarize_verification
from eval.pipeline import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_NUM_WORKERS,
    build_embedding_arrays,
    compute_authorization_metrics,
    embed_items,
    pick_operating_threshold,
)


def _as_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        return _as_str(x.flat[0])
    return str(x)


def _normalize_name(s: str) -> str:
    return s.strip().strip('"').strip("'")


def _find_pairs_var(mat: Dict[str, Any]) -> Optional[str]:
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 2 and value.dtype == object:
            return key
    for key in ("positive_pairs_names", "negative_pairs_names"):
        if key in mat:
            return key
    return None


def load_pairs(mat_path: str) -> List[Tuple[str, str]]:
    mat = loadmat(mat_path)
    key = _find_pairs_var(mat)
    if key is None:
        raise ValueError(f"Could not find pair list in {mat_path}")
    arr = mat[key]
    pairs: List[Tuple[str, str]] = []
    for idx in range(arr.shape[0]):
        a = _normalize_name(_as_str(arr[idx, 0]))
        b = _normalize_name(_as_str(arr[idx, 1]))
        if a and b:
            pairs.append((a, b))
    return pairs


def resolve_path(root: str, name: str) -> str:
    if os.path.isabs(name):
        return name
    if ("/" not in name) and ("\\" not in name):
        return os.path.join(root, "verification_images", name)
    return os.path.join(root, name)


def infer_person_id(name: str) -> str:
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    return stem.split("_")[0] if "_" in stem else stem


def score_pairs(
    pairs: List[Tuple[str, str]],
    embeddings: Dict[str, np.ndarray],
    label: int,
) -> Tuple[List[float], List[int], int]:
    scores: List[float] = []
    labels: List[int] = []
    skipped = 0
    for a, b in pairs:
        emb_a = embeddings.get(a)
        emb_b = embeddings.get(b)
        if emb_a is None or emb_b is None:
            skipped += 1
            continue
        scores.append(float(np.dot(emb_a, emb_b)))
        labels.append(label)
    return scores, labels, skipped


def _empty_verification_note() -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    verification = {
        "pairs": 0,
        "positives": 0,
        "negatives": 0,
        "eer": {"value": 1.0, "threshold": float("inf")},
        "roc_auc": 0.0,
        "far_targets": {
            "1e-02": {"far": 1e-2, "tar": 0.0, "threshold": float("inf"), "fnmr": 1.0},
            "1e-03": {"far": 1e-3, "tar": 0.0, "threshold": float("inf"), "fnmr": 1.0},
            "1e-04": {"far": 1e-4, "tar": 0.0, "threshold": float("inf"), "fnmr": 1.0},
        },
        "confidence_intervals": {},
    }
    authorization = {
        "gallery_size": 0,
        "enrolled_probe_count": 0,
        "unknown_probe_count": 0,
        "enrolled_identity_count": 0,
        "unknown_identity_count": 0,
        "rank1": None,
        "enrolled_accept_rate": None,
        "unknown_reject_rate": None,
        "threshold": float("inf"),
    }
    return verification, authorization, "No valid QMUL pairs were scorable with the current embedding coverage."


def run_qmul_verification(
    root: str,
    pairs_pos: str = "",
    pairs_neg: str = "",
    det_size: int = 640,
    max_pairs: int = 0,
    cache_path: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    use_gpu: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    pos_path = pairs_pos or os.path.join(root, "positive_pairs_names.mat")
    neg_path = pairs_neg or os.path.join(root, "negative_pairs_names.mat")

    pos_pairs = load_pairs(pos_path)
    neg_pairs = load_pairs(neg_path)
    if max_pairs > 0:
        pos_pairs = pos_pairs[:max_pairs]
        neg_pairs = neg_pairs[:max_pairs]

    all_names = sorted(set(name for pair in (pos_pairs + neg_pairs) for name in pair))
    item_paths = [(name, resolve_path(root, name)) for name in all_names]

    embeddings, embed_stats = embed_items(
        item_paths=item_paths,
        det_size=det_size,
        cache_path=cache_path,
        num_workers=num_workers,
        use_gpu=use_gpu,
        cache_context=f"qmul_verification:{os.path.abspath(root)}:{max_pairs}:{det_size}",
    )

    pos_scores, pos_labels, pos_skipped = score_pairs(pos_pairs, embeddings, 1)
    neg_scores, neg_labels, neg_skipped = score_pairs(neg_pairs, embeddings, 0)
    scores = np.array(pos_scores + neg_scores, dtype=np.float32)
    labels = np.array(pos_labels + neg_labels, dtype=np.int32)

    labeled_items = [(name, infer_person_id(name)) for name in all_names]
    embs_arr, labels_arr, _, _, _ = build_embedding_arrays(labeled_items, embeddings)

    if scores.size == 0:
        verification, authorization, note = _empty_verification_note()
    else:
        verification = summarize_verification(scores, labels, bootstrap_samples=bootstrap, seed=seed)
        threshold = pick_operating_threshold(verification)
        authorization = compute_authorization_metrics(embs_arr, labels_arr, threshold=threshold, seed=seed)
        note = None

    return {
        "dataset": "QMUL-SurvFace",
        "root": os.path.abspath(root),
        "config": {
            "det_size": int(det_size),
            "max_pairs": int(max_pairs),
            "num_workers": int(num_workers),
            "bootstrap": int(bootstrap),
            "use_gpu": bool(use_gpu),
            "seed": int(seed),
        },
        "coverage": {
            "total_images": int(len(all_names)),
            "embedded_images": int(len(embeddings)),
            "skipped_images": int(len(all_names) - len(embeddings)),
            "read_fail": int(embed_stats["failed_read"]),
            "no_face": int(embed_stats["failed_face"]),
            "total_pairs": int(len(pos_pairs) + len(neg_pairs)),
            "scored_pairs": int(scores.size),
            "skipped_pairs": int(pos_skipped + neg_skipped),
        },
        "verification": verification,
        "authorization": authorization,
        "execution": embed_stats["runtime"],
        "parallel_backend": embed_stats["parallel_backend"],
        "cache": embed_stats["cache"],
        "note": note,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    coverage = summary["coverage"]
    print("QMUL-SurvFace Verification")
    print(f"Total pairs: {coverage['total_pairs']}")
    print(f"Scored pairs: {coverage['scored_pairs']}")
    print(f"Skipped pairs: {coverage['skipped_pairs']}")
    print(f"Embedding failures: read={coverage['read_fail']}, no_face={coverage['no_face']}")
    print(
        f"Execution device: {summary['execution']['execution_device']} providers={summary['execution']['providers']} "
        f"workers={summary['config']['num_workers']} backend={summary['parallel_backend']}"
    )
    verification = summary["verification"]
    print(f"ROC-AUC: {verification['roc_auc']:.4f}")
    print(f"EER: {verification['eer']['value']:.4f} @ threshold {verification['eer']['threshold']}")
    for key, values in verification["far_targets"].items():
        print(f"TAR@FAR={key}: {values['tar']:.4f} @ threshold {values['threshold']}")
    auth = summary["authorization"]
    print("Authorization metrics:")
    print(f"  Rank-1: {auth['rank1'] if auth['rank1'] is not None else 'n/a'}")
    print(f"  Enrolled accept rate: {auth['enrolled_accept_rate'] if auth['enrolled_accept_rate'] is not None else 'n/a'}")
    print(f"  Unknown reject rate: {auth['unknown_reject_rate'] if auth['unknown_reject_rate'] is not None else 'n/a'}")
    if summary.get("note"):
        print(summary["note"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="QMUL Face_Verification_Test_Set root")
    ap.add_argument("--pairs-pos", default="", help="positive_pairs_names.mat override")
    ap.add_argument("--pairs-neg", default="", help="negative_pairs_names.mat override")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--cache", default="", help="Optional npz cache path")
    ap.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    result = run_qmul_verification(
        root=args.root,
        pairs_pos=args.pairs_pos,
        pairs_neg=args.pairs_neg,
        det_size=args.det_size,
        max_pairs=args.max_pairs,
        cache_path=args.cache or None,
        num_workers=args.num_workers,
        bootstrap=args.bootstrap,
        use_gpu=args.use_gpu,
        seed=args.seed,
    )
    _print_summary(result)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
