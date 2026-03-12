import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from eval.metrics import summarize_verification
from eval.pipeline import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_NUM_WORKERS,
    build_embedding_arrays,
    build_item_paths,
    collect_class_images,
    compute_authorization_metrics,
    embed_items,
    pick_operating_threshold,
    sample_items_balanced,
)
from eval.verify_metrics import sample_pairs


def run_vggface2_verification(
    root: str,
    det_size: int = 448,
    max_images: int = 1000,
    cache_path: str | None = None,
    genuine: int = 50000,
    impostor: int = 50000,
    seed: int = 0,
    num_workers: int = DEFAULT_NUM_WORKERS,
    bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    items = collect_class_images(root)
    selected_items = sample_items_balanced(items, max_images=max_images, seed=seed) if max_images > 0 else list(items)
    item_paths = build_item_paths(selected_items, root)

    embeddings, embed_stats = embed_items(
        item_paths=item_paths,
        det_size=det_size,
        cache_path=cache_path,
        num_workers=num_workers,
        use_gpu=use_gpu,
        cache_context=f"vggface2:{os.path.abspath(root)}:{max_images}:{seed}",
    )

    embs_arr, labels_arr, skipped, id_to_int, embedded_keys = build_embedding_arrays(selected_items, embeddings)
    if embs_arr.shape[0] == 0:
        raise RuntimeError(
            "No embeddings found for "
            f"det_size={det_size} root={os.path.abspath(root)} "
            f"(total={len(selected_items)} read_fail={embed_stats['failed_read']} "
            f"no_face={embed_stats['failed_face']} backend={embed_stats['parallel_backend']})."
        )

    scores, ytrue = sample_pairs(embs_arr, labels_arr, genuine, impostor, seed=seed)
    if len(scores) == 0:
        raise RuntimeError(
            "Not enough class diversity for verification metrics "
            f"for det_size={det_size} root={os.path.abspath(root)}."
        )

    verification = summarize_verification(scores, ytrue, bootstrap_samples=bootstrap, seed=seed)
    threshold = pick_operating_threshold(verification)
    authorization = compute_authorization_metrics(embs_arr, labels_arr, threshold=threshold, seed=seed)

    return {
        "dataset": "VGGFace2",
        "root": os.path.abspath(root),
        "config": {
            "det_size": int(det_size),
            "max_images": int(max_images),
            "genuine": int(genuine),
            "impostor": int(impostor),
            "seed": int(seed),
            "num_workers": int(num_workers),
            "bootstrap": int(bootstrap),
            "use_gpu": bool(use_gpu),
        },
        "coverage": {
            "total_images": int(len(selected_items)),
            "embedded_images": int(embs_arr.shape[0]),
            "skipped_images": int(skipped),
            "classes": int(len(id_to_int)),
            "read_fail": int(embed_stats["failed_read"]),
            "no_face": int(embed_stats["failed_face"]),
            "embedded_keys": embedded_keys,
        },
        "verification": verification,
        "authorization": authorization,
        "execution": embed_stats["runtime"],
        "parallel_backend": embed_stats["parallel_backend"],
        "cache": embed_stats["cache"],
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print("VGGFace2 Verification")
    coverage = summary["coverage"]
    print(f"Total images: {coverage['total_images']}")
    print(f"Embedded images: {coverage['embedded_images']}")
    print(f"Skipped images: {coverage['skipped_images']}")
    print(f"Classes: {coverage['classes']}")
    print(f"Embedding failures: read={coverage['read_fail']}, no_face={coverage['no_face']}")
    print(
        f"Execution device: {summary['execution']['execution_device']} providers={summary['execution']['providers']} "
        f"workers={summary['config']['num_workers']} backend={summary['parallel_backend']}"
    )
    verification = summary["verification"]
    print(f"ROC-AUC: {verification['roc_auc']:.4f}")
    print(f"EER: {verification['eer']['value']:.4f} @ threshold {verification['eer']['threshold']:.4f}")
    for key, values in verification["far_targets"].items():
        print(f"TAR@FAR={key}: {values['tar']:.4f} @ threshold {values['threshold']:.4f}")
    auth = summary["authorization"]
    print("Authorization metrics:")
    print(f"  Rank-1: {auth['rank1'] if auth['rank1'] is not None else 'n/a'}")
    print(f"  Enrolled accept rate: {auth['enrolled_accept_rate'] if auth['enrolled_accept_rate'] is not None else 'n/a'}")
    print(f"  Unknown reject rate: {auth['unknown_reject_rate'] if auth['unknown_reject_rate'] is not None else 'n/a'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="VGGFace2 split dir, e.g. .../test")
    ap.add_argument("--det-size", type=int, default=448, help="Detector input size.")
    ap.add_argument("--max-images", type=int, default=1000, help="Limit number of images for quick runs")
    ap.add_argument("--cache", default="", help="Optional npz cache path")
    ap.add_argument("--genuine", type=int, default=50000)
    ap.add_argument("--impostor", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    result = run_vggface2_verification(
        root=args.root,
        det_size=args.det_size,
        max_images=args.max_images,
        cache_path=args.cache or None,
        genuine=args.genuine,
        impostor=args.impostor,
        seed=args.seed,
        num_workers=args.num_workers,
        bootstrap=args.bootstrap,
        use_gpu=args.use_gpu,
    )
    _print_summary(result)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
