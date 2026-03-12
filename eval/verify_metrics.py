import argparse
import json
from typing import Any, Dict, Tuple

import numpy as np

from metrics import summarize_verification


def sample_pairs(
    embs: np.ndarray,
    labels: np.ndarray,
    num_genuine: int = 50000,
    num_impostor: int = 50000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(labels) < 2:
        return np.array([]), np.array([])

    by_label: Dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), []).append(idx)

    labs = [label for label, idxs in by_label.items() if len(idxs) >= 1]
    genuine_labs = [label for label, idxs in by_label.items() if len(idxs) >= 2]
    if len(labs) < 2 or not genuine_labs:
        return np.array([]), np.array([])

    scores: list[float] = []
    ytrue: list[int] = []

    for _ in range(num_genuine):
        lab = int(rng.choice(genuine_labs))
        idxs = by_label[lab]
        i, j = rng.choice(idxs, size=2, replace=False)
        scores.append(float(np.dot(embs[i], embs[j])))
        ytrue.append(1)

    for _ in range(num_impostor):
        la, lb = rng.choice(labs, size=2, replace=False)
        i = int(rng.choice(by_label[int(la)]))
        j = int(rng.choice(by_label[int(lb)]))
        scores.append(float(np.dot(embs[i], embs[j])))
        ytrue.append(0)

    return np.array(scores, dtype=np.float32), np.array(ytrue, dtype=np.int32)


def _print_summary(title: str, summary: Dict[str, Any]) -> None:
    print(title)
    print(f"Pairs: {summary['pairs']} (positives={summary['positives']}, negatives={summary['negatives']})")
    print(f"ROC-AUC: {summary['roc_auc']:.4f}")
    eer = summary['eer']
    print(f"EER: {eer['value']:.4f} @ threshold {eer['threshold']:.4f}")
    for key, values in summary['far_targets'].items():
        print(f"TAR@FAR={key}: {values['tar']:.4f} @ threshold {values['threshold']:.4f} (FNMR={values['fnmr']:.4f})")
    if summary['confidence_intervals']:
        print("Confidence intervals (95% bootstrap):")
        for key, values in summary['confidence_intervals'].items():
            print(f"  {key}: [{values['low']:.4f}, {values['high']:.4f}] samples={values['samples']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--genuine", type=int, default=50000)
    ap.add_argument("--impostor", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=50)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    z = np.load(args.embeds, allow_pickle=True)
    embs = z["embs"].astype(np.float32)
    labels = z["labels"].astype(np.int32)
    if len(labels) == 0:
        print("No embeddings found.")
        return

    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    scores, ytrue = sample_pairs(embs, labels, args.genuine, args.impostor, seed=args.seed)
    if len(scores) == 0:
        print("Not enough class diversity for verification metrics.")
        return

    summary = summarize_verification(scores, ytrue, bootstrap_samples=args.bootstrap, seed=args.seed)
    _print_summary("Verification metrics", summary)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
