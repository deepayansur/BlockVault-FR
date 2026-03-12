import os, sys, argparse
import numpy as np
import cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from face_service.ml import FaceEngine


def read_list(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            items.append((parts[0], parts[1]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    items = read_list(args.list)
    if args.max and args.max > 0:
        items = items[:args.max]

    engine = FaceEngine(det_size=args.det_size)
    engine.init_models()

    embs, labels, paths = [], [], []

    id_to_int = {}
    next_id = 0

    for rel, pid in tqdm(items, desc="Embedding"):
        p = rel
        if args.root and not os.path.isabs(p):
            p = os.path.join(args.root, p)
        img = cv2.imread(p)
        if img is None:
            continue
        emb = engine.embed_largest_face(img)
        if emb is None:
            continue

        if pid not in id_to_int:
            id_to_int[pid] = next_id
            next_id += 1

        embs.append(emb)
        labels.append(id_to_int[pid])
        paths.append(rel)

    embs = np.stack(embs, axis=0) if embs else np.zeros((0,512), dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    paths = np.array(paths, dtype=object)
    np.savez(args.out, embs=embs, labels=labels, paths=paths, id_to_int=id_to_int)
    print(f"Saved {len(labels)} embeddings to {args.out} (classes={len(id_to_int)})")


if __name__ == "__main__":
    main()
