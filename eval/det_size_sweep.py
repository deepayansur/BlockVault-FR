import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from face_service.ml import FaceEngine


def collect_images(root: str) -> List[str]:
    items: List[str] = []
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append(os.path.join(pdir, fn))
    return items


def prepare_image(path: str) -> np.ndarray | None:
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


def run_sweep(images: List[str], det_sizes: List[int]) -> None:
    for det in det_sizes:
        engine = FaceEngine(det_size=det)
        engine.init_models()

        read_fail = 0
        no_face = 0
        ok = 0

        for p in tqdm(images, desc=f"det_size={det}", total=len(images)):
            img = prepare_image(p)
            if img is None:
                read_fail += 1
                continue
            emb = engine.embed_largest_face(img)
            if emb is None:
                no_face += 1
            else:
                ok += 1

        total = len(images)
        print(f"det_size={det} total={total} ok={ok} no_face={no_face} read_fail={read_fail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (VGGFace2 split dir)")
    ap.add_argument("--max-images", type=int, default=100)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--min-det", type=int, default=320)
    ap.add_argument("--max-det", type=int, default=1280)
    ap.add_argument("--step", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = collect_images(args.root)
    if len(images) == 0:
        print("No images found.")
        return

    rng = np.random.default_rng(args.seed)
    if args.max_images > 0 and len(images) > args.max_images:
        idx = rng.choice(len(images), size=args.max_images, replace=False)
        images = [images[i] for i in idx]

    candidates = list(range(args.min_det, args.max_det + 1, args.step))
    if len(candidates) == 0:
        print("No det sizes to try.")
        return
    if args.trials > 0 and len(candidates) > args.trials:
        det_sizes = list(rng.choice(candidates, size=args.trials, replace=False))
    else:
        det_sizes = candidates

    print(f"Images: {len(images)}  Trials: {len(det_sizes)}  Sizes: {det_sizes}")
    run_sweep(images, det_sizes)


if __name__ == "__main__":
    main()
