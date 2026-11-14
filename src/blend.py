"""Average the probability maps of several prediction folders (ensembling).

Maps are brought to the finest resolution among the inputs before averaging.

Example:
    .venv\\Scripts\\python.exe src\\blend.py --preds r34_f0 r34_2048_f0 --split val --out blend_a
    .venv\\Scripts\\python.exe src\\blend.py --preds a b --weights 0.3 0.7 --split val --out blend_b
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
from tqdm import tqdm

from common import PRED_DIR
from postprocess import upsample_prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="one weight per --preds folder (normalized to sum 1); default: equal weights")
    args = ap.parse_args()
    weights = None
    if args.weights is not None:
        weights = np.asarray(args.weights, np.float64)
        if weights.size != len(args.preds) or (weights <= 0).any():
            raise ValueError("--weights needs one positive value per --preds folder")
        weights = weights / weights.sum()

    dirs = [PRED_DIR / p / args.split for p in args.preds]
    stems = sorted(f.stem for f in dirs[0].glob("*.png"))
    out_dir = PRED_DIR / args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in tqdm(stems):
        maps = [cv2.imread(str(d / f"{stem}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for d in dirs]
        size = max(m.shape[0] for m in maps)
        maps = [upsample_prob(m, size) for m in maps]
        if weights is None:  # plain average (kept exactly as before, so old blends rebuild byte for byte)
            avg = np.mean(maps, axis=0)
        else:  # weighted mean of the probabilities
            avg = np.sum([w * m for w, m in zip(weights, maps)], axis=0)
        cv2.imwrite(str(out_dir / f"{stem}.png"), np.round(avg * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
