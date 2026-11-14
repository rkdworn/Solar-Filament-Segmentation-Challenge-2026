"""Predict filament probability maps with one or more trained runs (ensemble).

Probabilities are averaged over runs and over 4 flip TTA variants and saved as
8-bit PNGs (prob * 255) at the working resolution (1024).

Examples:
    # out-of-fold predictions for post-processing tuning
    .venv\\Scripts\\python.exe src\\infer.py --runs r34_f0 --split val --fold 0 --out r34_f0
    # test predictions from an ensemble of runs
    .venv\\Scripts\\python.exe src\\infer.py --runs r34_f0 r34_full --split test --out ens1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import MODEL_DIR, PRED_DIR, TEST_IMG_DIR, load_annotations
from dataset import load_image_tensor
from model import load_trained, predict_prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--weights-root", default=str(MODEL_DIR), help="folder holding the run folders")
    args = ap.parse_args()

    if args.split == "val":
        index, _ = load_annotations()
        stems = sorted(index.loc[index["fold"] == args.fold, "stem"].unique())
    else:
        stems = sorted(p.stem for p in TEST_IMG_DIR.glob("*.jpeg"))

    models = [load_trained(Path(args.weights_root) / r, args.ckpt) for r in args.runs]
    out_size = max(size for _, size in models)  # save at the finest resolution used
    out_dir = PRED_DIR / args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.synchronize()
    t0 = time.time()
    for stem in tqdm(stems, desc=f"{args.split} x{len(models)} models"):
        probs = []
        for model, size in models:
            p = predict_prob(model, load_image_tensor(stem, size).cuda(), tta=not args.no_tta)
            if size != out_size:
                p = F.interpolate(p, size=(out_size, out_size), mode="bilinear", align_corners=False)
            probs.append(p)
        prob = torch.stack(probs).mean(0)[0, 0].cpu().numpy()
        cv2.imwrite(str(out_dir / f"{stem}.png"), np.round(prob * 255).astype(np.uint8))
    sec = time.time() - t0
    print(f"{len(stems)} images in {sec:.0f} s ({sec / len(stems):.2f} s/image incl. I/O, TTA={not args.no_tta})")


if __name__ == "__main__":
    main()
