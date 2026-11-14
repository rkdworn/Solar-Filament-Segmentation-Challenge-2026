"""Predict filament probability maps with one or more trained runs (ensemble).

Probabilities are averaged over runs and over 4 flip TTA variants and saved as
8-bit PNGs (prob * 255) at the finest network resolution among the runs.

Examples:
    # out-of-fold predictions for post-processing tuning
    .venv\\Scripts\\python.exe src\\infer.py --runs r34_f0 --split val --fold 0 --out r34_f0
    # test predictions from an ensemble of runs stored in the repository
    .venv\\Scripts\\python.exe src\\infer.py --runs r34_f0 r34_f1 --split test --out final --weights-root weights
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


def split_stems(split: str, fold: int = 0) -> list[str]:
    """Observation names of a validation fold or of the test set."""
    if split == "val":
        index, _ = load_annotations()
        return sorted(index.loc[index["fold"] == fold, "stem"].unique())
    return sorted(p.stem for p in TEST_IMG_DIR.glob("*.jpeg"))


def predict_maps(runs: list[str], stems: list[str], out_dir: Path, weights_root: Path = MODEL_DIR,
                 ckpt: str = "best.pt", tta: bool | str = True) -> float:
    """Write one averaged probability PNG per observation; returns seconds spent."""
    models = [load_trained(Path(weights_root) / r, ckpt) for r in runs]
    out_size = max(size for _, size in models)  # save at the finest resolution used
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.synchronize()
    t0 = time.time()
    for stem in tqdm(stems, desc=f"x{len(models)} models"):
        probs = []
        for model, size in models:
            p = predict_prob(model, load_image_tensor(stem, size).cuda(), tta=tta)
            if size != out_size:
                p = F.interpolate(p, size=(out_size, out_size), mode="bilinear", align_corners=False)
            probs.append(p)
        prob = torch.stack(probs).mean(0)[0, 0].cpu().numpy()
        cv2.imwrite(str(out_dir / f"{stem}.png"), np.round(prob * 255).astype(np.uint8))
    sec = time.time() - t0
    print(f"{len(stems)} images in {sec:.0f} s ({sec / len(stems):.2f} s/image incl. I/O, TTA={tta})")
    return sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--tta", default="flip4", choices=["none", "flip4", "d4"])
    ap.add_argument("--weights-root", default=str(MODEL_DIR), help="folder holding the run folders")
    args = ap.parse_args()
    predict_maps(args.runs, split_stems(args.split, args.fold), PRED_DIR / args.out / args.split,
                 Path(args.weights_root), args.ckpt, tta=args.tta)


if __name__ == "__main__":
    main()
