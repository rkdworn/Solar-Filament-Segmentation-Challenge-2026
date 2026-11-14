"""Compare out-of-fold (OOF) results of several configurations with uncertainty.

For each configuration (preds/<name>/val holding OOF maps for all 707
observations, with its tuned post_params.json, or ``<name>=<params .json>``
to use another parameter file such as post_params_learned.json):
  * PQ and Dice on all observations, and per fold (mean +- std over 5 folds)
  * 95% bootstrap confidence interval of PQ (resampling observations)
  * paired bootstrap of the PQ difference against the first configuration

A learned filter is evaluated cross-fitted: observations of fold k are
filtered by the scorer trained without fold k (``fold<k>.joblib`` next to the
saved ``all.joblib``), so its numbers are as honest as those of a fixed rule.

Example:
    .venv\\Scripts\\python.exe src\\compare.py --preds r34_oof r34_2048_oof blend_oof
    .venv\\Scripts\\python.exe src\\compare.py --preds r34_oof r34_oof=preds/r34_oof/post_params_learned.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import FULL_SIZE, N_FOLDS, PRED_DIR, ROOT, instance_masks, load_annotations
from metrics import pq_stats, to_flat
from postprocess import PostParams, to_instances, upsample_prob


def load_config(spec: str) -> tuple[str, PostParams]:
    """``name`` (uses preds/<name>/post_params.json) or ``name=<params .json>``."""
    name, _, file = spec.partition("=")
    path = Path(file) if file else PRED_DIR / name / "post_params.json"
    path = path if path.is_absolute() else ROOT / path
    return name, PostParams(**json.loads(path.read_text())["params"])


def fold_params(params: PostParams, fold: int) -> PostParams:
    """For a learned filter, the scorer trained without ``fold`` (cross-fitting); fixed rules unchanged."""
    if not params.scorer:
        return params
    scorer = Path(params.scorer)
    fold_file = (scorer if scorer.is_absolute() else ROOT / scorer).parent / f"fold{fold}.joblib"
    if not fold_file.exists():
        raise FileNotFoundError(f"cross-fitted scorer {fold_file} is missing")
    return PostParams(**{**params.as_dict(), "scorer": str(fold_file)})


def per_observation(spec: str, stems: list[str], sets: dict, gt: dict, fold_of: dict) -> np.ndarray:
    """Per observation: [IoU sum, TP, FP, FN, Dice sum, number of annotation sets]."""
    name, params = load_config(spec)
    by_fold = {k: fold_params(params, k) for k in range(N_FOLDS)}
    out = np.zeros((len(stems), 6))
    for j, stem in enumerate(tqdm(stems, desc=spec)):
        p8 = cv2.imread(str(PRED_DIR / name / "val" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        labels = to_instances(upsample_prob(p8.astype(np.float32) / 255.0, FULL_SIZE), by_fold[fold_of[stem]])
        pred_fg = labels.ravel() > 0
        for image_id in sets[stem]:
            s = pq_stats(labels, gt[image_id])
            union = np.unique(np.concatenate(gt[image_id]))
            denom = pred_fg.sum() + union.size
            d = 2.0 * pred_fg[union].sum() / denom if denom else 1.0
            out[j] += [s.iou_sum, s.tp, s.fp, s.fn, d, 1]
    return out


def pq(a: np.ndarray) -> float:
    return a[:, 0].sum() / (a[:, 1].sum() + 0.5 * a[:, 2].sum() + 0.5 * a[:, 3].sum())


def dice(a: np.ndarray) -> float:
    return a[:, 4].sum() / a[:, 5].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    index, polygons = load_annotations()
    stems = sorted(index["stem"].unique())
    sets = {s: list(g["image_id"]) for s, g in index.groupby("stem")}
    fold_of = dict(zip(index["stem"], index["fold"]))
    folds = np.array([fold_of[s] for s in stems])
    gt = {i: [to_flat(m) for m in instance_masks(polygons[i], FULL_SIZE)] for i in index["image_id"]}

    stats = {n: per_observation(n, stems, sets, gt, fold_of) for n in args.preds}
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(stems), (args.boot, len(stems)))
    base = args.preds[0]
    rows = []
    for n, a in stats.items():
        fold_pq = [pq(a[folds == k]) for k in range(N_FOLDS)]
        fold_dice = [dice(a[folds == k]) for k in range(N_FOLDS)]
        boots = np.array([pq(a[b]) for b in boot_idx])
        row = dict(config=n, PQ=pq(a), PQ_fold_mean=np.mean(fold_pq), PQ_fold_std=np.std(fold_pq, ddof=1),
                   PQ_CI95=f"[{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]",
                   Dice=dice(a), Dice_fold_std=np.std(fold_dice, ddof=1),
                   TP=int(a[:, 1].sum()), FP=int(a[:, 2].sum()), FN=int(a[:, 3].sum()),
                   folds=" ".join(f"{v:.3f}" for v in fold_pq))
        if n != base:
            d = np.array([pq(a[b]) - pq(stats[base][b]) for b in boot_idx])
            row[f"dPQ_vs_{base}"] = f"{d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
            row["P(better)"] = float((d > 0).mean())
        rows.append(row)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(PRED_DIR / "compare.csv", index=False)


if __name__ == "__main__":
    main()
