"""Grid-search post-processing parameters on out-of-fold predictions.

Everything is evaluated at the submission resolution (2048) against ground
truth rasterized with pycocotools, i.e. the same way the host scores.

Speed trick: for a fixed (threshold, merge_px) the instance map is computed
once. Removing an instance (min_area / min_conf filters) never changes the IoU
of the remaining ones, and with IoU > 0.5 each ground-truth filament has at
most one candidate match, so all filter settings are scored from one table.

Example:
    .venv\\Scripts\\python.exe src\\tune.py --pred r34_f0 --fold 0
"""
from __future__ import annotations

import argparse
import itertools
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import FULL_SIZE, PRED_DIR, instance_masks, load_annotations
from metrics import MATCH_IOU, PQStats, summarize, to_flat
from postprocess import PostParams, instance_confidence, to_instances, upsample_prob

THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
MERGE_PX = [0, 8, 16, 32, 48]          # at 2048 resolution
MIN_AREA = [0, 100, 200, 400, 800]
MIN_CONF = [0.0, 0.8, 0.9, 0.95, 0.99]   # peak probability
MIN_MEANP = [0.0, 0.6, 0.7, 0.8, 0.85, 0.9]  # mean probability


def instance_props(labels: np.ndarray, prob: np.ndarray):
    """Per predicted instance (index 0 = background): area, peak and mean prob."""
    area = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    maxp, meanp = instance_confidence(labels, prob, area)
    return area, maxp, meanp


def gt_matches(labels: np.ndarray, area: np.ndarray, gts: list[np.ndarray]):
    """Per ground-truth filament: (id of the only possible match, its IoU)."""
    flat = labels.ravel()
    n = area.size
    matches = []
    for idx in gts:
        inter = np.bincount(flat[idx], minlength=n)
        inter[0] = 0
        k = int(inter.argmax())
        iou = inter[k] / (idx.size + area[k] - inter[k]) if inter[k] else 0.0
        matches.append((k, iou))
    return matches


def union_overlap(labels: np.ndarray, area: np.ndarray, gts: list[np.ndarray]):
    """Overlap of every predicted instance with the ground-truth union mask,
    plus the union's size. Enough to compute Dice for any subset of instances."""
    union_idx = np.unique(np.concatenate(gts)) if gts else np.zeros(0, np.int64)
    inter = np.bincount(labels.ravel()[union_idx], minlength=area.size)
    inter[0] = 0
    return inter, int(union_idx.size)


def score_filters(tables, min_area: int, min_conf: float, min_meanp: float) -> tuple[list[PQStats], list[float]]:
    """PQ statistics and Dice for every annotation set under one filter setting."""
    stats, dices = [], []
    for area, maxp, meanp, per_set in tables:
        keep = (area >= max(min_area, 1)) & (maxp >= min_conf) & (meanp >= min_meanp)
        keep[0] = False
        n_kept = int(keep.sum())
        pred_px = int(area[keep].sum())
        for matches, (inter, gt_px) in per_set:  # one entry per annotation set
            s = PQStats()
            matched = set()
            for k, iou in matches:
                if iou > MATCH_IOU and keep[k]:
                    s.tp += 1
                    s.iou_sum += iou
                    matched.add(k)
                else:
                    s.fn += 1
            s.fp = n_kept - len(matched)
            stats.append(s)
            denom = pred_px + gt_px
            dices.append(2.0 * inter[keep].sum() / denom if denom else 1.0)
    return stats, dices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="name under preds/ (uses preds/<name>/val)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--objective", default="combo", choices=["combo", "dice", "pq_pooled", "pq_mean"],
                    help="combo = (Dice + PQ) / 2; the leaderboard shows one of them, the jury uses both")
    args = ap.parse_args()

    index, polygons = load_annotations()
    val = index[index["fold"] == args.fold]
    sets = {s: list(g["image_id"]) for s, g in val.groupby("stem")}
    print(f"loading ground truth for {len(val)} annotation sets at {FULL_SIZE}px ...")
    gt = {i: [to_flat(m) for m in instance_masks(polygons[i], FULL_SIZE)] for i in val["image_id"]}

    pred_dir = PRED_DIR / args.pred / "val"
    probs = {}
    for s in sets:
        p8 = cv2.imread(str(pred_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        probs[s] = upsample_prob(p8.astype(np.float32) / 255.0, FULL_SIZE)

    rows = []
    for thr, merge in tqdm(list(itertools.product(THRESHOLDS, MERGE_PX)), desc="threshold x merge"):
        tables = []
        for s, ids in sets.items():
            labels = to_instances(probs[s], PostParams(threshold=thr, merge_px=merge))
            area, maxp, meanp = instance_props(labels, probs[s])
            per_set = [(gt_matches(labels, area, gt[i]), union_overlap(labels, area, gt[i])) for i in ids]
            tables.append((area, maxp, meanp, per_set))
        for a, c, m in itertools.product(MIN_AREA, MIN_CONF, MIN_MEANP):
            stats, dices = score_filters(tables, a, c, m)
            res = summarize(stats)
            res["dice"] = float(np.mean(dices))
            res["combo"] = (res["dice"] + res["pq_pooled"]) / 2
            rows.append(dict(threshold=thr, merge_px=merge, min_area=a, min_conf=c, min_meanp=m, **res))

    df = pd.DataFrame(rows).sort_values(args.objective, ascending=False)
    pd.set_option("display.width", 220)
    print(f"--- top settings by {args.objective}")
    print(df.head(12).to_string(index=False))
    for other in {"dice", "pq_pooled"} - {args.objective}:
        print(f"--- best by {other}")
        print(df.sort_values(other, ascending=False).head(3).to_string(index=False))
    df.to_csv(PRED_DIR / args.pred / "tune.csv", index=False)

    best = df.iloc[0]
    params = PostParams(float(best.threshold), int(best.merge_px), int(best.min_area), float(best.min_conf),
                        float(best.min_meanp))
    summary = dict(params=params.as_dict(), objective=args.objective,
                   **{k: float(best[k]) for k in ["dice", "pq_pooled", "pq_mean", "sq", "rq"]})
    (PRED_DIR / args.pred / "post_params.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
