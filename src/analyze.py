"""Compare tuned runs on the validation fold beyond a single PQ number.

For every run (predictions under preds/<name>/val with its tuned
post_params.json) this reports, at 2048 px:
  * PQ / SQ / RQ / Dice and TP / FP / FN
  * recall (IoU > 0.5) by ground-truth filament size
  * fragmentation: GT filaments touched by >= 2 predictions (1:n)
  * merging: predictions touching >= 2 GT filaments (n:1)
  * PQ per GONG observatory (last two letters of the file name)

Example:
    .venv\\Scripts\\python.exe src\\analyze.py --preds r34_f0 r34_2048_f0 --fold 0
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from common import FULL_SIZE, PRED_DIR, instance_masks, load_annotations
from metrics import MATCH_IOU, PQStats, summarize, to_flat
from postprocess import PostParams, to_instances, upsample_prob

SIZE_BINS = [(0, 500, "small <500px"), (500, 2000, "medium 500-2000px"), (2000, 10**9, "large >2000px")]


def analyze_run(name: str, sets: dict, gt: dict, params_from: str | None = None) -> tuple[dict, dict]:
    """params_from: use another run's tuned post-processing (checks that it transfers)."""
    params = PostParams(**json.loads((PRED_DIR / (params_from or name) / "post_params.json").read_text())["params"])
    stats, dices = [], []
    per_site = defaultdict(list)
    size_hit = defaultdict(lambda: [0, 0])
    frag = merge = n_gt_total = n_pred_total = 0
    for stem, ids in sets.items():
        p8 = cv2.imread(str(PRED_DIR / name / "val" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        labels = to_instances(upsample_prob(p8.astype(np.float32) / 255.0, FULL_SIZE), params)
        flat = labels.ravel()
        n = int(flat.max()) + 1
        area = np.bincount(flat, minlength=n)
        pred_px = int(area[1:].sum())
        for image_id in ids:
            s, gts_per_pred = PQStats(), np.zeros(n, int)
            matched, inter_union = set(), 0
            union = np.unique(np.concatenate(gt[image_id])) if gt[image_id] else np.zeros(0, np.int64)
            for idx in gt[image_id]:
                inter = np.bincount(flat[idx], minlength=n)
                inter[0] = 0
                touching = np.flatnonzero(inter)
                gts_per_pred[touching] += 1
                frag += len(touching) >= 2
                k = int(inter.argmax())
                iou = inter[k] / (idx.size + area[k] - inter[k]) if inter[k] else 0.0
                hit = iou > MATCH_IOU
                for lo, hi, label in SIZE_BINS:
                    if lo <= idx.size < hi:
                        size_hit[label][0] += hit
                        size_hit[label][1] += 1
                if hit:
                    s.tp += 1
                    s.iou_sum += iou
                    matched.add(k)
                else:
                    s.fn += 1
            s.fp = (n - 1) - len(matched)
            merge += int((gts_per_pred[1:] >= 2).sum())
            n_gt_total += len(gt[image_id])
            n_pred_total += n - 1
            inter_union = int((flat[union] > 0).sum())
            denom = pred_px + union.size
            dices.append(2 * inter_union / denom if denom else 1.0)
            stats.append(s)
            per_site[stem[-2:]].append(s)
    row = summarize(stats)
    row.update(dice=float(np.mean(dices)),
               frag_rate=frag / n_gt_total, merge_rate=merge / max(n_pred_total, 1))
    for _, _, label in SIZE_BINS:
        hit, tot = size_hit[label]
        row[f"recall {label}"] = hit / max(tot, 1)
    sites = {site: summarize(v)["pq_pooled"] for site, v in sorted(per_site.items())}
    return row, sites


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--params-from", default=None, help="apply this run's post_params.json to all runs")
    args = ap.parse_args()

    index, polygons = load_annotations()
    val = index if args.fold < 0 else index[index["fold"] == args.fold]  # -1 = all folds (OOF)
    sets = {s: list(g["image_id"]) for s, g in val.groupby("stem")}
    gt = {i: [to_flat(m) for m in instance_masks(polygons[i], FULL_SIZE)] for i in val["image_id"]}

    rows, sites = {}, {}
    for name in args.preds:
        rows[name], sites[name] = analyze_run(name, sets, gt, args.params_from)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(pd.DataFrame(rows).T[["pq_pooled", "sq", "rq", "dice", "tp", "fp", "fn", "frag_rate", "merge_rate"]
                                + [f"recall {b[2]}" for b in SIZE_BINS]].T.to_string())
    print("\nPQ by GONG site:")
    print(pd.DataFrame(sites).to_string())


if __name__ == "__main__":
    main()
