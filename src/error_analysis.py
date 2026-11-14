"""Break PQ errors down by type (validation fold, tuned post-processing, 2048 px).

False negatives (ground-truth filaments without a match):
  * missed      : no predicted pixel overlaps the filament at all
  * low IoU     : overlapped, but best IoU <= 0.5 (boundary / fragment / merge)
False positives (predictions without a match):
  * no overlap  : overlaps no filament of this annotator
  * low IoU     : overlaps a filament, but IoU <= 0.5
For observations with several annotators, a "no overlap" false positive is
also checked against the *other* annotators' filaments: if one of them
matches it (IoU > 0.5), the error is a disagreement between experts rather
than a hallucination.

Example:
    .venv\\Scripts\\python.exe src\\error_analysis.py --preds r34_f0 r34_2048_f0 --fold 0
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from common import FULL_SIZE, PRED_DIR, instance_masks, load_annotations
from metrics import MATCH_IOU, to_flat
from postprocess import PostParams, to_instances, upsample_prob


def best_iou(flat: np.ndarray, area: np.ndarray, idx: np.ndarray) -> tuple[int, float, int]:
    """(best prediction id, its IoU, total overlapping pixels) for one GT filament."""
    inter = np.bincount(flat[idx], minlength=area.size)
    inter[0] = 0
    k = int(inter.argmax())
    iou = inter[k] / (idx.size + area[k] - inter[k]) if inter[k] else 0.0
    return k, float(iou), int(inter.sum())


def analyze(name: str, sets: dict, gt: dict) -> dict:
    params = PostParams(**json.loads((PRED_DIR / name / "post_params.json").read_text())["params"])
    c = dict(fn_missed=0, fn_low_iou=0, fp_no_overlap=0, fp_low_iou=0, fp_matches_other_annotator=0)
    for stem, ids in sets.items():
        p8 = cv2.imread(str(PRED_DIR / name / "val" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        labels = to_instances(upsample_prob(p8.astype(np.float32) / 255.0, FULL_SIZE), params)
        flat = labels.ravel()
        area = np.bincount(flat, minlength=int(flat.max()) + 1)
        # which predictions match (IoU > 0.5) a filament of each annotator
        matched_by = {}
        for image_id in ids:
            m = set()
            for idx in gt[image_id]:
                k, iou, _ = best_iou(flat, area, idx)
                if iou > MATCH_IOU:
                    m.add(k)
            matched_by[image_id] = m
        for image_id in ids:
            touched = set()
            for idx in gt[image_id]:
                k, iou, overlap = best_iou(flat, area, idx)
                touched |= set(np.flatnonzero(np.bincount(flat[idx], minlength=area.size)[1:]) + 1)
                if iou <= MATCH_IOU:
                    c["fn_missed" if overlap == 0 else "fn_low_iou"] += 1
            others = set().union(*(matched_by[o] for o in ids if o != image_id)) if len(ids) > 1 else set()
            for k in range(1, area.size):
                if k in matched_by[image_id]:
                    continue
                if k in touched:
                    c["fp_low_iou"] += 1
                else:
                    c["fp_no_overlap"] += 1
                    c["fp_matches_other_annotator"] += k in others
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    index, polygons = load_annotations()
    val = index if args.fold < 0 else index[index["fold"] == args.fold]  # -1 = all folds (OOF)
    sets = {s: list(g["image_id"]) for s, g in val.groupby("stem")}
    gt = {i: [to_flat(m) for m in instance_masks(polygons[i], FULL_SIZE)] for i in val["image_id"]}
    print(pd.DataFrame({name: analyze(name, sets, gt) for name in args.preds}).to_string())


if __name__ == "__main__":
    main()
