"""How well do human annotators agree with each other? (human ceiling)

For every observation annotated by 2-3 people, treat one annotation set as the
"prediction" and another as the "ground truth" and compute the leaderboard
metrics. This tells us roughly how high a score is achievable and is reported
in the technical report.

Usage:  .venv\\Scripts\\python.exe src\\annotator_agreement.py
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from common import FULL_SIZE, instance_masks, load_annotations
from metrics import dice, pq_stats, summarize


def to_label_map(masks: list[np.ndarray]) -> np.ndarray:
    labels = np.zeros((FULL_SIZE, FULL_SIZE), np.int32)
    for k, m in enumerate(masks, start=1):
        labels[m > 0] = k  # later polygons win where human polygons overlap
    return labels


def main() -> None:
    index, polygons = load_annotations()
    multi = index.groupby("stem").filter(lambda g: len(g) > 1)
    stats, dices = [], []
    for _, g in multi.groupby("stem"):
        masks = {i: instance_masks(polygons[i]) for i in g["image_id"]}
        for a, b in permutations(g["image_id"], 2):
            stats.append(pq_stats(to_label_map(masks[a]), masks[b]))
            dices.append(dice(np.any(masks[a], axis=0), np.any(masks[b], axis=0)))
    res = summarize(stats)
    print(f"observations with >1 annotator: {multi['stem'].nunique()}  pairs: {len(stats)}")
    print(f"human-vs-human  PQ pooled {res['pq_pooled']:.4f} | PQ mean {res['pq_mean']:.4f} | "
          f"SQ {res['sq']:.4f} | RQ {res['rq']:.4f} | Dice mean {np.mean(dices):.4f}")


if __name__ == "__main__":
    main()
