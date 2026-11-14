"""Evaluation metrics: Panoptic Quality (leaderboard) and Dice.

PQ follows the host's definition (Kirillov et al., CVPR 2019):

    PQ = sum_{(y, y_hat) in TP} IoU(y, y_hat) / (|TP| + 0.5 |FP| + 0.5 |FN|)

A predicted and a ground-truth segment are matched when IoU > 0.5. With
IoU > 0.5 the matching is unique, so no Hungarian assignment is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MATCH_IOU = 0.5


@dataclass
class PQStats:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __iadd__(self, other: "PQStats") -> "PQStats":
        self.iou_sum += other.iou_sum
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denom if denom > 0 else 1.0

    @property
    def sq(self) -> float:  # segmentation quality: mean IoU of matched pairs
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:  # recognition quality: an F1 score on instances
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom > 0 else 1.0


def to_flat(mask: np.ndarray) -> np.ndarray:
    """Compact representation of one instance: flat indices of its pixels."""
    return np.flatnonzero(mask)


def pq_stats(pred_labels: np.ndarray, gt_instances: list[np.ndarray]) -> PQStats:
    """PQ statistics for one image.

    pred_labels  : int label map, 0 = background, 1..K = predicted filaments
                   (predictions never overlap).
    gt_instances : one entry per ground-truth filament, either a 2-D binary
                   mask or the 1-D flat pixel indices from ``to_flat``
                   (ground-truth polygons may overlap slightly).
    """
    flat = pred_labels.ravel()
    n_pred = int(flat.max())
    pred_area = np.bincount(flat, minlength=n_pred + 1)
    matched_pred: set[int] = set()
    stats = PQStats()
    for gt in gt_instances:
        idx = gt if gt.ndim == 1 else np.flatnonzero(gt)
        gt_area = int(idx.size)
        if gt_area == 0:
            continue
        inter = np.bincount(flat[idx], minlength=n_pred + 1)
        inter[0] = 0
        k = int(inter.argmax())
        if inter[k] == 0:
            stats.fn += 1
            continue
        iou = inter[k] / (gt_area + pred_area[k] - inter[k])
        if iou > MATCH_IOU:
            stats.tp += 1
            stats.iou_sum += float(iou)
            matched_pred.add(k)
        else:
            stats.fn += 1
    stats.fp = n_pred - len(matched_pred)
    return stats


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice on the union masks of one image (1.0 if both are empty)."""
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return 2.0 * np.logical_and(p, g).sum() / denom if denom else 1.0


def summarize(per_image: list[PQStats]) -> dict[str, float]:
    """Both aggregations: pooled over the dataset and averaged per image.

    The host does not state which one the leaderboard uses, so we track both.
    """
    pooled = PQStats()
    for s in per_image:
        pooled += s
    return dict(
        pq_pooled=pooled.pq,
        pq_mean=float(np.mean([s.pq for s in per_image])) if per_image else 0.0,
        sq=pooled.sq,
        rq=pooled.rq,
        tp=pooled.tp,
        fp=pooled.fp,
        fn=pooled.fn,
    )
