"""Turn a filament probability map into separate filament instances.

The network predicts *where* filament pixels are (semantic segmentation).
The leaderboard (Panoptic Quality) scores *instances*, so we must decide which
pixels belong to the same filament:

1. threshold the probability map,
2. find connected components,
3. merge components that are closer than ``merge_px`` pixels
   (a filament broken by a faint gap is still one filament),
4. drop instances that are too small or not confident enough
   (each false positive costs 0.5 in the PQ denominator).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostParams:
    threshold: float = 0.5   # probability cut-off for filament pixels
    merge_px: int = 0        # merge components closer than this (in output pixels)
    min_area: int = 0        # drop instances smaller than this (in output pixels)
    min_conf: float = 0.0    # drop instances whose peak (max) probability is lower
    min_meanp: float = 0.0   # drop instances whose mean probability is lower

    def as_dict(self) -> dict:
        return asdict(self)


def to_instances(prob: np.ndarray, p: PostParams) -> np.ndarray:
    """prob: float32 HxW in [0, 1]. Returns an int32 label map (0 = background)."""
    binary = (prob > p.threshold).astype(np.uint8)
    if not binary.any():
        return np.zeros(prob.shape, np.int32)

    n, labels = cv2.connectedComponents(binary, connectivity=8)

    if p.merge_px > 0 and n > 2:
        # Dilating by r joins components whose gap is <= 2r; the merged
        # instance keeps only the original (undilated) pixels.
        r = max(1, p.merge_px // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        _, grouped = cv2.connectedComponents(cv2.dilate(binary, kernel), connectivity=8)
        labels = np.where(binary > 0, grouped, 0).astype(np.int32)

    # Filter by area and confidence, then relabel 1..K consecutively.
    flat = labels.ravel()
    n_lab = int(flat.max()) + 1
    area = np.bincount(flat, minlength=n_lab)
    max_prob, mean_prob = instance_confidence(labels, prob, area)
    keep = (area >= max(p.min_area, 1)) & (max_prob >= p.min_conf) & (mean_prob >= p.min_meanp)
    keep[0] = False
    remap = np.zeros(n_lab, np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels]


def instance_confidence(labels: np.ndarray, prob: np.ndarray, area: np.ndarray):
    """Peak and mean probability of every instance (index 0 = background)."""
    flat = labels.ravel()
    fg = flat > 0  # only ~1% of pixels are filament; skip the background
    lv, pv = flat[fg], prob.ravel()[fg].astype(np.float32)
    max_prob = np.zeros(area.size, np.float32)
    np.maximum.at(max_prob, lv, pv)
    mean_prob = np.bincount(lv, weights=pv, minlength=area.size) / np.maximum(area, 1)
    return max_prob, mean_prob


def upsample_prob(prob: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsampling of a probability map to the submission resolution."""
    if prob.shape[0] == size:
        return prob
    return cv2.resize(prob, (size, size), interpolation=cv2.INTER_LINEAR)
