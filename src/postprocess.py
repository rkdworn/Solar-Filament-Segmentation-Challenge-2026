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

Step 4 is either a fixed rule (``min_area``, ``min_conf``, ``min_meanp``) or,
when ``scorer`` is set, a learned one: a regression model trained on
out-of-fold maps (``instance_scorer.py``) predicts each instance's *matched
IoU* (its IoU with an annotated filament if that exceeds 0.5, else 0,
averaged over annotators), and instances below ``min_score`` are dropped.
Why this is the right quantity: reporting an instance always adds 0.5 to the
PQ denominator (a true positive removes a false negative, anything else is a
false positive) and adds its matched IoU to the numerator, so it raises PQ
exactly when its matched IoU exceeds PQ / 2. The best ``min_score`` is
therefore close to half the attainable PQ (about 0.2 here).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from common import ROOT
from instance_features import instance_features


@dataclass(frozen=True)
class PostParams:
    threshold: float = 0.5   # probability cut-off for filament pixels
    merge_px: int = 0        # merge components closer than this (in output pixels)
    min_area: int = 0        # drop instances smaller than this (in output pixels)
    min_conf: float = 0.0    # drop instances whose peak (max) probability is lower
    min_meanp: float = 0.0   # drop instances whose mean probability is lower
    scorer: str = ""         # learned filter: saved instance scorer (path, absolute or relative to the repo)
    min_score: float = 0.0   # learned filter: drop instances whose predicted matched IoU is not higher

    def as_dict(self) -> dict:
        return asdict(self)


def to_instances(prob: np.ndarray, p: PostParams) -> np.ndarray:
    """prob: float32 HxW in [0, 1]. Returns an int32 label map (0 = background)."""
    labels = label_components(prob, p.threshold, p.merge_px)
    if not labels.any():
        return labels

    # Filter by area and confidence (and the learned score), then relabel 1..K consecutively.
    flat = labels.ravel()
    n_lab = int(flat.max()) + 1
    area = np.bincount(flat, minlength=n_lab)
    max_prob, mean_prob = instance_confidence(labels, prob, area)
    keep = (area >= max(p.min_area, 1)) & (max_prob >= p.min_conf) & (mean_prob >= p.min_meanp)
    keep[0] = False
    if p.scorer and keep.any():
        keep &= predict_matched_iou(labels, prob, p) > p.min_score
    remap = np.zeros(n_lab, np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels]


def label_components(prob: np.ndarray, threshold: float, merge_px: int = 0) -> np.ndarray:
    """Steps 1-3: components of ``prob > threshold`` (8-connected), fragments within ``merge_px`` merged.

    Dilating by r joins components whose gap is <= 2r; the merged instance
    keeps only the original (undilated) pixels. Labels are 1..K.
    """
    binary = (prob > threshold).astype(np.uint8)
    if not binary.any():
        return np.zeros(prob.shape, np.int32)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    if merge_px > 0 and n > 2:
        r = max(1, merge_px // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        _, grouped = cv2.connectedComponents(cv2.dilate(binary, kernel), connectivity=8)
        labels = np.where(binary > 0, grouped, 0).astype(np.int32)
    return labels


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


# --------------------------------------------------------------------------- #
# Learned filter
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def load_scorer(path: str) -> dict:
    """Saved scorer (see ``instance_scorer.py``), loaded once per process.

    A dict with the fitted regressors (``models``, a bagged ensemble), their
    ``features``, the candidate definition they were trained for
    (``threshold``, ``merge_px``, ``min_area``) and the threshold step of the
    stability features.
    """
    import joblib  # only the learned filter needs it

    file = Path(path)
    return joblib.load(file if file.is_absolute() else ROOT / file)


def candidate_features(labels: np.ndarray, prob: np.ndarray, threshold: float, merge_px: int, step: float,
                       min_area: int):
    """Descriptors of every candidate, including the maps one ``step`` below / above the threshold."""
    lower = label_components(prob, threshold - step, merge_px)
    upper = label_components(prob, threshold + step, merge_px)
    return instance_features(labels, prob, lower, upper, min_area)


def score_candidates(scorer: dict, feats) -> np.ndarray:
    """Predicted matched IoU of each row of ``feats``: mean over the bagged regressors."""
    x = feats[scorer["features"]].to_numpy(np.float32)
    return np.mean([model.predict(x) for model in scorer["models"]], axis=0)


def predict_matched_iou(labels: np.ndarray, prob: np.ndarray, p: PostParams) -> np.ndarray:
    """Predicted matched IoU per label (index 0 = background; -1 for labels too small to score)."""
    s = load_scorer(p.scorer)
    if (p.threshold, p.merge_px) != (s["threshold"], s["merge_px"]):
        raise ValueError(f"scorer {p.scorer} was trained for threshold={s['threshold']}, "
                         f"merge_px={s['merge_px']}; got {p.threshold}, {p.merge_px}")
    feats = candidate_features(labels, prob, p.threshold, p.merge_px, s["step"], s["min_area"])
    score = np.full(int(labels.max()) + 1, -1.0)
    if len(feats):
        score[feats.index.to_numpy()] = score_candidates(s, feats)
    return score
