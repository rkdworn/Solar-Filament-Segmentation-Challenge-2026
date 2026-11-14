"""Descriptors of candidate filament instances (the inputs of the instance scorer).

A candidate is one connected component of the thresholded probability map
(after fragment merging, exactly as in ``postprocess.label_components``).
For each candidate we compute

* probability inside it: mean, spread, quantiles, share of pixels above 0.9;
* shape: area, bounding box, fill ratio, axes of the equivalent ellipse,
  elongation, boundary length, skeleton length and mean width;
* stability across thresholds (in the spirit of MSER): how much larger the
  region containing it is at a slightly lower threshold, how many pieces it
  breaks into and how much of it survives at a slightly higher threshold;
* image context: number of candidates and share of the map above threshold.

Only the probability map is used, so the scorer can be applied to the maps of
any model or ensemble (after retraining it on that model's OOF maps).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize

FEATURES = [
    "log_area", "meanp", "stdp", "p10", "p50", "p90", "pmax", "frac90",        # probability
    "bbox_h", "bbox_w", "fill", "major", "minor", "elong", "perim_rel",        # shape
    "skel", "width",
    "lower_ratio", "n_upper", "upper_ratio", "n_siblings",                      # stability
    "n_candidates", "map_frac",                                                 # image context
]


def instance_features(labels: np.ndarray, prob: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      min_area: int = 1) -> pd.DataFrame:
    """One row per component of ``labels`` with at least ``min_area`` pixels (index = label id).

    lower / upper : label maps of the same map at a slightly lower / higher
                    threshold (same merge radius). Components are nested:
                    every component of ``labels`` lies inside one ``lower``
                    component and contains zero or more ``upper`` components.
    """
    n = int(labels.max()) + 1
    flat = labels.ravel()
    fg = np.flatnonzero(flat)
    lv, pv = flat[fg], prob.ravel()[fg].astype(np.float64)
    area = np.bincount(lv, minlength=n).astype(np.float64)
    a1 = np.maximum(area, 1)

    # probability statistics; quantiles from one sort of all pixels by (label, probability)
    meanp = np.bincount(lv, pv, n) / a1
    stdp = np.sqrt(np.maximum(np.bincount(lv, pv ** 2, n) / a1 - meanp ** 2, 0))
    frac90 = np.bincount(lv, (pv > 0.9).astype(np.float64), n) / a1
    sorted_p = pv[np.lexsort((pv, lv))]
    count = area[1:].astype(np.int64)
    start = np.cumsum(count) - count  # first sorted pixel of labels 1..n-1

    def quantile(q: float) -> np.ndarray:
        out = np.zeros(n)
        ok = count > 0
        out[1:][ok] = sorted_p[start[ok] + np.floor(q * (count[ok] - 1)).astype(np.int64)]
        return out

    # second moments -> axes of the ellipse with the same inertia
    ys, xs = np.divmod(fg, labels.shape[1])
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    mx, my = np.bincount(lv, xs, n) / a1, np.bincount(lv, ys, n) / a1
    cxx = np.bincount(lv, xs * xs, n) / a1 - mx ** 2
    cyy = np.bincount(lv, ys * ys, n) / a1 - my ** 2
    cxy = np.bincount(lv, xs * ys, n) / a1 - mx * my
    half_trace = (cxx + cyy) / 2
    disc = np.sqrt(np.maximum(half_trace ** 2 - (cxx * cyy - cxy ** 2), 0))
    major = 4 * np.sqrt(half_trace + disc)
    minor = 4 * np.sqrt(np.maximum(half_trace - disc, 0))

    # stability: containing region at the lower threshold, pieces at the higher one
    lower_id = np.zeros(n, np.int64)
    lower_id[lv] = lower.ravel()[fg]  # nested, so all pixels of a component agree
    lower_area = np.bincount(lower.ravel(), minlength=int(lower.max()) + 1)[lower_id]
    candidate = area >= max(min_area, 1)
    candidate[0] = False
    siblings = np.bincount(lower_id[candidate], minlength=int(lower.max()) + 1)[lower_id]
    up_flat = upper.ravel()
    up_fg = np.flatnonzero(up_flat)
    up_area = np.bincount(up_flat[up_fg], minlength=int(upper.max()) + 1)
    up_owner = np.zeros(up_area.size, np.int64)
    up_owner[up_flat[up_fg]] = flat[up_fg]
    big = up_area >= max(min_area, 1)  # ignore crumbs at the higher threshold
    big[0] = False
    n_upper = np.bincount(up_owner[big], minlength=n)
    upper_area = np.bincount(up_owner[1:], weights=up_area[1:], minlength=n)

    q10, q50, q90, qmax = quantile(0.1), quantile(0.5), quantile(0.9), quantile(1.0)
    # descriptors that need the component's own pixels: bounding box, boundary, skeleton
    keep = np.flatnonzero(candidate)
    bbox = np.zeros((len(keep), 2))
    perimeter, skel = np.zeros(len(keep)), np.ones(len(keep))
    slices = ndimage.find_objects(labels)
    for i, k in enumerate(keep):
        crop = labels[slices[k - 1]] == k
        bbox[i] = crop.shape
        contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter[i] = sum(cv2.arcLength(c, True) for c in contours)
        skel[i] = max(int(skeletonize(crop).sum()), 1)

    a = area[keep]
    return pd.DataFrame(dict(
        area=a, log_area=np.log(np.maximum(a, 1)),
        meanp=meanp[keep], stdp=stdp[keep], p10=q10[keep], p50=q50[keep], p90=q90[keep], pmax=qmax[keep],
        frac90=frac90[keep],
        bbox_h=bbox[:, 0], bbox_w=bbox[:, 1], fill=a / np.maximum(bbox[:, 0] * bbox[:, 1], 1),
        major=major[keep], minor=minor[keep], elong=major[keep] / np.maximum(minor[keep], 1.0),
        perim_rel=perimeter / np.sqrt(a), skel=skel, width=a / skel,
        lower_ratio=a / lower_area[keep], n_upper=n_upper[keep], upper_ratio=upper_area[keep] / a,
        n_siblings=siblings[keep],
        n_candidates=float(len(keep)), map_frac=fg.size / flat.size,
    ), index=pd.Index(keep, name="label"))
