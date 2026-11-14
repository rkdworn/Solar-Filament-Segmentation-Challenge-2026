"""Shared paths, annotation loading, image I/O and polygon rasterization.

Conventions used throughout the project
---------------------------------------
* ``stem``      : observation name without extension, e.g. ``20140609195854Bh``.
* ``image_id``  : one *annotation set* = annotator batch + stem,
                  e.g. ``040301-20140609195854Bh``. One stem can have 1-3 sets.
* Full resolution is 2048 x 2048. The network works at ``WORK_SIZE`` (1024).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG_DIR = DATA_DIR / "train" / "train_images"
TEST_IMG_DIR = DATA_DIR / "test" / "test_images"
ANN_PATH = DATA_DIR / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE_DIR = ROOT / "data" / "cache"
MODEL_DIR = ROOT / "models"
PRED_DIR = ROOT / "preds"
SUB_DIR = ROOT / "submissions"

FULL_SIZE = 2048
WORK_SIZE = 1024  # default network resolution; 2048 = native resolution
N_FOLDS = 5
SEED = 42


def img_cache_dir(size: int) -> Path:
    return CACHE_DIR / f"img{size}"


def mask_cache_dir(size: int) -> Path:
    return CACHE_DIR / f"mask{size}"


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def load_annotations(path: Path = ANN_PATH) -> tuple[pd.DataFrame, dict[str, list[list[float]]]]:
    """Load the COCO-style training annotations.

    Returns
    -------
    index : DataFrame with one row per annotation set
            (columns: image_id, stem, annotator, n_filaments, fold).
    polygons : image_id -> list of flat polygons ``[x0, y0, x1, y1, ...]``
               in full-resolution (2048) pixel coordinates.
    """
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)

    polygons: dict[str, list[list[float]]] = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        # Each filament is a single closed polygon (documented by the host).
        polygons[ann["image_id"]].append(ann["segmentation"][0])

    rows = []
    for img in coco["images"]:
        stem = Path(img["file_name"]).stem
        rows.append(
            dict(
                image_id=img["id"],
                stem=stem,
                annotator=img["id"].split("-")[0],
                n_filaments=len(polygons[img["id"]]),
            )
        )
    index = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    index["fold"] = index["stem"].map(make_folds(index["stem"].unique()))
    return index, polygons


def make_folds(stems, n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    """Assign folds by *observation* (stem), not by annotation set.

    The same observation annotated by different people must stay in the same
    fold, otherwise the validation score leaks through the shared image.
    """
    stems = sorted(stems)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stems))
    return {stems[i]: int(k % n_folds) for k, i in enumerate(order)}


# --------------------------------------------------------------------------- #
# Rasterization (pycocotools = the same routine the host uses for RLE)
# --------------------------------------------------------------------------- #
def polygon_to_mask(poly: list[float], size: int = FULL_SIZE) -> np.ndarray:
    """Rasterize one full-resolution polygon at ``size`` x ``size`` (uint8 0/1)."""
    scale = size / FULL_SIZE
    scaled = (np.asarray(poly, dtype=np.float64) * scale).tolist()
    rle = mask_utils.frPyObjects([scaled], size, size)
    return mask_utils.decode(rle)[:, :, 0]


def instance_masks(polys: list[list[float]], size: int = FULL_SIZE) -> list[np.ndarray]:
    """One binary mask per ground-truth filament."""
    return [polygon_to_mask(p, size) for p in polys]


def union_mask(polys: list[list[float]], size: int = FULL_SIZE) -> np.ndarray:
    """Semantic (filament vs background) mask used as the training target."""
    out = np.zeros((size, size), np.uint8)
    for p in polys:
        out |= polygon_to_mask(p, size)
    return out


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def read_gray(path: Path) -> np.ndarray:
    """H-alpha observations are single-channel; never treat them as RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def normalize_disk(img: np.ndarray, disk_threshold: int = 20) -> np.ndarray:
    """Per-image standardization using on-disk pixels only.

    GONG sites differ in exposure and contrast. Using the median / std of the
    solar disk (pixels brighter than the dark sky) makes the input comparable
    across observatories. Output is float32, clipped to [-6, 6].
    """
    img = img.astype(np.float32)
    disk = img[img > disk_threshold]
    if disk.size < 1000:  # degenerate image; fall back to global stats
        disk = img.ravel()
    med = float(np.median(disk))
    std = float(disk.std()) + 1e-6
    return np.clip((img - med) / std, -6.0, 6.0)
