"""Build the image / mask cache at a given network resolution (run once per size).

* data/cache/img{S}/{stem}.png          : grayscale image resized to S x S
* data/cache/mask{S}/{image_id}.png     : union filament mask of one annotation set
* data/cache/spine{S}/{image_id}.png    : spine (center line) mask, optional training aid
* data/cache/train_index.csv            : annotation sets with their CV fold

Usage:
    .venv\\Scripts\\python.exe src\\prepare.py                          (1024, default)
    .venv\\Scripts\\python.exe src\\prepare.py --size 2048              (native resolution)
    .venv\\Scripts\\python.exe src\\prepare.py --test-only              (inference only)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm

from common import (CACHE_DIR, FULL_SIZE, TEST_IMG_DIR, TRAIN_IMG_DIR, WORK_SIZE, img_cache_dir,
                    load_annotations, load_spines, mask_cache_dir, read_gray, spine_cache_dir, spine_mask,
                    union_mask)


def _resize_image(src, dst_dir, size):
    dst = dst_dir / f"{src.stem}.png"
    if not dst.exists():
        img = read_gray(src)
        if size != FULL_SIZE:
            # INTER_AREA averages pixels, which keeps thin dark filaments visible.
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img)


def cache_images(size: int = WORK_SIZE, include_train: bool = True) -> None:
    """Resized grayscale copies of the test (and optionally training) images."""
    out = img_cache_dir(size)
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if include_train:
        sources = sorted(TRAIN_IMG_DIR.glob("*.jpeg")) + sources
    with ThreadPoolExecutor(8) as ex:
        list(tqdm(ex.map(lambda s: _resize_image(s, out, size), sources), total=len(sources), desc="images"))


def cache_targets(size: int = WORK_SIZE) -> None:
    """Training targets: union filament masks and spine masks per annotation set."""
    index, polygons = load_annotations()
    spines = load_spines()
    mask_dir, spine_dir = mask_cache_dir(size), spine_cache_dir(size)
    mask_dir.mkdir(parents=True, exist_ok=True)
    spine_dir.mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(index["image_id"], desc="masks"):
        if not (mask_dir / f"{image_id}.png").exists():
            cv2.imwrite(str(mask_dir / f"{image_id}.png"), union_mask(polygons[image_id], size) * 255)
        if not (spine_dir / f"{image_id}.png").exists():
            cv2.imwrite(str(spine_dir / f"{image_id}.png"), spine_mask(spines[image_id], size) * 255)
    index.to_csv(CACHE_DIR / "train_index.csv", index=False)
    print(index.groupby("fold").agg(sets=("image_id", "size"), stems=("stem", "nunique")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=WORK_SIZE)
    ap.add_argument("--test-only", action="store_true", help="only resize test images (for inference)")
    args = ap.parse_args()
    cache_images(args.size, include_train=not args.test_only)
    if not args.test_only:
        cache_targets(args.size)


if __name__ == "__main__":
    main()
