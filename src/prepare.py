"""Build the image / mask cache at a given network resolution (run once per size).

* data/cache/img{S}/{stem}.png          : grayscale image resized to S x S
* data/cache/mask{S}/{image_id}.png     : union filament mask of one annotation set
* data/cache/train_index.csv            : annotation sets with their CV fold

Usage:
    .venv\\Scripts\\python.exe src\\prepare.py              (1024, default)
    .venv\\Scripts\\python.exe src\\prepare.py --size 2048  (native resolution)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm

from common import (CACHE_DIR, FULL_SIZE, TEST_IMG_DIR, TRAIN_IMG_DIR, WORK_SIZE, img_cache_dir,
                    load_annotations, mask_cache_dir, read_gray, union_mask)


def _resize_image(src, dst_dir, size):
    dst = dst_dir / f"{src.stem}.png"
    if not dst.exists():
        img = read_gray(src)
        if size != FULL_SIZE:
            # INTER_AREA averages pixels, which keeps thin dark filaments visible.
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=WORK_SIZE)
    size = ap.parse_args().size

    img_dir, mask_dir = img_cache_dir(size), mask_cache_dir(size)
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(TRAIN_IMG_DIR.glob("*.jpeg")) + sorted(TEST_IMG_DIR.glob("*.jpeg"))
    with ThreadPoolExecutor(8) as ex:
        list(tqdm(ex.map(lambda s: _resize_image(s, img_dir, size), sources), total=len(sources), desc="images"))

    index, polygons = load_annotations()
    for image_id in tqdm(index["image_id"], desc="masks"):
        dst = mask_dir / f"{image_id}.png"
        if not dst.exists():
            cv2.imwrite(str(dst), union_mask(polygons[image_id], size) * 255)

    index.to_csv(CACHE_DIR / "train_index.csv", index=False)
    print(index.groupby("fold").agg(sets=("image_id", "size"), stems=("stem", "nunique")))


if __name__ == "__main__":
    main()
