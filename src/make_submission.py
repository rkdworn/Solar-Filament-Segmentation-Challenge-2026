"""Write the Kaggle submission CSV from test probability maps.

One row per predicted filament:
    filament_id      = <observation stem>_<k>
    segmentation_rle = COCO compressed RLE *counts* of a 2048x2048 mask
                       (pycocotools ``mask.encode``; size is fixed, not stored)

Example:
    .venv\\Scripts\\python.exe src\\make_submission.py --pred r34_f0 --params r34_f0 --out r34_f0_v1
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from common import FULL_SIZE, PRED_DIR, SUB_DIR, TEST_IMG_DIR
from postprocess import PostParams, to_instances, upsample_prob


def encode(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="name under preds/ (uses preds/<name>/test)")
    ap.add_argument("--params", required=True, help="name under preds/ holding post_params.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = PostParams(**json.loads((PRED_DIR / args.params / "post_params.json").read_text())["params"])
    print("post-processing:", params)
    stems = sorted(p.stem for p in TEST_IMG_DIR.glob("*.jpeg"))

    rows, empty = [], []
    for stem in tqdm(stems):
        p8 = cv2.imread(str(PRED_DIR / args.pred / "test" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if p8 is None:
            raise FileNotFoundError(f"missing test prediction for {stem}")
        labels = to_instances(upsample_prob(p8.astype(np.float32) / 255.0, FULL_SIZE), params)
        n = int(labels.max())
        if n == 0:
            empty.append(stem)
        for k in range(1, n + 1):
            rows.append(dict(filament_id=f"{stem}_{k}", segmentation_rle=encode(labels == k)))

    sub = pd.DataFrame(rows)
    # Round-trip check: decoding must give back a 2048x2048 mask of the same area.
    for r in sub.sample(min(5, len(sub)), random_state=0).itertuples():
        m = mask_utils.decode({"size": [FULL_SIZE, FULL_SIZE], "counts": r.segmentation_rle.encode("ascii")})
        assert m.shape == (FULL_SIZE, FULL_SIZE) and m.sum() > 0
    assert not sub["segmentation_rle"].str.contains("[,\"']").any()

    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"{args.out}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} filaments in {len(stems) - len(empty)}/{len(stems)} images; "
          f"empty images: {len(empty)}")


if __name__ == "__main__":
    main()
