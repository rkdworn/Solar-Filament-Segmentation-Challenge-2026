"""Write the Kaggle submission CSV from test probability maps.

One row per predicted filament:
    filament_id      = <observation stem>_<k>
    segmentation_rle = COCO compressed RLE *counts* of a 2048x2048 mask
                       (pycocotools ``mask.encode``; size is fixed, not stored)

Example:
    .venv\\Scripts\\python.exe src\\make_submission.py --pred r34_f0 --params r34_f0 --out r34_f0_v1
    .venv\\Scripts\\python.exe src\\make_submission.py --pred final --params weights\\manifest.json --out final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from common import FULL_SIZE, PRED_DIR, SUB_DIR, TEST_IMG_DIR
from postprocess import PostParams, to_instances, upsample_prob


def load_params(spec: str) -> PostParams:
    """``weights/manifest.json`` (repository) or a run name under preds/ (post_params.json)."""
    path = Path(spec)
    if path.suffix == ".json":
        return PostParams(**json.loads(path.read_text())["post_processing"])
    return PostParams(**json.loads((PRED_DIR / spec / "post_params.json").read_text())["params"])


def encode(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def build_submission(prob_dir: Path, stems: list[str], params: PostParams) -> pd.DataFrame:
    """Probability PNGs -> one row per predicted filament (non-overlapping masks)."""
    rows, empty = [], []
    for stem in tqdm(stems):
        p8 = cv2.imread(str(prob_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if p8 is None:
            raise FileNotFoundError(f"missing probability map for {stem}")
        labels = to_instances(upsample_prob(p8.astype(np.float32) / 255.0, FULL_SIZE), params)
        n = int(labels.max())
        if n == 0:
            empty.append(stem)
        for k in range(1, n + 1):
            rows.append(dict(filament_id=f"{stem}_{k}", segmentation_rle=encode(labels == k)))
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    print(f"{len(sub)} filaments in {len(stems) - len(empty)}/{len(stems)} images; empty images: {len(empty)}")
    return sub


def validate(sub: pd.DataFrame) -> None:
    """Format checks: unique ids, decodable 2048x2048 non-empty masks, no quotes/commas."""
    assert sub["filament_id"].is_unique
    assert not sub["segmentation_rle"].str.contains("[,\"']").any()
    for r in sub.sample(min(20, len(sub)), random_state=0).itertuples():
        m = mask_utils.decode({"size": [FULL_SIZE, FULL_SIZE], "counts": r.segmentation_rle.encode("ascii")})
        assert m.shape == (FULL_SIZE, FULL_SIZE) and m.sum() > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="name under preds/ (uses preds/<name>/test)")
    ap.add_argument("--params", required=True, help="weights/manifest.json, or a name under preds/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = load_params(args.params)
    print("post-processing:", params)
    prob_dir = PRED_DIR / args.pred / "test"
    stems = sorted(p.stem for p in TEST_IMG_DIR.glob("*.jpeg"))  # every test image must have a map
    sub = build_submission(prob_dir, stems, params)
    validate(sub)
    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"{args.out}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
