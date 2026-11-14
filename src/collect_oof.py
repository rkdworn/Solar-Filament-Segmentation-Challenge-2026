"""Gather the out-of-fold validation maps of a 5-fold run into one folder.

preds/<prefix>_f{k}/val/*.png  ->  preds/<prefix>_oof/val/*.png
The result covers every training observation exactly once, so post-processing
can be tuned on all 707 images (``tune.py --pred <prefix>_oof --fold -1``).

Example:
    .venv\\Scripts\\python.exe src\\collect_oof.py --prefix r34
"""
from __future__ import annotations

import argparse
import shutil

from common import N_FOLDS, PRED_DIR, load_annotations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    index, _ = load_annotations()
    out = PRED_DIR / f"{args.prefix}_oof" / "val"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(N_FOLDS):
        stems = sorted(index.loc[index["fold"] == k, "stem"].unique())
        src = PRED_DIR / f"{args.prefix}_f{k}" / "val"
        for s in stems:
            shutil.copy(src / f"{s}.png", out / f"{s}.png")
        print(f"fold {k}: {len(stems)} observations from {src}")


if __name__ == "__main__":
    main()
