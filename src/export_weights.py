"""Copy trained runs into the repository's ``weights/`` folder.

Weights are stored in float16 (half the size; GitHub limits files to 100 MB).
Inference converts them back to float32 when loading.

Example:
    .venv\\Scripts\\python.exe src\\export_weights.py --runs r34_full --post r34_f0
"""
from __future__ import annotations

import argparse
import json
import shutil

import torch

from common import MODEL_DIR, PRED_DIR, ROOT

WEIGHTS_DIR = ROOT / "weights"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under models/")
    ap.add_argument("--post", required=True, help="name under preds/ holding the tuned post_params.json")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--submitted", default="final.csv", help="file name of the Kaggle submission under submissions/")
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)
    for run in args.runs:
        src, dst = MODEL_DIR / run, WEIGHTS_DIR / run
        dst.mkdir(exist_ok=True)
        state = torch.load(src / args.ckpt, map_location="cpu", weights_only=True)
        half = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
        torch.save(half, dst / "best.pt")
        shutil.copy(src / "config.json", dst / "config.json")
        print(f"{run}: {(dst / 'best.pt').stat().st_size / 1e6:.1f} MB")

    post = json.loads((PRED_DIR / args.post / "post_params.json").read_text())
    manifest = dict(models=args.runs, post_processing=post["params"], submitted_file=args.submitted,
                    validation=dict((k, post[k]) for k in ("dice", "pq_pooled", "pq_mean", "sq", "rq")))
    (WEIGHTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", WEIGHTS_DIR / "manifest.json")


if __name__ == "__main__":
    main()
