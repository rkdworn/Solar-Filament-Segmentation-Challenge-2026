"""Copy trained runs into the repository's ``weights/`` folder.

Weights are stored in float16 (half the size; GitHub limits files to 100 MB).
Inference converts them back to float32 when loading.

Example:
    .venv\\Scripts\\python.exe src\\export_weights.py --runs r34_full --post r34_f0
    .venv\\Scripts\\python.exe src\\export_weights.py --runs r34_full --post r34_oof --post-file post_params_learned.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil

import torch

from common import MODEL_DIR, PRED_DIR, ROOT

WEIGHTS_DIR = ROOT / "weights"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under models/")
    ap.add_argument("--post", required=True, help="name under preds/ holding the tuned post-processing")
    ap.add_argument("--post-file", default="post_params.json",
                    help="parameter file in preds/<post>/, e.g. post_params_learned.json for the learned filter")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--submitted", default="final.csv", help="file name of the Kaggle submission under submissions/")
    ap.add_argument("--tta", default="d4", choices=["none", "flip4", "d4"],
                    help="test-time augmentation the submission was predicted with (read by the notebook)")
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

    post = json.loads((PRED_DIR / args.post / args.post_file).read_text())
    params = post["params"]
    if params.get("scorer"):  # learned filter: ship the fitted instance scorer with the weights
        shutil.copy(ROOT / params["scorer"], WEIGHTS_DIR / "instance_scorer.joblib")
        params["scorer"] = "weights/instance_scorer.joblib"
    manifest = dict(models=args.runs, tta=args.tta, post_processing=params, submitted_file=args.submitted,
                    validation={k: post[k] for k in ("dice", "pq_pooled", "pq_mean", "sq", "rq") if k in post})
    submitted = ROOT / "submissions" / args.submitted
    if submitted.exists():  # fingerprint of the submitted file (submissions/ itself is not in the repository)
        manifest["submitted_sha256"] = hashlib.sha256(submitted.read_bytes()).hexdigest()
        manifest["submitted_rows"] = sum(1 for _ in submitted.open()) - 1
    (WEIGHTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", WEIGHTS_DIR / "manifest.json")


if __name__ == "__main__":
    main()
