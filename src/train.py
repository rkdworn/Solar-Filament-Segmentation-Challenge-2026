"""Train a filament segmentation model on one CV fold (or on all data).

Example:
    .venv\\Scripts\\python.exe src\\train.py --name r34_f0 --encoder resnet34 --fold 0
    .venv\\Scripts\\python.exe src\\train.py --name r34_full --encoder resnet34 --fold -1

Validation (every epoch) runs on whole 1024x1024 images of the held-out
observations and reports
  * Dice  : mean per annotation set (union masks), and
  * PQ    : Panoptic Quality after default post-processing.
The checkpoint with the best (Dice + PQ) / 2 is kept as ``best.pt``.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from common import MODEL_DIR, SEED, WORK_SIZE, instance_masks, load_annotations, mask_cache_dir
from dataset import FilamentDataset, load_image_tensor
from metrics import dice, pq_stats, summarize, to_flat
from model import amp_dtype, build_model, predict_prob
from postprocess import PostParams, to_instances

def default_post(size: int) -> PostParams:
    """Fixed post-processing used only to monitor training (tuned later)."""
    return PostParams(threshold=0.5, merge_px=0, min_area=int(20 * (size / 1024) ** 2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--fold", type=int, default=0, help="-1 = train on all data, no validation")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--size", type=int, default=WORK_SIZE, help="network resolution: 1024 or 2048")
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--spine", action="store_true", help="auxiliary center-line (spine) output")
    ap.add_argument("--spine-weight", type=float, default=0.5)
    return ap.parse_args()


class Validator:
    """Held-out observations and their ground truth at the network resolution."""

    def __init__(self, index: pd.DataFrame, polygons: dict, fold: int, size: int):
        self.size = size
        self.post = default_post(size)
        val = index[index["fold"] == fold]
        self.stems = sorted(val["stem"].unique())
        self.sets = {s: list(g["image_id"]) for s, g in val.groupby("stem")}
        self.gt_inst = {i: [to_flat(m) for m in instance_masks(polygons[i], size)] for i in val["image_id"]}

    def _union(self, image_id: str) -> np.ndarray:
        return cv2.imread(str(mask_cache_dir(self.size) / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 127

    @torch.no_grad()
    def run(self, model: torch.nn.Module) -> dict:
        model.eval()
        dices, stats = [], []
        for stem in self.stems:
            prob = predict_prob(model, load_image_tensor(stem, self.size).cuda(), tta=False)[0, 0].cpu().numpy()
            labels = to_instances(prob, self.post)
            for image_id in self.sets[stem]:
                dices.append(dice(prob > self.post.threshold, self._union(image_id)))
                stats.append(pq_stats(labels, self.gt_inst[image_id]))
        res = summarize(stats)
        res["dice"] = float(np.mean(dices))
        return res


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    run_dir = MODEL_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    index, polygons = load_annotations()
    train_df = index if args.fold < 0 else index[index["fold"] != args.fold]
    ds = FilamentDataset(train_df["image_id"], train_df["stem"], train=True, crop=args.crop, size=args.size,
                         spine=args.spine)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.workers > 0)
    validator = Validator(index, polygons, args.fold, args.size) if args.fold >= 0 else None

    model = build_model(args.arch, args.encoder, classes=2 if args.spine else 1).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(dl)
    warmup = len(dl)  # one epoch of linear warm-up, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))
    dtype = amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)  # loss scaling only for fp16
    bce = torch.nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=dtype):
                logits = model(x)
            logits = logits.float()
            loss = bce(logits[:, :1], y[:, :1]) + dice_loss(logits[:, :1], y[:, :1])
            if args.spine:  # auxiliary task: predict the filament center line
                loss = loss + args.spine_weight * (bce(logits[:, 1:2], y[:, 1:2]) + dice_loss(logits[:, 1:2], y[:, 1:2]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        row = dict(epoch=epoch, loss=float(np.mean(losses)), lr=sched.get_last_lr()[0], sec=time.time() - t0)
        if validator is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            row.update(validator.run(model))
            score = (row["dice"] + row["pq_pooled"]) / 2
            if score > best:
                best = score
                torch.save(model.state_dict(), run_dir / "best.pt")
                row["saved"] = "*"
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    torch.save(model.state_dict(), run_dir / "last.pt")
    if validator is None:  # full-data run: the final weights are the model
        torch.save(model.state_dict(), run_dir / "best.pt")


if __name__ == "__main__":
    main()
