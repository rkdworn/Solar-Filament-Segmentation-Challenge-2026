"""Train a filament segmentation model on one CV fold (or on all data).

Example:
    .venv\\Scripts\\python.exe src\\train.py --name r34_f0 --encoder resnet34 --fold 0
    .venv\\Scripts\\python.exe src\\train.py --name r34_full --encoder resnet34 --fold -1
    # soft (annotator-mean) targets + weight EMA; needs `prepare.py --soft-only` once
    .venv\\Scripts\\python.exe src\\train.py --name r34s_f0 --fold 0 --soft --epochs 65 --ema

Training targets (loss BCE + Dice in both cases, valid for soft targets, see
``dice_loss_fn``):
  * default: one sample per annotation set, target = that annotator's mask;
  * ``--soft``: one sample per observation, target = per-pixel mean of all its
    annotators' masks. There are ~565 instead of ~920 training samples per
    fold, so use ~65 instead of 40 epochs to keep the number of optimizer steps.

Validation (every epoch) runs on whole 1024x1024 images of the held-out
observations, separately for each annotation set (in both modes), and reports
  * Dice  : mean per annotation set (union masks), and
  * PQ    : Panoptic Quality after default post-processing.
The checkpoint with the best (Dice + PQ) / 2 (``--select combo``, default) or
the best pooled PQ (``--select pq``) is kept as ``best.pt``. With ``--ema``
the exponential moving average of the weights is validated and saved instead
of the raw weights.
"""
from __future__ import annotations

import argparse
import copy
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
    ap.add_argument("--attention", default=None, choices=["scse"], help="decoder attention blocks (default: none)")
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
    ap.add_argument("--soft", action="store_true",
                    help="one sample per observation with the annotator-mean mask as target "
                         "(needs prepare.py --soft-only); use ~65 epochs instead of 40")
    ap.add_argument("--ema", action="store_true",
                    help="validate and save an exponential moving average of the weights")
    ap.add_argument("--ema-decay", type=float, default=0.998, help="EMA decay per optimizer step")
    ap.add_argument("--select", choices=["combo", "pq"], default="combo",
                    help="best.pt criterion: (Dice + pooled PQ) / 2 (combo) or pooled PQ (pq)")
    args = ap.parse_args()
    if args.soft and args.spine:
        ap.error("--spine targets are per annotation set and cannot be combined with --soft")
    return args


class ModelEMA:
    """Exponential moving average (EMA) of the model weights.

    After every optimizer step: ``ema <- d * ema + (1 - d) * weights``.
    Floating-point buffers (BatchNorm running statistics) are averaged the
    same way; integer buffers (BatchNorm batch counters) are copied.

    The decay is bias-corrected as in Adam: after n updates
    ``d_n = decay * (1 - decay^(n-1)) / (1 - decay^n)``, so the average weights
    the iterates w_1..w_n in proportion to decay^(n-k) and gives no weight to
    the initial (partly random) weights. ``d_n`` tends to ``decay`` within a
    few multiples of ``1 / (1 - decay)`` steps (500 steps for 0.998).
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - self.decay ** (self.updates - 1)) / (1 - self.decay ** self.updates)
        for ema_t, t in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_t.is_floating_point():
                ema_t.lerp_(t, 1.0 - d)
            else:
                ema_t.copy_(t)


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
            # one entry per annotation set (each annotator of the observation)
            for image_id in self.sets[stem]:
                dices.append(dice(prob > self.post.threshold, self._union(image_id)))
                stats.append(pq_stats(labels, self.gt_inst[image_id]))
        res = summarize(stats)
        res["dice"] = float(np.mean(dices))
        return res


def dice_loss_fn(soft: bool) -> torch.nn.Module:
    """Binary Dice loss on logits, also valid for soft (fractional) targets.

    The classic soft Dice loss ``1 - 2<p, t> / (|p| + |t|)`` is not minimized
    at p = t when t is fractional (e.g. 1/3 = one of three annotators): it
    pushes p towards 0 or 1, against the BCE term. smp (pinned 0.6.0) instead
    computes the intersection as ``(|p| + |t| - |p - t|) / 2``, which gives the
    Dice semimetric loss ``|p - t|_1 / (|p|_1 + |t|_1)`` of Wang & Blaschko
    (MICCAI 2023, arXiv 2303.16296): identical to the classic loss for hard
    targets (so the default training is unchanged) and zero only at p = t.
    For ``soft`` training we check this property, because older smp versions
    implement the classic form.
    """
    loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    if soft:
        t = torch.tensor([0.0, 1 / 3, 1 / 2, 2 / 3, 1.0]).view(1, 1, -1)
        if loss(torch.logit(t, eps=1e-6), t).item() > 1e-4:
            raise RuntimeError("smp.losses.DiceLoss is not minimized at p = target for soft targets; "
                               "install segmentation-models-pytorch==0.6.0 (requirements.txt)")
    return loss


def selection_score(res: dict, select: str) -> float:
    """Validation score used to pick ``best.pt`` (higher is better)."""
    if select == "pq":
        return res["pq_pooled"]
    return (res["dice"] + res["pq_pooled"]) / 2


def build_train_set(train_df: pd.DataFrame, args: argparse.Namespace) -> FilamentDataset:
    """Annotation sets (default) or observations with annotator-mean targets (``--soft``)."""
    if args.soft:
        return FilamentDataset(None, sorted(train_df["stem"].unique()), train=True, crop=args.crop,
                               size=args.size, soft=True)
    return FilamentDataset(train_df["image_id"], train_df["stem"], train=True, crop=args.crop, size=args.size,
                           spine=args.spine)


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
    ds = build_train_set(train_df, args)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.workers > 0)
    validator = Validator(index, polygons, args.fold, args.size) if args.fold >= 0 else None

    model = build_model(args.arch, args.encoder, classes=2 if args.spine else 1,
                        attention=args.attention).cuda().to(memory_format=torch.channels_last)
    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    eval_model = ema.module if ema is not None else model  # the weights that are validated and saved
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(dl)
    warmup = len(dl)  # one epoch of linear warm-up, then cosine decay
    decay_steps = max(1, total - warmup)  # avoids 0/0 when --epochs 1 (smoke tests)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / decay_steps)))
    dtype = amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)  # loss scaling only for fp16
    bce = torch.nn.BCEWithLogitsLoss()  # also valid for soft targets in [0, 1]: minimized at p = target
    dice_loss = dice_loss_fn(args.soft)
    print(f"{len(ds)} training samples ({'observations, soft targets' if args.soft else 'annotation sets'}), "
          f"{len(dl)} steps/epoch, {total} steps", flush=True)

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
            if ema is not None:
                ema.update(model)
            losses.append(loss.item())

        row = dict(epoch=epoch, loss=float(np.mean(losses)), lr=sched.get_last_lr()[0], sec=time.time() - t0)
        if validator is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            row.update(validator.run(eval_model))
            score = selection_score(row, args.select)
            if score > best:
                best = score
                torch.save(eval_model.state_dict(), run_dir / "best.pt")
                row["saved"] = "*"
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    torch.save(eval_model.state_dict(), run_dir / "last.pt")
    if validator is None:  # full-data run: the final weights are the model
        torch.save(eval_model.state_dict(), run_dir / "best.pt")


if __name__ == "__main__":
    main()
