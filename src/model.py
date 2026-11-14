"""Model construction and (test-time-augmented) prediction."""
from __future__ import annotations

import json
from pathlib import Path

import segmentation_models_pytorch as smp
import torch


def build_model(arch: str = "Unet", encoder: str = "resnet34", pretrained: bool = True,
                classes: int = 1) -> torch.nn.Module:
    """U-Net family model with an ImageNet-pretrained encoder.

    ``in_channels=1``: the H-alpha images are grayscale; smp adapts the first
    pretrained convolution by summing its RGB filters.
    """
    cls = getattr(smp, arch)
    return cls(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,
        classes=classes,  # channel 0 = filament; optional channel 1 = spine (training aid)
    )


def load_trained(run_dir: Path, ckpt: str = "best.pt", device: str = "cuda") -> tuple[torch.nn.Module, int]:
    """Return the trained model and the resolution it was trained at."""
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["arch"], cfg["encoder"], pretrained=False, classes=2 if cfg.get("spine") else 1)
    model.load_state_dict(torch.load(run_dir / ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval().to(memory_format=torch.channels_last), int(cfg.get("size", 1024))


@torch.no_grad()
def predict_prob(model: torch.nn.Module, x: torch.Tensor, tta: bool = True) -> torch.Tensor:
    """Sigmoid probabilities for a batch ``x`` (Bx1xHxW, already on device).

    TTA averages the 4 flip variants (identity, h-flip, v-flip, both); each
    prediction is flipped back before averaging.
    """
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    out = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for dims in flips:
            xi = torch.flip(x, dims) if dims else x
            # only the filament channel is used; a spine channel (if any) is ignored
            p = torch.sigmoid(model(xi.contiguous(memory_format=torch.channels_last))[:, :1].float())
            out = out + (torch.flip(p, dims) if dims else p)
    return out / len(flips)
