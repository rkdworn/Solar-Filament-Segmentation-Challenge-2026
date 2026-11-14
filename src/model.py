"""Model construction and (test-time-augmented) prediction."""
from __future__ import annotations

import json
from pathlib import Path

import segmentation_models_pytorch as smp
import torch


def amp_dtype() -> torch.dtype:
    """bfloat16 on Ampere or newer GPUs; float16 (with loss scaling in training) on older ones such as the T4."""
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


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


def tta_views(tta: bool | str) -> list[tuple[bool, list[int]]]:
    """(transpose?, flip dims) for each test-time view.

    "flip4" (or True): identity, h-flip, v-flip, both.
    "d4": the 4 flips of the image and of its transpose = all 8 rotations/reflections.
    """
    flips = [[], [3], [2], [2, 3]]
    if tta in (False, None, "none"):
        return [(False, [])]
    if tta in (True, "flip4"):
        return [(False, f) for f in flips]
    if tta == "d4":
        return [(t, f) for t in (False, True) for f in flips]
    raise ValueError(f"unknown TTA mode {tta!r}")


@torch.no_grad()
def predict_prob(model: torch.nn.Module, x: torch.Tensor, tta: bool | str = True) -> torch.Tensor:
    """Sigmoid probabilities for a batch ``x`` (Bx1xHxW, already on device).

    Each TTA view is transformed, predicted, transformed back, and averaged.
    """
    views = tta_views(tta)
    out = 0
    with torch.autocast("cuda", dtype=amp_dtype()):
        for transpose, dims in views:
            xi = x.transpose(2, 3) if transpose else x
            xi = torch.flip(xi, dims) if dims else xi
            # only the filament channel is used; a spine channel (if any) is ignored
            p = torch.sigmoid(model(xi.contiguous(memory_format=torch.channels_last))[:, :1].float())
            p = torch.flip(p, dims) if dims else p
            out = out + (p.transpose(2, 3) if transpose else p)
    return out / len(views)
