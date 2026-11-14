"""PyTorch dataset reading the cached images and masks at a given resolution."""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import WORK_SIZE, img_cache_dir, mask_cache_dir, normalize_disk, soft_cache_dir, spine_cache_dir


def photometric_aug() -> A.Compose:
    """Applied to the uint8 image: mimics differences between GONG sites."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(80, 125), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
    ])


def geometric_aug(crop: int | None, mask_interpolation: int = cv2.INTER_NEAREST) -> A.Compose:
    """Applied after normalization. The solar disk has no preferred
    orientation, so arbitrary rotations and flips are label-preserving.

    The image is resampled bilinearly; targets use ``mask_interpolation``
    (nearest neighbour keeps binary masks binary).
    """
    ops = [
        A.Affine(rotate=(-180, 180), scale=(0.9, 1.1), translate_percent=(-0.03, 0.03),
                 border_mode=cv2.BORDER_CONSTANT, fill=-6.0, fill_mask=0, mask_interpolation=mask_interpolation,
                 p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if crop:
        ops.append(A.RandomCrop(crop, crop))
    return A.Compose(ops)


class FilamentDataset(Dataset):
    """Training samples (image + target) read from the cache.

    Two kinds of targets:

    * hard (default): one sample = one annotation set (an observation + one
      annotator's union mask). Observations annotated by several people appear
      several times, each time with that annotator's labels; this acts as
      label-noise regularization. ``image_ids`` name the annotation sets.
    * soft (``soft=True``): one sample = one observation; the target is the
      per-pixel mean of all its annotators' masks (0, 1/3, 1/2, 2/3 or 1, see
      ``prepare.cache_soft_targets``). ``stems`` must be unique and
      ``image_ids`` is ignored (may be None).
    """

    def __init__(self, image_ids, stems, train: bool, crop: int | None = None, size: int = WORK_SIZE,
                 spine: bool = False, soft: bool = False):
        if soft and spine:
            raise ValueError("the spine target is per annotation set and cannot be combined with soft targets")
        self.stems = list(stems)
        self.soft = soft
        # File name of each sample's target in the cache: observation (soft) or annotation set (hard).
        self.target_names = self.stems if soft else list(image_ids)
        self.img_dir = img_cache_dir(size)
        self.target_dirs = [soft_cache_dir(size) if soft else mask_cache_dir(size)]
        if spine:  # optional 2nd target channel: the filament center line
            self.target_dirs.append(spine_cache_dir(size))
        if soft and not self.target_dirs[0].is_dir():
            raise FileNotFoundError(f"{self.target_dirs[0]} missing: run prepare.py --soft-only --size {size}")
        self.photo = photometric_aug() if train else None
        # Hard masks: nearest neighbour, they stay exactly 0/1. Soft targets go
        # through the warp as uint8 0-255 with *bilinear* interpolation, like
        # the image: an interpolated value is a convex combination of
        # neighbouring targets, and OpenCV saturates uint8 results, so after
        # dividing by 255 every target value lies in [0, 1]. The constant
        # border (outside the rotated image) is filled with 0 = background.
        self.geo = geometric_aug(crop, cv2.INTER_LINEAR if soft else cv2.INTER_NEAREST) if train else None

    def __len__(self) -> int:
        return len(self.stems)

    def _read_targets(self, i: int) -> list[np.ndarray]:
        """Target channels as uint8: 0/1 for hard masks, 0-255 for soft targets."""
        name = self.target_names[i]
        targets = [cv2.imread(str(d / f"{name}.png"), cv2.IMREAD_GRAYSCALE) for d in self.target_dirs]
        if self.soft:
            return targets  # scaled to [0, 1] after the geometric augmentation
        return [(t > 127).astype(np.uint8) for t in targets]

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.img_dir / f"{self.stems[i]}.png"), cv2.IMREAD_GRAYSCALE)
        masks = self._read_targets(i)
        if self.photo is not None:
            img = self.photo(image=img)["image"]
        x = normalize_disk(img)  # statistics from the whole disk, before cropping
        if self.geo is not None:
            out = self.geo(image=x, masks=masks)
            x, masks = out["image"], out["masks"]
        y = np.stack(masks).astype(np.float32)
        if self.soft:
            y /= 255.0
        return torch.from_numpy(np.ascontiguousarray(x)[None]), torch.from_numpy(y)


def load_image_tensor(stem: str, size: int = WORK_SIZE) -> torch.Tensor:
    """Normalized whole image for inference, shape 1x1xSxS."""
    img = cv2.imread(str(img_cache_dir(size) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(normalize_disk(img))[None, None]
