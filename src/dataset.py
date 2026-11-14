"""PyTorch dataset reading the cached images and masks at a given resolution."""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import WORK_SIZE, img_cache_dir, mask_cache_dir, normalize_disk


def photometric_aug() -> A.Compose:
    """Applied to the uint8 image: mimics differences between GONG sites."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(80, 125), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
    ])


def geometric_aug(crop: int | None) -> A.Compose:
    """Applied after normalization. The solar disk has no preferred
    orientation, so arbitrary rotations and flips are label-preserving."""
    ops = [
        A.Affine(rotate=(-180, 180), scale=(0.9, 1.1), translate_percent=(-0.03, 0.03),
                 border_mode=cv2.BORDER_CONSTANT, fill=-6.0, fill_mask=0, p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if crop:
        ops.append(A.RandomCrop(crop, crop))
    return A.Compose(ops)


class FilamentDataset(Dataset):
    """One sample = one annotation set (an observation + one annotator's masks).

    Observations annotated by several people appear several times, each time
    with that annotator's labels; this acts as label-noise regularization.
    """

    def __init__(self, image_ids, stems, train: bool, crop: int | None = None, size: int = WORK_SIZE):
        self.image_ids = list(image_ids)
        self.stems = list(stems)
        self.img_dir, self.mask_dir = img_cache_dir(size), mask_cache_dir(size)
        self.photo = photometric_aug() if train else None
        self.geo = geometric_aug(crop) if train else None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.img_dir / f"{self.stems[i]}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_dir / f"{self.image_ids[i]}.png"), cv2.IMREAD_GRAYSCALE) > 127
        mask = mask.astype(np.uint8)
        if self.photo is not None:
            img = self.photo(image=img)["image"]
        x = normalize_disk(img)  # statistics from the whole disk, before cropping
        if self.geo is not None:
            out = self.geo(image=x, mask=mask)
            x, mask = out["image"], out["mask"]
        return torch.from_numpy(np.ascontiguousarray(x)[None]), torch.from_numpy(mask[None].astype(np.float32))


def load_image_tensor(stem: str, size: int = WORK_SIZE) -> torch.Tensor:
    """Normalized whole image for inference, shape 1x1xSxS."""
    img = cv2.imread(str(img_cache_dir(size) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(normalize_disk(img))[None, None]
