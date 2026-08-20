"""Shared components for the Stage B alpha-matting training and evaluation.

The dataset deliberately keeps train/validation/test discovery in the calling
script so training cannot accidentally touch the held-out test split.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def paired_paths(split_root: Path) -> list[tuple[Path, Path]]:
    """Return strictly same-name RGB/alpha pairs and reject malformed splits."""
    inputs = {path.name: path for path in (split_root / "input").glob("*.png")}
    alphas = {path.name: path for path in (split_root / "alpha").glob("*.png")}
    if not inputs or inputs.keys() != alphas.keys():
        missing_alpha = sorted(inputs.keys() - alphas.keys())[:5]
        missing_input = sorted(alphas.keys() - inputs.keys())[:5]
        raise RuntimeError(
            f"Invalid paired split {split_root}: {len(inputs)} inputs, {len(alphas)} alphas; "
            f"missing alpha={missing_alpha}, missing input={missing_input}"
        )
    return [(inputs[name], alphas[name]) for name in sorted(inputs)]


class MattingDataset(Dataset):
    """Paired RGB and alpha dataset with one synchronized geometric transform."""

    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int, train: bool) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def _geometry(self, image: Image.Image, alpha: Image.Image) -> tuple[Image.Image, Image.Image]:
        # Resize both images by the same factor so a square crop never pads.
        scale = self.image_size / min(image.width, image.height)
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        alpha = alpha.resize(new_size, Image.Resampling.BILINEAR)
        max_left = image.width - self.image_size
        max_top = image.height - self.image_size
        if self.train:
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)
        else:
            left, top = max_left // 2, max_top // 2
        box = (left, top, left + self.image_size, top + self.image_size)
        image, alpha = image.crop(box), alpha.crop(box)
        if self.train and random.random() < 0.5:
            image, alpha = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), alpha.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, alpha

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        input_path, alpha_path = self.pairs[index]
        with Image.open(input_path) as source:
            image = source.convert("RGB")
        with Image.open(alpha_path) as source:
            alpha = source.convert("L")
        image, alpha = self._geometry(image, alpha)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
        return {
            "image": torch.from_numpy(image_array.transpose(2, 0, 1)).contiguous(),
            "alpha": torch.from_numpy(alpha_array[None]).contiguous(),
            "name": input_path.name,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LightUNet(nn.Module):
    """Small four-level U-Net producing a continuous sigmoid alpha channel."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1, self.enc2 = ConvBlock(3, c), ConvBlock(c, c * 2)
        self.enc3, self.enc4 = ConvBlock(c * 2, c * 4), ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.dec4 = ConvBlock(c * 16 + c * 8, c * 8)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.head = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return torch.sigmoid(self.head(d1))


def boundary_mask(alpha: torch.Tensor, radius: int = 4, threshold: float = 1.0 / 255.0) -> torch.Tensor:
    """GT-derived transition band, dilated to make thin edges count materially."""
    kernel = 2 * radius + 1
    high = F.max_pool2d(alpha, kernel, stride=1, padding=radius)
    low = -F.max_pool2d(-alpha, kernel, stride=1, padding=radius)
    return (high - low > threshold).float()


def losses_and_sums(prediction: torch.Tensor, alpha: torch.Tensor, boundary_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return optimization loss plus additive sums for dataset-level MAE metrics."""
    error = (prediction - alpha).abs()
    boundary = boundary_mask(alpha)
    full_l1 = error.mean()
    boundary_l1 = (error * boundary).sum() / boundary.sum().clamp_min(1.0)
    soft = ((alpha > 0.0) & (alpha < 1.0)).float()
    loss = full_l1 + boundary_weight * boundary_l1
    return loss, {
        "error_sum": error.sum().detach(), "pixel_count": torch.tensor(error.numel(), device=error.device),
        "boundary_error_sum": (error * boundary).sum().detach(), "boundary_count": boundary.sum().detach(),
        "soft_error_sum": (error * soft).sum().detach(), "soft_count": soft.sum().detach(),
        "full_l1": full_l1.detach(), "boundary_l1": boundary_l1.detach(),
    }
