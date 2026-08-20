#!/usr/bin/env python3
"""Generate RGB backgrounds that resemble AIGC pseudo-transparency checkerboards.

The output deliberately has no alpha channel.  It models the observed family
of mostly axis-aligned grey checkerboards: varied cell sizes and palettes,
subtle per-cell variation, gentle illumination gradients, occasional local
bright/dark regions, and light resampling/compression artefacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter


PaletteName = Literal["dark_gray", "light_gray_white", "medium_gray"]


PALETTES: dict[PaletteName, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "dark_gray": ((30, 30, 30), (88, 88, 88)),
    "light_gray_white": ((193, 193, 193), (242, 242, 242)),
    "medium_gray": ((83, 83, 83), (167, 167, 167)),
}


@dataclass(frozen=True)
class CheckerboardParams:
    seed: int
    palette: PaletteName
    tile_size: int
    cell_jitter: float
    gradient_strength: float
    local_spot_strength: float
    blur_radius: float
    rescale_factor: float
    jpeg_quality: int | None


def sample_params(seed: int, palette: PaletteName | None = None) -> CheckerboardParams:
    """Sample deterministic parameters matching the reference AIGC examples."""
    rng = np.random.default_rng(seed)
    selected_palette = palette or ("dark_gray", "light_gray_white", "medium_gray")[seed % 3]
    # The reference images primarily use axis-aligned squares from fine to
    # moderately broad; avoid rotation/perspective in this first generator.
    tile_size = int(rng.choice([8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]))
    return CheckerboardParams(
        seed=seed,
        palette=selected_palette,
        tile_size=tile_size,
        cell_jitter=float(rng.uniform(0.008, 0.055)),
        gradient_strength=float(rng.uniform(0.0, 0.10)),
        local_spot_strength=float(rng.uniform(0.0, 0.12)),
        blur_radius=float(rng.choice([0.0, 0.0, 0.15, 0.3, 0.5])),
        rescale_factor=float(rng.choice([1.0, 1.0, 1.0, 0.88, 0.94, 1.08, 1.14])),
        jpeg_quality=int(rng.choice([0, 0, 0, 91, 94, 96])) or None,
    )


def _apply_resampling(image: Image.Image, factor: float) -> Image.Image:
    if factor == 1.0:
        return image
    width, height = image.size
    resized = image.resize(
        (max(1, round(width * factor)), max(1, round(height * factor))),
        Image.Resampling.BILINEAR,
    )
    return resized.resize((width, height), Image.Resampling.BILINEAR)


def generate_checkerboard(
    size: tuple[int, int],
    seed: int,
    palette: PaletteName | None = None,
    params: CheckerboardParams | None = None,
) -> tuple[Image.Image, CheckerboardParams]:
    """Return an RGB pseudo-transparency background and its exact parameters."""
    if size[0] < 1 or size[1] < 1:
        raise ValueError("size dimensions must be positive")
    active = params or sample_params(seed, palette)
    if palette is not None and params is not None and palette != params.palette:
        raise ValueError("palette conflicts with params.palette")

    width, height = size
    rng = np.random.default_rng(active.seed)
    colour_a, colour_b = (np.array(colour, dtype=np.float32) for colour in PALETTES[active.palette])
    y_cells = height // active.tile_size + 2
    x_cells = width // active.tile_size + 2
    parity = (np.add.outer(np.arange(y_cells), np.arange(x_cells)) % 2).astype(bool)
    base_cells = np.where(parity[..., None], colour_a, colour_b)
    # AIGC/checker exports often vary a little from cell to cell rather than
    # using a mathematically perfect pair of constant swatches.
    variation = rng.normal(0, active.cell_jitter * 255, size=(y_cells, x_cells, 1))
    cells = np.clip(base_cells + variation, 0, 255)
    pixels = np.repeat(np.repeat(cells, active.tile_size, axis=0), active.tile_size, axis=1)[:height, :width]

    yy, xx = np.mgrid[0:height, 0:width]
    xx = (xx / max(width - 1, 1)) - 0.5
    yy = (yy / max(height - 1, 1)) - 0.5
    angle = rng.uniform(0, 2 * np.pi)
    gradient = (np.cos(angle) * xx + np.sin(angle) * yy) * active.gradient_strength * 255
    illumination = gradient
    # One or two soft local bright/dark regions capture the non-uniformity in
    # the supplied examples without turning the background into a scene.
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)
        radius = rng.uniform(0.18, 0.52)
        strength = rng.uniform(-1, 1) * active.local_spot_strength * 255
        illumination += strength * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)))
    pixels = np.clip(pixels + illumination[..., None], 0, 255).astype(np.uint8)

    image = Image.fromarray(pixels, mode="RGB")
    image = _apply_resampling(image, active.rescale_factor)
    if active.blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(active.blur_radius))
    if active.jpeg_quality is not None:
        encoded = BytesIO()
        image.save(encoded, format="JPEG", quality=active.jpeg_quality, subsampling=0)
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            image = decoded.convert("RGB")
    return image, active


def params_dict(params: CheckerboardParams) -> dict[str, object]:
    """CSV/JSON-friendly exact generation metadata."""
    return asdict(params)
