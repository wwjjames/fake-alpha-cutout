"""Inference helpers shared by the local Stage A + Stage B MVP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from scripts.matting_core import LightUNet


class GridNetBaseline(nn.Module):
    """Stage A architecture for 400_3depth(baseline).pth."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def select_device() -> torch.device:
    return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model_files(root: Path, stage_a_path: Path, stage_b_path: Path) -> None:
    """Download missing release assets once, then verify and cache them locally."""
    manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    repository = manifest["repository"]
    destinations = {stage_a_path.resolve(), stage_b_path.resolve()}
    missing = [path for path in destinations if not path.is_file()]
    if not missing:
        return
    if repository.startswith("REPLACE_WITH_"):
        raise FileNotFoundError(
            "Model weights are missing. Configure the GitHub repository in model_manifest.json, "
            "then run: python scripts/download_models.py"
        )
    base_url = f"https://github.com/{repository}/releases/download/{manifest['release_tag']}"
    for spec in manifest["models"].values():
        destination = (root / spec["destination"]).resolve()
        if destination not in destinations or destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(
            f"{base_url}/{spec['asset']}", headers={"User-Agent": "cutout-lab-model-downloader"}
        )
        try:
            with urllib.request.urlopen(request) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if file_sha256(partial) != spec["sha256"]:
                raise RuntimeError(f"Downloaded {spec['asset']} did not match its published SHA-256.")
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise


def load_models(stage_a_path: Path, stage_b_path: Path, device: torch.device) -> tuple[GridNetBaseline, LightUNet, dict]:
    stage_a = GridNetBaseline()
    stage_a.load_state_dict(torch.load(stage_a_path, map_location="cpu", weights_only=True))
    stage_a.to(device).eval()
    checkpoint = torch.load(stage_b_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stage_b = LightUNet(config["base_channels"])
    stage_b.load_state_dict(checkpoint["model_state"])
    stage_b.to(device).eval()
    return stage_a, stage_b, config


def fake_probability(image: Image.Image, model: GridNetBaseline, device: torch.device) -> float:
    """Use Stage A's original Resize(256) + center-crop(224) preprocessing."""
    rgb = image.convert("RGB")
    scale = 256 / min(rgb.size)
    resized = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.BILINEAR)
    left, top = (resized.width - 224) // 2, (resized.height - 224) // 2
    crop = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(crop, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0, 1].item()


def predict_alpha(image: Image.Image, model: LightUNet, image_size: int, device: torch.device) -> Image.Image:
    """Predict at model resolution without cropping away any part of the source."""
    rgb = image.convert("RGB")
    original_size = rgb.size
    scale = image_size / max(original_size)
    resized_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    resized = rgb.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (127, 127, 127))
    left, top = (image_size - resized.width) // 2, (image_size - resized.height) // 2
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor)[0, 0].detach().cpu().numpy()
    alpha_canvas = Image.fromarray(np.round(np.clip(prediction, 0, 1) * 255).astype(np.uint8), "L")
    alpha = alpha_canvas.crop((left, top, left + resized.width, top + resized.height))
    return alpha.resize(original_size, Image.Resampling.LANCZOS)


def checkerboard_preview(rgba: Image.Image, tile_size: int = 32) -> Image.Image:
    """Composite an RGBA result over a neutral checkerboard for display."""
    background = Image.new("RGB", rgba.size, (224, 224, 224))
    draw = ImageDraw.Draw(background)
    for y in range(0, rgba.height, tile_size):
        for x in range(0, rgba.width, tile_size):
            if (x // tile_size + y // tile_size) % 2:
                draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), fill=(174, 174, 174))
    return Image.alpha_composite(background.convert("RGBA"), rgba).convert("RGB")


def render_rgba(image: Image.Image, alpha: Image.Image) -> Image.Image:
    result = image.convert("RGB").copy()
    result.putalpha(alpha)
    return result
