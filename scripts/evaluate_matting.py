"""One-time held-out Stage B test evaluation and qualitative visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from matting_core import LightUNet, MattingDataset, losses_and_sums, paired_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("training_assets/matting_synthetic"))
    parser.add_argument("--checkpoint", type=Path, default=Path("matting_models/stage_b_alpha_v1/best.pt"))
    parser.add_argument("--report-dir", type=Path, default=Path("matting_reports"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args()


def save_figure(image: torch.Tensor, alpha: torch.Tensor, prediction: torch.Tensor, path: Path) -> None:
    error = (prediction - alpha).abs()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image.permute(1, 2, 0).numpy()); axes[0].set_title("input RGB")
    for axis, array, title, cmap, vmax in ((axes[1], alpha[0].numpy(), "GT alpha", "gray", 1), (axes[2], prediction[0].numpy(), "predicted alpha", "gray", 1), (axes[3], error[0].numpy(), "absolute error", "magma", 1)):
        axis.imshow(array, cmap=cmap, vmin=0, vmax=vmax); axis.set_title(title)
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = parse_args(); report = args.report_dir; visuals = report / "test_examples"; visuals.mkdir(parents=True, exist_ok=True)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if args.device == "mps" and not mps_available:
        raise RuntimeError("--device mps was requested, but MPS is not available in this Python/macOS session.")
    device = torch.device("mps" if args.device == "mps" or (args.device == "auto" and mps_available) else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]; model = LightUNet(config["base_channels"]); model.load_state_dict(state["model_state"]); model.to(device).eval()
    pairs = paired_paths(args.data_root / "test")
    dataset = MattingDataset(pairs, config["image_size"], train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    totals = {key: 0.0 for key in ("error_sum", "pixel_count", "boundary_error_sum", "boundary_count", "soft_error_sum", "soft_count")}
    # Pick high-error examples after evaluating every held-out sample (not cherry-picked filenames).
    candidates: list[tuple[float, str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for batch in loader:
            image, alpha = batch["image"].to(device), batch["alpha"].to(device)
            prediction = model(image); _, values = losses_and_sums(prediction, alpha, config["boundary_weight"])
            for key in totals: totals[key] += float(values[key])
            errors = (prediction - alpha).abs().mean(dim=(1, 2, 3))
            for i, score in enumerate(errors.tolist()): candidates.append((score, batch["name"][i], image[i].cpu(), alpha[i].cpu(), prediction[i].cpu()))
    metrics = {"split": "test", "checkpoint": str(args.checkpoint), "device": device.type, "test_pairs": len(pairs), "full_alpha_mae": totals["error_sum"] / totals["pixel_count"], "boundary_mae": totals["boundary_error_sum"] / max(totals["boundary_count"], 1), "soft_alpha_mae": totals["soft_error_sum"] / max(totals["soft_count"], 1), "boundary_pixels": int(totals["boundary_count"]), "soft_alpha_pixels": int(totals["soft_count"])}
    (report / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for rank, (_, name, image, alpha, prediction) in enumerate(sorted(candidates, reverse=True)[:args.samples], 1): save_figure(image, alpha, prediction, visuals / f"{rank:02d}_{name}")
    print(json.dumps(metrics, indent=2)); print(f"Saved {min(args.samples, len(candidates))} highest-error held-out test examples to {visuals}")


if __name__ == "__main__": main()
