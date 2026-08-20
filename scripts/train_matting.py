"""Train the Stage B continuous alpha-matting model using train/val only."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from matting_core import LightUNet, MattingDataset, losses_and_sums, paired_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("training_assets/matting_synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("matting_models/stage_b_alpha_v1"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto", help="auto prefers Apple Metal (MPS) when available.")
    parser.add_argument("--log-interval", type=int, default=50, help="Print progress every N batches; 0 prints only epoch summaries.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--limit-train", type=int, default=0, help="For smoke testing only; 0 uses every train pair.")
    parser.add_argument("--limit-val", type=int, default=0, help="For smoke testing only; 0 uses every val pair.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "mps" and not mps_available:
        raise RuntimeError("--device mps was requested, but MPS is not available in this Python/macOS session.")
    return torch.device("mps" if requested == "mps" or (requested == "auto" and mps_available) else "cpu")


def synchronize(device: torch.device) -> None:
    # MPS is asynchronous; synchronize only at reporting boundaries for a real ETA.
    if device.type == "mps": torch.mps.synchronize()


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds)); hours, seconds = divmod(seconds, 3600); minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def aggregate(loader: DataLoader, model: torch.nn.Module, optimizer: AdamW | None, boundary_weight: float, device: torch.device, phase: str, epoch: int, total_epochs: int, log_interval: int) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {key: torch.zeros((), device=device) for key in ("loss_sum", "error_sum", "pixel_count", "boundary_error_sum", "boundary_count", "soft_error_sum", "soft_count")}
    batch_count, seen, phase_start = 0, 0, time.perf_counter()
    total_batches, total_samples = len(loader), len(loader.dataset)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=device.type == "mps")
            alpha = batch["alpha"].to(device, non_blocking=device.type == "mps")
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            loss, values = losses_and_sums(prediction, alpha, boundary_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            sums["loss_sum"] += loss.detach()
            batch_count += 1; seen += image.shape[0]
            for key in ("error_sum", "pixel_count", "boundary_error_sum", "boundary_count", "soft_error_sum", "soft_count"):
                sums[key] += values[key]
            if log_interval and (batch_index % log_interval == 0 or batch_index == total_batches):
                synchronize(device)
                elapsed = time.perf_counter() - phase_start
                batches_per_second = batch_index / elapsed
                remaining = (total_batches - batch_index) / batches_per_second
                print(f"{phase} epoch {epoch:03d}/{total_epochs} | batch {batch_index:04d}/{total_batches} | samples {seen}/{total_samples} ({seen / total_samples:.1%}) | {batches_per_second:.2f} batch/s | ETA {format_duration(remaining)}", flush=True)
    synchronize(device)
    return {
        "loss": (sums["loss_sum"] / max(batch_count, 1)).item(),
        "mae": (sums["error_sum"] / sums["pixel_count"].clamp_min(1.0)).item(),
        "boundary_mae": (sums["boundary_error_sum"] / sums["boundary_count"].clamp_min(1.0)).item(),
        "soft_alpha_mae": (sums["soft_error_sum"] / sums["soft_count"].clamp_min(1.0)).item(),
    }


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 5))
    for key, label in (("train_loss", "train loss"), ("val_loss", "val loss"), ("val_mae", "val full MAE"), ("val_boundary_mae", "val boundary MAE"), ("val_soft_alpha_mae", "val soft-alpha MAE")):
        plt.plot(epochs, [row[key] for row in history], label=label)
    plt.xlabel("epoch"); plt.ylabel("error (0–1)"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()


def save_validation_examples(dataset: MattingDataset, model: torch.nn.Module, directory: Path, device: torch.device) -> None:
    """Write the same hard-ish validation samples every epoch for visual tracking."""
    desired = ("hair", "bicycle", "mesh", "tile", "chess", "translucent", "glass", "flowing")
    selected: list[int] = []
    used_sources: set[str] = set()
    for term in desired:
        for i, (_, alpha) in enumerate(dataset.pairs):
            # Drop the final variant suffix to prevent one foreground filling the panel.
            source = alpha.stem.rsplit("__v", 1)[0]
            if term in alpha.name.lower() and source not in used_sources:
                selected.append(i); used_sources.add(source); break
    if len(selected) < 6:
        for i, (_, alpha) in enumerate(dataset.pairs):
            source = alpha.stem.rsplit("__v", 1)[0]
            if source not in used_sources:
                selected.append(i); used_sources.add(source)
            if len(selected) == 6: break
    model.eval()
    with torch.no_grad():
        for rank, index in enumerate(selected[:6], 1):
            sample = dataset[index]; image, alpha = sample["image"], sample["alpha"]
            prediction = model(image.unsqueeze(0).to(device))[0].cpu()
            error = (prediction - alpha).abs()
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(image.permute(1, 2, 0).numpy()); axes[0].set_title("input RGB")
            for axis, array, title, cmap in ((axes[1], alpha[0].numpy(), "GT alpha", "gray"), (axes[2], prediction[0].numpy(), "predicted alpha", "gray"), (axes[3], error[0].numpy(), "absolute error", "magma")):
                axis.imshow(array, cmap=cmap, vmin=0, vmax=1); axis.set_title(title)
            for axis in axes: axis.axis("off")
            fig.tight_layout(); fig.savefig(directory / f"{rank:02d}_{sample['name']}", dpi=140, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    if args.epochs < 1 or args.batch_size < 1: raise ValueError("epochs and batch size must be positive")
    if args.log_interval < 0: raise ValueError("log interval cannot be negative")
    device = select_device(args.device)
    # Intentionally discover only train/val: test is reserved for evaluate_matting.py.
    train_pairs, val_pairs = paired_paths(args.data_root / "train"), paired_paths(args.data_root / "val")
    if args.limit_train: train_pairs = train_pairs[:args.limit_train]
    if args.limit_val: val_pairs = val_pairs[:args.limit_val]
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"created_utc": datetime.now(timezone.utc).isoformat(), "train_pairs": len(train_pairs), "val_pairs": len(val_pairs), "architecture": "LightUNet", "resolved_device": device.type}
    config["data_root"] = str(args.data_root); config["output_dir"] = str(output)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    train_loader = DataLoader(MattingDataset(train_pairs, args.image_size, train=True), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = MattingDataset(val_pairs, args.image_size, train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    validation_examples = output / "validation_examples"; validation_examples.mkdir(exist_ok=True)
    model = LightUNet(args.base_channels).to(device); optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []; best_mae = float("inf")
    print(f"Training on {device.type.upper()}: {len(train_pairs)} train / {len(val_pairs)} val pairs; test split is not opened.", flush=True)
    for epoch in range(1, args.epochs + 1):
        train = aggregate(train_loader, model, optimizer, args.boundary_weight, device, "train", epoch, args.epochs, args.log_interval)
        val = aggregate(val_loader, model, None, args.boundary_weight, device, "val", epoch, args.epochs, args.log_interval)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in val.items()}}
        history.append(row)
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_val_mae": min(best_mae, val["mae"]), "config": config}
        torch.save(state, output / "last.pt")
        if val["mae"] < best_mae:
            best_mae = val["mae"]; torch.save(state, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        plot_history(history, output / "training_curve.png")
        save_validation_examples(val_dataset, model, validation_examples, device)
        print(f"epoch {epoch:03d}/{args.epochs}: train_loss={train['loss']:.5f} val_loss={val['loss']:.5f} val_mae={val['mae']:.5f} boundary={val['boundary_mae']:.5f} soft={val['soft_alpha_mae']:.5f}{'  [best]' if val['mae'] == best_mae else ''}", flush=True)
    print(f"Finished. best val MAE={best_mae:.6f}; artifacts: {output}", flush=True)


if __name__ == "__main__": main()
