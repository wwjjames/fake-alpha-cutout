#!/usr/bin/env python3
"""Evaluate the original 3-layer pseudo-transparency classifier on a test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


CLASS_NAMES = ("0_ordinary", "1_fake")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class GridNetBaseline(nn.Module):
    """Architecture used by models/400_3depth(baseline).pth."""

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
        x = self.dropout(x)
        return self.fc2(x)


def preprocess(image: Image.Image) -> torch.Tensor:
    """Match torchvision Resize(256), CenterCrop(224), ToTensor, Normalize."""
    image = image.convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1))


def list_examples(test_root: Path) -> list[tuple[Path, int]]:
    examples: list[tuple[Path, int]] = []
    for label, name in enumerate(CLASS_NAMES):
        class_dir = test_root / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        examples.extend((path, label) for path in sorted(class_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS)
    return examples


def list_single_class_examples(root: Path, label: int) -> list[tuple[Path, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {root}")
    return [(path, label) for path in sorted(root.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]


def parse_benchmark_group(value: str) -> tuple[str, str, Path]:
    """Parse NAME:dataset|ordinary|fake:PATH benchmark specifications."""
    try:
        name, kind, raw_path = value.split(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected NAME:dataset|ordinary|fake:PATH"
        ) from error
    if not name or kind not in {"dataset", "ordinary", "fake"} or not raw_path:
        raise argparse.ArgumentTypeError("Expected NAME:dataset|ordinary|fake:PATH")
    return name, kind, Path(raw_path)


def calculate_metrics(matrix: np.ndarray) -> dict[str, object]:
    total = int(matrix.sum())
    return {
        "examples": total,
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "ordinary_false_positive_rate": float(matrix[0, 1] / max(1, matrix[0].sum())),
        "fake_false_negative_rate": float(matrix[1, 0] / max(1, matrix[1].sum())),
        "ordinary_recall": float(matrix[0, 0] / max(1, matrix[0].sum())),
        "fake_recall": float(matrix[1, 1] / max(1, matrix[1].sum())),
    }


def save_confusion_matrix(matrix: np.ndarray, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=(0, 1), yticks=(0, 1), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Test-set confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=Path("dataset/test_dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/400_3depth(baseline).pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline_test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--benchmark-group",
        action="append",
        type=parse_benchmark_group,
        metavar="NAME:KIND:PATH",
        help=(
            "Additional labelled benchmark. KIND is dataset (PATH contains 0_ordinary/1_fake), "
            "ordinary, or fake. May be repeated."
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    groups: list[tuple[str, list[tuple[Path, int]]]] = [("classifier_test", list_examples(args.test_root))]
    for name, kind, root in args.benchmark_group or []:
        examples = list_examples(root) if kind == "dataset" else list_single_class_examples(root, int(kind == "fake"))
        if not examples:
            parser.error(f"Benchmark group '{name}' contains no supported image files: {root}")
        groups.append((name, examples))
    model = GridNetBaseline()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, object]] = []
    matrix = np.zeros((2, 2), dtype=np.int64)
    group_matrices = {name: np.zeros((2, 2), dtype=np.int64) for name, _ in groups}
    with torch.no_grad():
        for group_name, examples in groups:
            for start in range(0, len(examples), args.batch_size):
                batch = examples[start : start + args.batch_size]
                tensors = torch.stack([preprocess(Image.open(path)) for path, _ in batch])
                probabilities = torch.softmax(model(tensors), dim=1).numpy()
                for (path, truth), probability in zip(batch, probabilities):
                    prediction = int(np.argmax(probability))
                    matrix[truth, prediction] += 1
                    group_matrices[group_name][truth, prediction] += 1
                    predictions.append(
                        {
                            "group": group_name,
                            "path": path.as_posix(),
                            "true_label": truth,
                            "true_class": CLASS_NAMES[truth],
                            "predicted_label": prediction,
                            "predicted_class": CLASS_NAMES[prediction],
                            "probability_ordinary": float(probability[0]),
                            "probability_fake": float(probability[1]),
                            "correct": str(prediction == truth).lower(),
                        }
                    )

    metrics = {
        "checkpoint": args.checkpoint.as_posix(),
        "test_root": args.test_root.as_posix(),
        "overall": calculate_metrics(matrix),
        "groups": {name: calculate_metrics(group_matrix) for name, group_matrix in group_matrices.items()},
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    save_confusion_matrix(matrix, args.output_dir / "confusion_matrix.png")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Reports: {args.output_dir}")


if __name__ == "__main__":
    main()
