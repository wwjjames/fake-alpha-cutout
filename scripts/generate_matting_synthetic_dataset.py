#!/usr/bin/env python3
"""Create paired RGB pseudo-transparency inputs and alpha labels for matting.

Each approved RGBA foreground is deterministically assigned to exactly one
split.  Its variants all remain in that split, preventing source-image leakage
between train, validation, and test sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image

from checkerboard_background import generate_checkerboard, params_dict


SPLITS = ("train", "val", "test")
METADATA_COLUMNS = [
    "sample_id",
    "source_path",
    "split",
    "variant",
    "source_width",
    "source_height",
    "output_width",
    "output_height",
    "has_soft_alpha",
    "input_path",
    "alpha_path",
    "seed",
    "palette",
    "tile_size",
    "cell_jitter",
    "gradient_strength",
    "local_spot_strength",
    "blur_radius",
    "rescale_factor",
    "jpeg_quality",
]


def stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def split_for_path(path: str) -> str:
    """Assign a foreground source to 80/10/10 splits, independent of run order."""
    bucket = stable_integer(path) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def output_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def read_approved_sources(manifest: Path, input_root: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    approved = [row for row in rows if row.get("keep", "").strip().lower() in {"yes", "keep", "true", "1"}]
    missing = [row["path"] for row in approved if not Path(row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest sources are missing; first: {missing[0]}")
    if not approved:
        raise ValueError("No approved foregrounds found; set keep=yes in the manifest.")
    return sorted(approved, key=lambda row: row["path"])


def sample_id(source_path: str, variant: int) -> str:
    source = Path(source_path)
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:10]
    return f"{source.parent.name}__{source.stem}__{digest}__v{variant:02d}"


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. Use --overwrite to replace this generated dataset."
            )
        shutil.rmtree(output_root)
    for split in SPLITS:
        (output_root / split / "input").mkdir(parents=True, exist_ok=True)
        (output_root / split / "alpha").mkdir(parents=True, exist_ok=True)


def generate(args: argparse.Namespace) -> list[dict[str, object]]:
    sources = read_approved_sources(args.manifest, args.input_root)
    if args.limit_sources is not None:
        sources = sources[: args.limit_sources]
    prepare_output_root(args.output_root, args.overwrite)
    rows: list[dict[str, object]] = []
    for index, source_row in enumerate(sources, start=1):
        source_path = Path(source_row["path"])
        split = split_for_path(source_row["path"])
        with Image.open(source_path) as source:
            source_rgba = source.convert("RGBA")
        source_width, source_height = source_rgba.size
        target_size = output_size(source_rgba.size, args.max_side)
        if target_size != source_rgba.size:
            source_rgba = source_rgba.resize(target_size, Image.Resampling.LANCZOS)
        alpha = source_rgba.getchannel("A")
        has_soft_alpha = str(any(alpha.histogram()[1:255])).lower()
        for variant in range(1, args.variants_per_source + 1):
            identifier = sample_id(source_row["path"], variant)
            seed = stable_integer(f"{args.seed}:{source_row['path']}:{variant}") % (2**63 - 1)
            background, params = generate_checkerboard(source_rgba.size, seed)
            composited = Image.alpha_composite(background.convert("RGBA"), source_rgba).convert("RGB")
            input_relative = Path(split) / "input" / f"{identifier}.png"
            alpha_relative = Path(split) / "alpha" / f"{identifier}.png"
            composited.save(args.output_root / input_relative, format="PNG")
            alpha.save(args.output_root / alpha_relative, format="PNG")
            rows.append(
                {
                    "sample_id": identifier,
                    "source_path": source_row["path"],
                    "split": split,
                    "variant": variant,
                    "source_width": source_width,
                    "source_height": source_height,
                    "output_width": source_rgba.width,
                    "output_height": source_rgba.height,
                    "has_soft_alpha": has_soft_alpha,
                    "input_path": input_relative.as_posix(),
                    "alpha_path": alpha_relative.as_posix(),
                    "seed": seed,
                    **params_dict(params),
                }
            )
        print(f"[{index}/{len(sources)}] {source_row['path']}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("training_assets/foreground_manifest.csv"))
    parser.add_argument("--input-root", type=Path, default=Path("training_assets/alpha_foregrounds"))
    parser.add_argument("--output-root", type=Path, default=Path("training_assets/matting_synthetic"))
    parser.add_argument("--variants-per-source", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=768, help="Downscale larger sources; never upscale smaller sources.")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--limit-sources", type=int, help="Generate only the first N approved sources; useful for a smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated output directory.")
    args = parser.parse_args()
    if args.variants_per_source < 1 or args.max_side < 1 or (args.limit_sources is not None and args.limit_sources < 1):
        parser.error("variants-per-source, max-side, and limit-sources must be positive")

    rows = generate(args)
    with (args.output_root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    print(f"Generated {len(rows)} paired samples: {split_counts}")
    print(f"Dataset: {args.output_root}")


if __name__ == "__main__":
    main()
