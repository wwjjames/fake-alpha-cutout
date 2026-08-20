#!/usr/bin/env python3
"""Create a compact, reproducible visual QA set for Step 2 backgrounds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from checkerboard_background import PALETTES, generate_checkerboard, params_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("training_assets/checkerboard_backgrounds_preview"))
    parser.add_argument("--count-per-palette", type=int, default=16)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.count_per_palette < 1 or args.width < 1 or args.height < 1:
        parser.error("count and dimensions must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for palette_index, palette in enumerate(PALETTES):
        for variant in range(1, args.count_per_palette + 1):
            seed = args.seed + palette_index * 10_000 + variant
            image, params = generate_checkerboard((args.width, args.height), seed, palette=palette)
            relative_path = Path(palette) / f"{palette}_{variant:02d}.png"
            destination = args.output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG")
            rows.append({"path": relative_path.as_posix(), "width": args.width, "height": args.height, **params_dict(params)})

    fieldnames = ["path", "width", "height", "seed", "palette", "tile_size", "cell_jitter", "gradient_strength", "local_spot_strength", "blur_radius", "rescale_factor", "jpeg_quality"]
    with (args.output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} RGB preview backgrounds in {args.output_dir}")


if __name__ == "__main__":
    main()
