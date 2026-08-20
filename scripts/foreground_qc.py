#!/usr/bin/env python3
"""Inspect RGBA foreground PNGs and make consistent visual-QC previews.

This script intentionally does not modify source foreground images.  It scans
``training_assets/alpha_foregrounds``, composites each source over white,
black, and grey checkerboard backgrounds, and writes an auditable CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


MANIFEST_COLUMNS = [
    "path",
    "keep",
    "quality_note",
    "has_soft_alpha",
    "source_group",
    "width",
    "height",
    "image_mode",
    "is_rgba",
    "has_alpha_channel",
    "alpha_min",
    "alpha_max",
    "transparent_pixel_count",
    "transparent_pixel_ratio",
    "soft_alpha_pixel_count",
    "soft_alpha_pixel_ratio",
    "opaque_pixel_count",
    "auto_qc_status",
]


def alpha_stats(alpha: Image.Image) -> dict[str, int | float | bool]:
    histogram = alpha.histogram()
    total = alpha.width * alpha.height
    transparent = histogram[0]
    opaque = histogram[255]
    soft = sum(histogram[1:255])
    present = [value for value, count in enumerate(histogram) if count]
    return {
        "alpha_min": present[0],
        "alpha_max": present[-1],
        "transparent_pixel_count": transparent,
        "transparent_pixel_ratio": transparent / total,
        "soft_alpha_pixel_count": soft,
        "soft_alpha_pixel_ratio": soft / total,
        "opaque_pixel_count": opaque,
        "has_soft_alpha": bool(soft),
    }


def checkerboard(size: tuple[int, int], tile_size: int) -> Image.Image:
    image = Image.new("RGB", size, (238, 238, 238))
    draw = ImageDraw.Draw(image)
    alternate = (198, 198, 198)
    for top in range(0, size[1], tile_size):
        for left in range(0, size[0], tile_size):
            if (left // tile_size + top // tile_size) % 2:
                draw.rectangle(
                    (left, top, left + tile_size - 1, top + tile_size - 1),
                    fill=alternate,
                )
    return image


def backgrounds(size: tuple[int, int], tile_size: int) -> dict[str, Image.Image]:
    return {
        "white": Image.new("RGB", size, "white"),
        "black": Image.new("RGB", size, "black"),
        "checkerboard": checkerboard(size, tile_size),
    }


def save_previews(rgba: Image.Image, relative_path: Path, output_root: Path, tile_size: int) -> None:
    for name, background in backgrounds(rgba.size, tile_size).items():
        destination = output_root / name / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Avoid PNG's expensive optimization pass: these are review artefacts,
        # and fast, repeatable whole-corpus generation matters more than a
        # marginally smaller preview file.
        Image.alpha_composite(background.convert("RGBA"), rgba).convert("RGB").save(
            destination, format="PNG"
        )


def qc_row(
    source: Path, input_root: Path, output_root: Path, tile_size: int, skip_previews: bool
) -> dict[str, object]:
    relative_path = source.relative_to(input_root)
    base = {
        "path": (Path("training_assets") / "alpha_foregrounds" / relative_path).as_posix(),
        "source_group": relative_path.parts[0] if len(relative_path.parts) > 1 else "",
    }
    try:
        with Image.open(source) as opened:
            original_mode = opened.mode
            has_alpha_channel = "A" in opened.getbands() or "transparency" in opened.info
            rgba = opened.convert("RGBA")
        alpha = rgba.getchannel("A")
        stats = alpha_stats(alpha)
        issues: list[str] = []
        if original_mode != "RGBA":
            issues.append("source_mode_not_RGBA")
        if not has_alpha_channel:
            issues.append("missing_alpha_channel")
        if stats["transparent_pixel_count"] == 0:
            issues.append("no_fully_transparent_pixels")
        if stats["alpha_min"] == 255:
            issues.append("alpha_all_255")
        if not skip_previews:
            save_previews(rgba, relative_path, output_root, tile_size)
        status = "fail" if issues else "pass"
        return {
            **base,
            "keep": "no" if issues else "pending_review",
            "quality_note": ";".join(issues) if issues else "auto_check_passed; manual_visual_review_required",
            "width": rgba.width,
            "height": rgba.height,
            "image_mode": original_mode,
            "is_rgba": str(original_mode == "RGBA").lower(),
            "has_alpha_channel": str(has_alpha_channel).lower(),
            **stats,
            "has_soft_alpha": str(stats["has_soft_alpha"]).lower(),
            "auto_qc_status": status,
        }
    except (OSError, ValueError) as exc:
        return {
            **base,
            "keep": "no",
            "quality_note": f"unreadable_or_invalid_png: {exc}",
            "has_soft_alpha": "",
            "width": "",
            "height": "",
            "image_mode": "",
            "is_rgba": "false",
            "has_alpha_channel": "false",
            "alpha_min": "",
            "alpha_max": "",
            "transparent_pixel_count": "",
            "transparent_pixel_ratio": "",
            "soft_alpha_pixel_count": "",
            "soft_alpha_pixel_ratio": "",
            "opaque_pixel_count": "",
            "auto_qc_status": "fail",
        }


def load_existing_manifest(manifest: Path) -> dict[str, dict[str, str]]:
    """Load prior review decisions, if this is a subsequent QC pass."""
    if not manifest.is_file():
        return {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        return {row["path"]: row for row in csv.DictReader(handle) if row.get("path")}


def preserve_review(existing: dict[str, str] | None, fresh: dict[str, object]) -> dict[str, object]:
    """Keep human review fields on a passing file during later automatic scans."""
    if existing and fresh["auto_qc_status"] == "pass":
        fresh["keep"] = existing.get("keep", fresh["keep"])
        fresh["quality_note"] = existing.get("quality_note", fresh["quality_note"])
    return fresh


def png_files(input_root: Path) -> Iterable[Path]:
    return sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() == ".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("training_assets/alpha_foregrounds"))
    parser.add_argument("--preview-root", type=Path, default=Path("training_assets/foreground_qc_previews"))
    parser.add_argument("--manifest", type=Path, default=Path("training_assets/foreground_manifest.csv"))
    parser.add_argument("--checker-tile-size", type=int, default=32)
    parser.add_argument("--skip-previews", action="store_true", help="Only refresh the manifest; leave previews untouched.")
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="QC and make previews only for source files absent from the existing manifest; retain prior rows unchanged.",
    )
    args = parser.parse_args()
    if args.checker_tile_size < 1:
        parser.error("--checker-tile-size must be positive")
    if not args.input_root.is_dir():
        parser.error(f"input directory does not exist: {args.input_root}")

    existing_rows = load_existing_manifest(args.manifest)
    rows: list[dict[str, object]] = []
    scanned_count = 0
    for path in png_files(args.input_root):
        relative = path.relative_to(args.input_root)
        manifest_path = (Path("training_assets") / "alpha_foregrounds" / relative).as_posix()
        if args.only_new and manifest_path in existing_rows:
            rows.append(existing_rows[manifest_path])
            continue
        fresh = qc_row(path, args.input_root, args.preview_root, args.checker_tile_size, args.skip_previews)
        rows.append(preserve_review(existing_rows.get(manifest_path), fresh))
        scanned_count += 1
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row["auto_qc_status"] == "pass" for row in rows)
    print(f"QC scanned {scanned_count} PNG files; manifest contains {len(rows)} files: {passed} passed automatic alpha checks, {len(rows) - passed} failed.")
    print(f"Manifest: {args.manifest}")
    print(f"Previews: {args.preview_root}")


if __name__ == "__main__":
    main()
