#!/usr/bin/env python3
"""Download the versioned Stage A and Stage B checkpoints from a GitHub Release."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {target.name}…", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "cutout-lab-model-downloader"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual_sha256 = sha256(partial)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch for {target.name}: expected {expected_sha256}, got {actual_sha256}")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub OWNER/REPOSITORY. Overrides model_manifest.json.")
    parser.add_argument("--force", action="store_true", help="Re-download models even when the checksum matches.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    repository = args.repo or manifest["repository"]
    if repository.startswith("REPLACE_WITH_"):
        parser.error("Set 'repository' in model_manifest.json or pass --repo OWNER/REPOSITORY.")
    base_url = f"https://github.com/{repository}/releases/download/{manifest['release_tag']}"
    for name, spec in manifest["models"].items():
        destination = ROOT / spec["destination"]
        if destination.exists() and sha256(destination) == spec["sha256"] and not args.force:
            print(f"{name}: already present and verified.")
            continue
        download(f"{base_url}/{spec['asset']}", destination, spec["sha256"])
        print(f"{name}: downloaded and verified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
