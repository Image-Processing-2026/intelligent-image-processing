"""Explicitly prepare and hash-check semantic segmentation assets.

No public segmentation call downloads anything. Populate ``assets.json`` with
exact URLs and SHA-256 values, then run this script before a real-model test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _prepare_one(entry: dict[str, Any], *, force: bool, timeout: int) -> dict[str, Any]:
    asset_id = str(entry.get("id", entry.get("path", "asset")))
    destination = _path(str(entry.get("path", "")))
    url = entry.get("url")
    expected = str(entry.get("sha256") or "").lower()
    if not destination.name:
        raise ValueError(f"asset {asset_id} has no path")
    if not isinstance(url, str) or not url:
        raise ValueError(f"asset {asset_id} has no explicit URL")
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"asset {asset_id} requires a 64-character hexadecimal sha256")
    if destination.is_file() and not force:
        actual = _sha256(destination)
        if actual != expected:
            raise RuntimeError(f"asset {asset_id} hash mismatch: expected {expected}, got {actual}")
        return {"id": asset_id, "path": str(destination), "sha256": actual, "status": "verified"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        actual = _sha256(temporary)
        if actual != expected:
            raise RuntimeError(f"asset {asset_id} hash mismatch: expected {expected}, got {actual}")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"id": asset_id, "path": str(destination), "sha256": expected, "status": "downloaded"}


def prepare(manifest_path: Path, *, force: bool = False, timeout: int = 60) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"{manifest_path} must contain a non-empty assets list")
    return [_prepare_one(asset, force=force, timeout=timeout) for asset in assets]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/prompt_segmentation/assets.json"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, force=args.force, timeout=args.timeout), indent=2))


if __name__ == "__main__":
    main()
