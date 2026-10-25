"""Tải và kiểm tra model phân vùng ngữ nghĩa đã ghim (GroundingDINO + MobileSAM).

Đọc ``tests/fixtures/prompt_segmentation/assets.json``: GroundingDINO lấy theo revision Hugging Face
đã ghim, MobileSAM lấy đúng URL và sha256. Mọi file đều được kiểm sha256 trước khi đặt vào
``models/segmentation/``. Không có API công khai nào tải model khi chạy suy luận.

Chạy từ thư mục gốc repo:
    python scripts/fetch_segmentation_models.py [--force] [--timeout 120]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "prompt_segmentation" / "assets.json"
SEGMENTATION_ROOT = REPO_ROOT / "models" / "segmentation"
_LOGGER = logging.getLogger("fetch_segmentation_models")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")


def _load_assets(manifest: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    return {entry["id"]: entry for entry in payload["assets"]}


def _fetch_snapshot(entry: dict[str, Any], *, force: bool) -> None:
    """Tải snapshot Hugging Face theo revision ghim rồi kiểm sha256 từng file đã khai báo."""
    destination = REPO_ROOT / entry["path"]
    expected_files: dict[str, str] = entry["files"]
    if not force and all((destination / name).is_file() for name in expected_files):
        for name, digest in expected_files.items():
            _verify(destination / name, digest)
        _LOGGER.info("snapshot already present and verified: %s", destination)
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - phụ thuộc tùy chọn
        raise SystemExit(
            "huggingface_hub is required; install requirements-semantic.txt first"
        ) from exc
    snapshot_download(
        repo_id=entry["repo_id"],
        revision=entry["revision"],
        local_dir=destination,
        allow_patterns=list(entry["required_files"]),
    )
    for name, digest in expected_files.items():
        _verify(destination / name, digest)
    _LOGGER.info("snapshot downloaded and verified: %s", destination)


def _fetch_file(entry: dict[str, Any], *, force: bool, timeout: int) -> None:
    """Tải một file (URL ghim) vào ``.part`` rồi đổi tên sau khi sha256 khớp."""
    destination = REPO_ROOT / entry["path"]
    expected = entry["sha256"]
    if destination.is_file() and not force:
        if _sha256(destination) == expected:
            _LOGGER.info("file already present and verified: %s", destination)
            return
        _LOGGER.warning("existing file has wrong sha256, downloading again: %s", destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(entry["url"], timeout=timeout) as response:
            with temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
        _verify(temporary, expected)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    _LOGGER.info("file downloaded and verified: %s", destination)


def fetch(manifest: Path, *, force: bool = False, timeout: int = 120) -> None:
    assets = _load_assets(manifest)
    _fetch_snapshot(assets["grounding_dino_tiny_snapshot"], force=force)
    _fetch_file(assets["mobile_sam_vit_t"], force=force, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="tải lại ngay cả khi file đã có")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    SEGMENTATION_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        fetch(args.manifest, force=args.force, timeout=args.timeout)
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        _LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
