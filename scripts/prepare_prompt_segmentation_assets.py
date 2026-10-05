"""Explicitly prepare and hash-check semantic segmentation assets.

No public segmentation call downloads anything. Populate ``assets.json`` with
exact URLs and SHA-256 values, then run this script before a real-model test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
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


def _validate_hash(expected: str, asset_id: str) -> None:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"asset {asset_id} requires a 64-character hexadecimal sha256")


def _download(url: str, parent: Path, name: str, timeout: int) -> Path:
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{name}.", suffix=".part", dir=parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        return temporary
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _safe_extract(archive: Path, destination: Path, archive_format: str) -> None:
    root = destination.resolve()
    if archive_format == "zip":
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            for member in members:
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"archive member escapes destination: {member.filename}")
            handle.extractall(destination)
        return
    if archive_format in {"tar", "tar.gz", "tgz", "tar.xz", "tar.bz2"}:
        with tarfile.open(archive, mode="r:*") as handle:
            members = handle.getmembers()
            for member in members:
                target = (destination / member.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"archive member escapes destination: {member.name}")
            if sys.version_info >= (3, 12):
                handle.extractall(destination, filter="data")
            else:  # The member path checks above protect Python 3.10/3.11.
                handle.extractall(destination)
        return
    raise ValueError(f"unsupported archive format: {archive_format}")


def _verify_required_files(destination: Path, required_files: list[str], asset_id: str) -> None:
    missing = [str(destination / relative) for relative in required_files if not (destination / relative).is_file()]
    if missing:
        raise RuntimeError(f"asset {asset_id} is missing required files: {', '.join(missing)}")


def _prepare_archive(entry: dict[str, Any], *, force: bool, timeout: int) -> dict[str, Any]:
    asset_id = str(entry.get("id", entry.get("path", "asset")))
    destination = _path(str(entry.get("path", "")))
    url = entry.get("url")
    expected = str(entry.get("sha256") or "").lower()
    if not destination.name:
        raise ValueError(f"asset {asset_id} has no path")
    if not isinstance(url, str) or not url:
        raise ValueError(f"asset {asset_id} has no explicit archive URL")
    _validate_hash(expected, asset_id)
    required_files = [str(item) for item in entry.get("required_files", [])]
    sidecar = destination / ".asset.sha256"
    if destination.is_dir() and not force:
        if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip().lower() != expected:
            raise RuntimeError(
                f"asset {asset_id} exists without a matching archive hash; "
                "use --force only after checking provenance"
            )
        _verify_required_files(destination, required_files, asset_id)
        return {"id": asset_id, "path": str(destination), "sha256": expected, "status": "verified"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive: Path | None = None
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        archive = _download(url, destination.parent, destination.name, timeout)
        actual = _sha256(archive)
        if actual != expected:
            raise RuntimeError(f"asset {asset_id} hash mismatch: expected {expected}, got {actual}")
        archive_format = str(entry.get("archive_format", "")).lower()
        if not archive_format:
            suffixes = "".join(Path(url.split("?", 1)[0]).suffixes).lower()
            archive_format = "zip" if suffixes.endswith(".zip") else suffixes.lstrip(".")
        _safe_extract(archive, staging, archive_format)
        _verify_required_files(staging, required_files, asset_id)
        if destination.exists():
            if not force:
                raise RuntimeError(f"destination already exists: {destination}")
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(staging, destination)
        staging = None
        (destination / ".asset.sha256").write_text(expected + "\n", encoding="ascii")
    finally:
        if archive is not None:
            archive.unlink(missing_ok=True)
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return {"id": asset_id, "path": str(destination), "sha256": expected, "status": "downloaded"}


def _prepare_one(entry: dict[str, Any], *, force: bool, timeout: int) -> dict[str, Any]:
    if str(entry.get("kind", "file")).lower() in {"archive", "snapshot", "dataset"}:
        return _prepare_archive(entry, force=force, timeout=timeout)
    asset_id = str(entry.get("id", entry.get("path", "asset")))
    destination = _path(str(entry.get("path", "")))
    url = entry.get("url")
    expected = str(entry.get("sha256") or "").lower()
    if not destination.name:
        raise ValueError(f"asset {asset_id} has no path")
    if not isinstance(url, str) or not url:
        raise ValueError(f"asset {asset_id} has no explicit URL")
    _validate_hash(expected, asset_id)
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
    results = [_prepare_one(asset, force=force, timeout=timeout) for asset in assets]
    for dataset_id, dataset in manifest.get("datasets", {}).items():
        if not isinstance(dataset, dict) or not dataset.get("url"):
            continue
        dataset_entry = {"id": dataset_id, "kind": "dataset", **dataset}
        results.append(_prepare_archive(dataset_entry, force=force, timeout=timeout))
    return results


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
