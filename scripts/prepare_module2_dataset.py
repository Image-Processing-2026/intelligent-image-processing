"""Download only pinned Module 2 dataset sources and verify them before use.

The production region APIs never call this script.  Source records must have
an official URL, release/revision, license/terms and archive SHA-256.  Archive
extraction rejects path traversal and leaves the original archive in `raw/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


def _repo_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    result = (REPO_ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if REPO_ROOT != result and REPO_ROOT not in result.parents:
        raise ValueError("destination must remain inside the repository")
    return result


def _valid_hash(value: object) -> str:
    expected = str(value or "").lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("sha256 must be a 64-character hexadecimal string")
    return expected


def _safe_extract(archive: Path, destination: Path, archive_format: str) -> None:
    root = destination.resolve()
    if archive_format == "zip":
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"archive member escapes destination: {member.filename}")
            handle.extractall(destination)
        return
    if archive_format in {"tar", "tar.gz", "tgz", "tar.xz", "tar.bz2"}:
        with tarfile.open(archive, mode="r:*") as handle:
            for member in handle.getmembers():
                target = (destination / member.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"archive member escapes destination: {member.name}")
            if hasattr(tarfile, "data_filter"):
                handle.extractall(destination, filter="data")
            else:  # Python 3.10/3.11 remain safe after the member path check.
                handle.extractall(destination)
        return
    raise ValueError(f"unsupported archive_format {archive_format!r}")


def _download(url: str, destination: Path, timeout: int) -> None:
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".part", dir=destination.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_source(source: dict[str, Any], *, timeout: int, offline: bool) -> dict[str, Any]:
    source_id = str(source.get("id", "<unknown>"))
    for field in ("id", "url", "release", "license", "sha256", "archive_path", "archive_format", "raw_path"):
        if not source.get(field):
            raise ValueError(f"source {source_id} requires {field}")
    expected = _valid_hash(source["sha256"])
    archive_path = _repo_path(source["archive_path"])
    raw_path = _repo_path(source["raw_path"])
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        actual = _sha256(archive_path)
        if actual != expected:
            raise RuntimeError(f"source {source_id} archive SHA-256 mismatch")
        download_status = "already-verified"
    else:
        if offline:
            raise FileNotFoundError(f"source {source_id} archive is absent in offline mode")
        _download(str(source["url"]), archive_path, timeout)
        actual = _sha256(archive_path)
        if actual != expected:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(f"source {source_id} downloaded archive SHA-256 mismatch")
        download_status = "downloaded"
    required_files = [str(item) for item in source.get("required_files", [])]
    if raw_path.is_dir() and all((raw_path / item).is_file() for item in required_files):
        return {"id": source_id, "archive": str(archive_path), "status": f"{download_status}; raw-verified"}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{source_id}.", dir=raw_path.parent))
    try:
        _safe_extract(archive_path, staging, str(source["archive_format"]).lower())
        missing = [item for item in required_files if not (staging / item).is_file()]
        if missing:
            raise RuntimeError(f"source {source_id} is missing required files: {', '.join(missing)}")
        if raw_path.exists():
            raise RuntimeError(f"source {source_id} raw destination exists but is incomplete: {raw_path}")
        os.replace(staging, raw_path)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return {"id": source_id, "archive": str(archive_path), "raw_path": str(raw_path), "status": download_status}


def prepare(manifest_path: Path, *, timeout: int = 60, offline: bool = False) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources manifest must contain at least one pinned source")
    if not all(isinstance(source, dict) for source in sources):
        raise ValueError("every source must be an object")
    if len({str(source.get("id")) for source in sources}) != len(sources):
        raise ValueError("source IDs must be unique")
    return {"manifest": str(manifest_path), "sources": [
        _prepare_source(source, timeout=timeout, offline=offline)
        for source in sources
    ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare(args.manifest, timeout=args.timeout, offline=args.offline)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
