"""Explicitly download and hash-check the face detector checkpoint.

The production detector never downloads assets. This script is the only
place where a network fetch is performed, and it requires a URL plus a
sha256 declared in the manifest (or supplied on the command line).
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


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = data.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{manifest_path} must contain a model object")
    return data


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _prepare_entry(
    entry: dict[str, Any],
    *,
    url_override: str | None = None,
    sha256_override: str | None = None,
    path_override: Path | None = None,
    force: bool,
    timeout: int,
) -> dict[str, Any]:
    url = url_override or entry.get("url")
    expected = (sha256_override or entry.get("sha256") or "").lower()
    destination = path_override or _resolve_repo_path(str(entry.get("path", "")))
    if not destination.name:
        raise ValueError("asset manifest entry must provide a non-empty path")
    if not url:
        raise ValueError(
            "asset URL is not configured; add entry.url or pass --model-url explicitly for the model"
        )
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("a 64-character hexadecimal sha256 is required")

    if destination.is_file() and not force:
        actual = _sha256(destination)
        if actual != expected:
            raise RuntimeError(
                f"existing asset hash mismatch for {destination}: expected {expected}, got {actual}; "
                "pass --force only after checking the asset provenance"
            )
        return {"path": str(destination), "sha256": actual, "status": "already-verified"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(str(url), timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".part", dir=destination.parent, delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        actual = _sha256(temporary_path)
        if actual != expected:
            raise RuntimeError(f"downloaded asset hash mismatch: expected {expected}, got {actual}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {"path": str(destination), "sha256": expected, "status": "downloaded", "url": str(url)}


def prepare(
    manifest_path: Path,
    *,
    model_url: str | None = None,
    expected_sha256: str | None = None,
    model_path: Path | None = None,
    force: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    data = _load_manifest(manifest_path)
    result: dict[str, Any] = {
        "model": _prepare_entry(
            data["model"],
            url_override=model_url,
            sha256_override=expected_sha256,
            path_override=model_path,
            force=force,
            timeout=timeout,
        )
    }
    for fixture in data.get("fixtures", []):
        if not isinstance(fixture, dict):
            raise ValueError("each fixtures entry must be an object")
        fixture_id = str(fixture.get("id", fixture.get("path", "fixture")))
        result[fixture_id] = _prepare_entry(fixture, force=force, timeout=timeout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/fixtures/face_detection/assets.json"))
    parser.add_argument("--model-url")
    parser.add_argument("--sha256", dest="expected_sha256")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    result = prepare(
        args.manifest,
        model_url=args.model_url,
        expected_sha256=args.expected_sha256,
        model_path=args.model_path,
        force=args.force,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
