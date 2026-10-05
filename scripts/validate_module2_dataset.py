"""Validate locked Module 2 cases without loading an inference model.

Exit status is intentionally meaningful: 0 for a complete valid manifest, 1
for schema/content errors and 2 for missing local data.  The validator does
not download, relabel or silently skip a case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED = {
    "case_id",
    "source_id",
    "image",
    "image_sha256",
    "prompt",
    "split",
    "gt_kind",
    "annotation_origin",
    "policy_version",
    "review_status",
}
_GT_KINDS = frozenset(("instance", "semantic", "face_oval", "bbox", "empty"))
_SPLITS = frozenset(("calibration", "test"))
_REVIEWED = frozenset(("reviewed",))


def _repo_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else REPO_ROOT / candidate).resolve()
    if REPO_ROOT != resolved and REPO_ROOT not in resolved.parents:
        raise ValueError("path must remain inside the repository")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return [], [f"cases manifest is missing: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: case must be an object")
            continue
        cases.append(value)
    if not cases and not errors:
        errors.append("cases manifest contains no cases")
    return cases, errors


def _validate_case(case: dict[str, Any], *, require_reviewed: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    missing: list[str] = []
    case_id = str(case.get("case_id", "<unknown>"))
    absent = sorted(_REQUIRED - case.keys())
    if absent:
        return [f"{case_id}: missing required fields: {', '.join(absent)}"], missing
    if not isinstance(case["case_id"], str) or not case["case_id"].strip():
        errors.append(f"{case_id}: case_id must be a non-empty string")
    if case["split"] not in _SPLITS:
        errors.append(f"{case_id}: split must be calibration or test")
    if case["gt_kind"] not in _GT_KINDS:
        errors.append(f"{case_id}: unsupported gt_kind {case['gt_kind']!r}")
    if require_reviewed and case["review_status"] not in _REVIEWED:
        errors.append(f"{case_id}: review_status must be reviewed")
    image_path: Path | None = None
    try:
        image_path = _repo_path(case["image"])
    except ValueError as exc:
        errors.append(f"{case_id}: invalid image path ({exc})")
    if image_path is not None:
        if not image_path.is_file():
            missing.append(f"{case_id}: image missing: {image_path}")
        else:
            expected_hash = str(case["image_sha256"]).lower()
            if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
                errors.append(f"{case_id}: image_sha256 must be 64 lowercase hexadecimal characters")
            elif _sha256(image_path) != expected_hash:
                errors.append(f"{case_id}: image SHA-256 does not match manifest")
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                dimensions = case.get("dimensions")
                if dimensions is not None and dimensions != [height, width]:
                    errors.append(f"{case_id}: dimensions must be [H, W] matching the canonical image")
            except Exception as exc:
                errors.append(f"{case_id}: image cannot be decoded ({exc})")
    if case["gt_kind"] != "empty":
        for field in ("mask", "mask_sha256"):
            if field not in case:
                errors.append(f"{case_id}: {field} is required for non-empty ground truth")
        if "mask" in case:
            try:
                mask_path = _repo_path(case["mask"])
            except ValueError as exc:
                errors.append(f"{case_id}: invalid mask path ({exc})")
            else:
                if not mask_path.is_file():
                    missing.append(f"{case_id}: mask missing: {mask_path}")
                elif "mask_sha256" in case and _sha256(mask_path) != str(case["mask_sha256"]).lower():
                    errors.append(f"{case_id}: mask SHA-256 does not match manifest")
                elif image_path is not None and image_path.is_file():
                    try:
                        mask = np.asarray(Image.open(mask_path))
                        with Image.open(image_path) as image:
                            expected_shape = (image.height, image.width)
                        if mask.ndim != 2 or mask.shape != expected_shape:
                            errors.append(f"{case_id}: mask must be 2-D and match image H×W")
                    except Exception as exc:
                        errors.append(f"{case_id}: mask cannot be decoded ({exc})")
    return errors, missing


def validate(cases_path: Path, *, require_reviewed: bool = False) -> dict[str, Any]:
    cases, schema_errors = _load_cases(cases_path)
    errors = list(schema_errors)
    missing: list[str] = []
    case_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", "<unknown>"))
        if case_id in case_ids:
            errors.append(f"{case_id}: duplicate case_id")
        case_ids.add(case_id)
        case_errors, case_missing = _validate_case(case, require_reviewed=require_reviewed)
        errors.extend(case_errors)
        missing.extend(case_missing)
    return {
        "cases_path": str(cases_path),
        "case_count": len(cases),
        "valid": not errors and not missing,
        "errors": errors,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    report = validate(args.cases, require_reviewed=args.require_reviewed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing"]:
        raise SystemExit(2)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
