"""Generate independent, offline baselines for ``blend_regions``.

The expected arrays are deliberately calculated without importing the
production blending function.  Run this script explicitly when fixtures need
to be regenerated; tests only consume the checked-in arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


def _fraction_oracle(original: np.ndarray, processed: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact pre-quantization values and floor-to-byte expected output."""
    reference = np.empty(original.shape, dtype=np.float64)
    expected = np.empty(original.shape, dtype=np.uint8)
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            alpha = Fraction.from_float(float(mask[y, x]))
            for channel in range(original.shape[2]):
                value = alpha * int(processed[y, x, channel]) + (1 - alpha) * int(
                    original[y, x, channel]
                )
                reference[y, x, channel] = float(value)
                expected[y, x, channel] = value.numerator // value.denominator
    return expected, reference


def _case_data() -> dict[str, dict[str, Any]]:
    """Build deterministic cases from documented inputs and a fixed RNG seed."""
    cases: dict[str, dict[str, Any]] = {}

    cases["w3c_opaque"] = {
        "original": np.array([[[255, 0, 0]]], dtype=np.uint8),
        "processed": np.array([[[0, 0, 255]]], dtype=np.uint8),
        "mask": np.array([[1.0]], dtype=np.float32),
        "baseline_type": "W3C 5.1.1 opaque normalized to uint8 RGB",
    }
    cases["w3c_half"] = {
        "original": np.array([[[255, 0, 0]]], dtype=np.uint8),
        "processed": np.array([[[0, 0, 255]]], dtype=np.uint8),
        "mask": np.array([[0.5]], dtype=np.float32),
        "baseline_type": "W3C 5.1.1 half-alpha normalized to uint8 RGB",
    }

    original_2x2 = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
    processed_2x2 = np.full((2, 2, 3), [110, 220, 230], dtype=np.uint8)
    cases["spatial_2x2"] = {
        "original": original_2x2,
        "processed": processed_2x2,
        "mask": np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32),
        "baseline_type": "project hand-calculated spatial mask",
    }

    original_dyadic = np.zeros((1, 257, 3), dtype=np.uint8)
    processed_dyadic = np.zeros_like(original_dyadic)
    original_dyadic[0, :, 0] = np.arange(257, dtype=np.uint16) % 256
    original_dyadic[0, :, 1] = 255 - original_dyadic[0, :, 0]
    original_dyadic[0, :, 2] = 127
    processed_dyadic[0, :, 0] = 255 - original_dyadic[0, :, 0]
    processed_dyadic[0, :, 1] = np.arange(257, dtype=np.uint16) % 256
    processed_dyadic[0, :, 2] = 128
    t = np.arange(257, dtype=np.float32)
    cases["dyadic"] = {
        "original": original_dyadic,
        "processed": processed_dyadic,
        "mask": (t / np.float32(256.0)).reshape(1, 257),
        "baseline_type": "independent int64 oracle for alpha t/256",
        "alpha_numerator": list(range(257)),
    }

    rng = np.random.default_rng(20260921)
    random_original = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
    random_processed = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
    random_mask = rng.random((17, 23), dtype=np.float32)
    cases["random"] = {
        "original": random_original,
        "processed": random_processed,
        "mask": random_mask,
        "baseline_type": "Fraction oracle from stored float32 mask",
        "rng": {"algorithm": "numpy.default_rng", "seed": 20260921},
    }

    same = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    cases["same"] = {
        "original": same,
        "processed": same.copy(),
        "mask": rng.random((2, 3), dtype=np.float32),
        "baseline_type": "identity-channel preservation",
    }

    cases["zero"] = {
        "original": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "processed": (255 - np.arange(18, dtype=np.uint8)).reshape(2, 3, 3),
        "mask": np.zeros((2, 3), dtype=np.float32),
        "baseline_type": "mask zero identity",
    }
    cases["one"] = {
        "original": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "processed": (255 - np.arange(18, dtype=np.uint8)).reshape(2, 3, 3),
        "mask": np.ones((2, 3), dtype=np.float32),
        "baseline_type": "mask one processed image",
    }
    cases["none"] = {
        "original": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "processed": (255 - np.arange(18, dtype=np.uint8)).reshape(2, 3, 3),
        "mask": None,
        "baseline_type": "None mask selects processed image",
    }

    near = np.array(
        [np.float32(0.1), np.nextafter(np.float32(0.1), np.float32(0.0)), np.nextafter(np.float32(0.1), np.float32(1.0))],
        dtype=np.float32,
    ).reshape(1, 3)
    cases["near_integer"] = {
        "original": np.full((1, 3, 3), 10, dtype=np.uint8),
        "processed": np.full((1, 3, 3), 110, dtype=np.uint8),
        "mask": near,
        "baseline_type": "Fraction oracle for float32 0.1 and adjacent values",
    }
    return cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cases_manifest: list[dict[str, Any]] = []

    for case_name, data in _case_data().items():
        original = data["original"]
        processed = data["processed"]
        mask = data["mask"]
        if mask is None:
            expected = processed.copy()
            reference = processed.astype(np.float64)
        else:
            expected, reference = _fraction_oracle(original, processed, mask)

        np.save(output / f"{case_name}_original.npy", original)
        np.save(output / f"{case_name}_processed.npy", processed)
        if mask is not None:
            np.save(output / f"{case_name}_mask.npy", mask)
        np.save(output / f"{case_name}_expected.npy", expected)
        np.save(output / f"{case_name}_reference_float64.npy", reference)

        files = [
            f"{case_name}_original.npy",
            f"{case_name}_processed.npy",
            f"{case_name}_expected.npy",
            f"{case_name}_reference_float64.npy",
        ]
        if mask is not None:
            files.insert(2, f"{case_name}_mask.npy")
        cases_manifest.append(
            {
                "name": case_name,
                "baseline_type": data["baseline_type"],
                "files": files,
                "shape": list(original.shape),
                "dtypes": {
                    "original": str(original.dtype),
                    "processed": str(processed.dtype),
                    "mask": None if mask is None else str(mask.dtype),
                    "expected": str(expected.dtype),
                    "reference_float64": str(reference.dtype),
                },
                "mask_shape": None if mask is None else list(mask.shape),
                "mask_values": None
                if mask is None
                else {"min": float(mask.min()), "max": float(mask.max())},
                "rng": data.get("rng"),
                "alpha_numerator": data.get("alpha_numerator"),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Independent fixtures for src.region_engine.mask_utils.blend_regions",
        "generated_by": "scripts/generate_blend_baselines.py",
        "provenance": {
            "w3c_url": "https://www.w3.org/TR/2024/CRD-compositing-1-20240321/",
            "w3c_section": "5.1.1 Examples of simple alpha compositing",
            "w3c_normalization": "RGB channels 0/1 converted to uint8 0/255; opaque background",
            "project_cases": "Hand-designed inputs documented in docs/implementation-docs/02_module-2-blend-regions.md",
            "network_required": False,
        },
        "numeric_contract": {
            "formula": "M * processed + (1 - M) * original",
            "calculation_dtype": "float32",
            "output_dtype": "uint8",
            "quantization": "clip to [0, 255], then truncate toward zero",
            "oracle": "Python Fraction from stored float32 values; dyadic case also records int64 formula",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "cases": cases_manifest,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hashes = {path.name: _sha256(path) for path in sorted(output.glob("*.npy"))}
    manifest["sha256"] = hashes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {len(cases_manifest)} blend fixture cases in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
