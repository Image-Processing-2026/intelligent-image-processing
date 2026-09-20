"""Contract, numerical, and ownership tests for :func:`blend_regions`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.region_engine.mask_utils import blend_regions

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "blend_regions"


def _fixture_names() -> list[str]:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [case["name"] for case in manifest["cases"]]


def _load_case(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    original = np.load(FIXTURE_DIR / f"{name}_original.npy")
    processed = np.load(FIXTURE_DIR / f"{name}_processed.npy")
    mask_path = FIXTURE_DIR / f"{name}_mask.npy"
    mask = np.load(mask_path) if mask_path.exists() else None
    expected = np.load(FIXTURE_DIR / f"{name}_expected.npy")
    return original, processed, mask, expected


@pytest.mark.parametrize("name", _fixture_names())
def test_documented_fixture_cases(name: str) -> None:
    original, processed, mask, expected = _load_case(name)
    original_before = original.copy()
    processed_before = processed.copy()
    mask_before = None if mask is None else mask.copy()
    actual = blend_regions(original, processed, mask)

    if name == "near_integer":
        # The documented float32 boundary case may move one value across an
        # integer boundary.  All other fixture cases are exact.
        difference = actual.astype(np.int16) - expected.astype(np.int16)
        assert np.max(np.abs(difference)) <= 1
        assert np.count_nonzero(difference) == 3
    else:
        np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.shape == original.shape
    assert not np.shares_memory(actual, original)
    assert not np.shares_memory(actual, processed)
    np.testing.assert_array_equal(original, original_before)
    np.testing.assert_array_equal(processed, processed_before)
    if mask is not None:
        np.testing.assert_array_equal(mask, mask_before)


def test_w3c_half_alpha_and_spatial_mask() -> None:
    _, _, _, w3c_expected = _load_case("w3c_half")
    actual = blend_regions(
        np.array([[[255, 0, 0]]], dtype=np.uint8),
        np.array([[[0, 0, 255]]], dtype=np.uint8),
        np.array([[0.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(actual, w3c_expected)

    original, processed, mask, expected = _load_case("spatial_2x2")
    np.testing.assert_array_equal(blend_regions(original, processed, mask[..., None]), expected)


def test_none_mask_returns_a_copy_of_processed_image() -> None:
    original = np.zeros((2, 3, 3), dtype=np.uint8)
    processed = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    actual = blend_regions(original, processed, None)

    np.testing.assert_array_equal(actual, processed)
    assert actual is not processed
    assert not np.shares_memory(actual, processed)
    actual[0, 0, 0] = 255
    assert processed[0, 0, 0] != 255


def test_zero_one_and_equal_channels_are_exact() -> None:
    original = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    processed = 255 - original
    np.testing.assert_array_equal(
        blend_regions(original, processed, np.zeros((2, 3), dtype=np.float32)), original
    )
    np.testing.assert_array_equal(
        blend_regions(original, processed, np.ones((2, 3), dtype=np.float32)), processed
    )

    same = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    random_mask = np.array([[0.1, 0.25, 0.9], [0.0, 0.5, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(blend_regions(same, same.copy(), random_mask), same)


def test_dyadic_alpha_matches_independent_integer_oracle() -> None:
    original = np.array([[[0, 255, 127]] * 257], dtype=np.uint8)
    processed = np.array([[[255, 0, 128]] * 257], dtype=np.uint8)
    t = np.arange(257, dtype=np.int64).reshape(1, 257, 1)
    mask = (t.astype(np.float32) / np.float32(256.0))[..., 0]
    expected = (((256 - t) * original.astype(np.int64) + t * processed.astype(np.int64)) // 256).astype(
        np.uint8
    )
    np.testing.assert_array_equal(blend_regions(original, processed, mask), expected)


def test_non_contiguous_and_read_only_inputs_are_supported() -> None:
    original, processed, mask, expected = _load_case("random")
    # Striding all three inputs preserves the values while exercising the
    # non-contiguous path without relying on implicit copies in the caller.
    original_padded = np.empty((original.shape[0], original.shape[1] * 2, 3), dtype=np.uint8)
    processed_padded = np.empty_like(original_padded)
    mask_padded = np.empty((mask.shape[0], mask.shape[1] * 2), dtype=np.float32)
    original_padded[:, ::2] = original
    processed_padded[:, ::2] = processed
    mask_padded[:, ::2] = mask
    original_view = original_padded[:, ::2]
    processed_view = processed_padded[:, ::2]
    mask_view = mask_padded[:, ::2]
    original_view.setflags(write=False)
    processed_view.setflags(write=False)
    mask_view.setflags(write=False)

    actual = blend_regions(original_view, processed_view, mask_view)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "bad_original",
    [
        [[0, 1, 2]],
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3, 4), dtype=np.uint8),
        np.zeros((0, 3, 3), dtype=np.uint8),
    ],
)
def test_invalid_original_image_is_rejected(bad_original: object) -> None:
    valid = np.zeros((2, 3, 3), dtype=np.uint8)
    expected_exception = TypeError if not isinstance(bad_original, np.ndarray) or bad_original.dtype != np.uint8 else ValueError
    with pytest.raises(expected_exception):
        blend_regions(bad_original, valid, None)  # type: ignore[arg-type]


def test_image_shape_and_dtype_mismatches_are_rejected() -> None:
    original = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        blend_regions(original, original.astype(np.int16), None)
    with pytest.raises(ValueError):
        blend_regions(original, np.zeros((2, 4, 3), dtype=np.uint8), None)


@pytest.mark.parametrize(
    "bad_mask, exception",
    [
        ([[0.0]], TypeError),
        (np.zeros((2, 3), dtype=np.float64), TypeError),
        (np.zeros((2, 3), dtype=np.uint8), TypeError),
        (np.zeros((2, 3), dtype=bool), TypeError),
        (np.zeros((2, 3, 3), dtype=np.float32), ValueError),
        (np.zeros((2, 4), dtype=np.float32), ValueError),
        (np.zeros((2, 3, 1, 1), dtype=np.float32), ValueError),
        (np.float32(0.5), TypeError),
    ],
)
def test_invalid_mask_type_or_shape_is_rejected(bad_mask: object, exception: type[Exception]) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(exception):
        blend_regions(image, image, bad_mask)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, np.nan, np.inf, -np.inf],
)
def test_invalid_mask_values_are_rejected(value: float) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    mask = np.zeros((2, 3), dtype=np.float32)
    mask[0, 0] = value
    with pytest.raises(ValueError):
        blend_regions(image, image, mask)
