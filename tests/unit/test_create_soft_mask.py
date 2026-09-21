"""Contract and numerical tests for :func:`create_soft_mask`."""

import numpy as np
import pytest

from src.region_engine.mask_utils import create_soft_mask


def _scipy_reference(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    normalized = mask.astype(np.float64)
    if mask.dtype == np.uint8 and np.any(mask == 255):
        normalized /= 255.0
    kernel_size = feather_radius if feather_radius % 2 else feather_radius + 1
    return scipy_ndimage.gaussian_filter(
        normalized,
        sigma=(kernel_size / 3.0, kernel_size / 3.0),
        order=0,
        mode="mirror",
        radius=(kernel_size - 1) // 2,
        output=np.float64,
    )


def test_zero_and_one_masks_are_constant_at_the_edges() -> None:
    for value in (0, 1, 255):
        mask = np.full((32, 48), value, dtype=np.uint8)
        actual = create_soft_mask(mask, feather_radius=15)
        expected = 1.0 if value else 0.0
        assert actual.dtype == np.float32
        assert np.all(actual == expected)


@pytest.mark.parametrize("feather_radius", [0, 1])
def test_zero_and_one_radius_only_normalize(feather_radius: int) -> None:
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    before = mask.copy()

    actual = create_soft_mask(mask, feather_radius=feather_radius)

    np.testing.assert_array_equal(actual, np.array([[0, 1], [1, 0]], dtype=np.float32))
    np.testing.assert_array_equal(mask, before)
    assert not np.shares_memory(actual, mask)


def test_impulse_matches_independent_gaussian_equation() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    q = np.exp(-0.5)
    a = q / (1.0 + 2.0 * q)
    b = 1.0 / (1.0 + 2.0 * q)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, a * a, a * b, a * a, 0],
            [0, a * b, b * b, a * b, 0],
            [0, a * a, a * b, a * a, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    actual = create_soft_mask(mask, feather_radius=3)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("encoding", ["bool", "01", "0255"])
def test_encodings_match(encoding: str) -> None:
    binary = np.zeros((31, 47), dtype=np.uint8)
    binary[8:23, 11:36] = 1
    if encoding == "bool":
        mask = binary.astype(bool)
    elif encoding == "0255":
        mask = binary * 255
    else:
        mask = binary

    actual = create_soft_mask(mask, feather_radius=15)
    expected = _scipy_reference(mask, feather_radius=15)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_even_parameter_rounds_to_the_same_kernel_as_next_odd() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1

    actual = create_soft_mask(mask, feather_radius=14)
    expected = create_soft_mask(mask, feather_radius=15)

    np.testing.assert_array_equal(actual, expected)


def test_non_contiguous_input_is_supported() -> None:
    mask = np.zeros((32, 64), dtype=np.uint8)
    mask[8:24, 16:48] = 255
    view = mask[:, ::2]

    actual = create_soft_mask(view, feather_radius=15)
    expected = _scipy_reference(view, feather_radius=15)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "mask",
    [
        np.empty((0, 10), dtype=np.uint8),
        np.empty((10,), dtype=np.uint8),
        np.empty((10, 10, 1), dtype=np.uint8),
    ],
)
def test_invalid_shapes_raise_value_error(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        create_soft_mask(mask)


@pytest.mark.parametrize(
    "mask",
    [
        [[0, 1]],
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.int16),
        np.zeros((4, 4), dtype=np.complex64),
    ],
)
def test_invalid_array_types_raise_type_error(mask: object) -> None:
    with pytest.raises(TypeError):
        create_soft_mask(mask)  # type: ignore[arg-type]


def test_invalid_uint8_values_raise_value_error() -> None:
    with pytest.raises(ValueError):
        create_soft_mask(np.array([[0, 1, 255]], dtype=np.uint8))
    with pytest.raises(ValueError):
        create_soft_mask(np.array([[0, 128]], dtype=np.uint8))


@pytest.mark.parametrize("feather_radius", [-1, np.int64(-1)])
def test_negative_radius_raises_value_error(feather_radius: object) -> None:
    with pytest.raises(ValueError):
        create_soft_mask(np.zeros((2, 2), dtype=np.uint8), feather_radius=feather_radius)  # type: ignore[arg-type]


@pytest.mark.parametrize("feather_radius", [True, np.bool_(False), 2.5, "15"])
def test_non_integer_radius_raises_type_error(feather_radius: object) -> None:
    with pytest.raises(TypeError):
        create_soft_mask(np.zeros((2, 2), dtype=np.uint8), feather_radius=feather_radius)  # type: ignore[arg-type]


def test_output_contract_and_input_immutability() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    before = mask.copy()

    actual = create_soft_mask(mask, feather_radius=15)

    assert actual.shape == mask.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert float(actual.min()) >= 0.0
    assert float(actual.max()) <= 1.0
    np.testing.assert_array_equal(mask, before)
