"""Contract, geometry, invariant, and caller tests for quadrant masks."""

import numpy as np
import pytest

from src.region_engine.detector import segment_by_prompt
from src.region_engine.spatial import create_quadrant_mask

QUADRANTS = ("top", "bottom", "left", "right", "center")


def _hard_oracle(shape: tuple[int, int], quadrant: str) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    if quadrant == "top":
        selected = yy < height // 2
    elif quadrant == "bottom":
        selected = yy >= height // 2
    elif quadrant == "left":
        selected = xx < width // 2
    elif quadrant == "right":
        selected = xx >= width // 2
    else:
        selected = (
            (yy >= height // 4)
            & (yy < (3 * height) // 4)
            & (xx >= width // 4)
            & (xx < (3 * width) // 4)
        )
    return selected.astype(np.float32)


def test_public_right_slicing_baseline() -> None:
    expected = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=np.float32)
    np.testing.assert_array_equal(create_quadrant_mask((1, 10), "right", 0), expected)


def test_public_center_rectangle_baseline() -> None:
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(create_quadrant_mask((6, 6), "center", 0), expected)


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (2, 2), (3, 3), (6, 10), (7, 9), (8, 12)])
@pytest.mark.parametrize("quadrant", QUADRANTS)
def test_hard_geometry_matches_independent_index_oracle(
    shape: tuple[int, int], quadrant: str
) -> None:
    actual = create_quadrant_mask(shape, quadrant, feather_radius=0)
    expected = _hard_oracle(shape, quadrant)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1), {"top": [[0]], "bottom": [[1]], "left": [[0]], "right": [[1]], "center": [[0]]}),
        (
            (1, 5),
            {
                "top": [[0, 0, 0, 0, 0]],
                "bottom": [[1, 1, 1, 1, 1]],
                "left": [[1, 1, 0, 0, 0]],
                "right": [[0, 0, 1, 1, 1]],
                "center": [[0, 0, 0, 0, 0]],
            },
        ),
        (
            (5, 1),
            {
                "top": [[1], [1], [0], [0], [0]],
                "bottom": [[0], [0], [1], [1], [1]],
                "left": [[0], [0], [0], [0], [0]],
                "right": [[1], [1], [1], [1], [1]],
                "center": [[0], [0], [0], [0], [0]],
            },
        ),
    ],
)
def test_tiny_shapes_keep_empty_slices(shape: tuple[int, int], expected: dict[str, list[list[int]]]) -> None:
    for quadrant in QUADRANTS:
        np.testing.assert_array_equal(
            create_quadrant_mask(shape, quadrant, feather_radius=0),
            np.asarray(expected[quadrant], dtype=np.float32),
        )


@pytest.mark.parametrize("quadrant", QUADRANTS)
def test_output_contract_and_default_radius(quadrant: str) -> None:
    actual = create_quadrant_mask((17, 23), quadrant)

    assert actual.shape == (17, 23)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert float(actual.min()) >= 0.0
    assert float(actual.max()) <= 1.0
    np.testing.assert_array_equal(actual, create_quadrant_mask((17, 23), quadrant, 25))


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (1, 1), (7, 9)])
def test_hard_complement_invariants(shape: tuple[int, int]) -> None:
    top = create_quadrant_mask(shape, "top", 0)
    bottom = create_quadrant_mask(shape, "bottom", 0)
    left = create_quadrant_mask(shape, "left", 0)
    right = create_quadrant_mask(shape, "right", 0)

    np.testing.assert_array_equal(top + bottom, np.ones(shape, dtype=np.float32))
    np.testing.assert_array_equal(left + right, np.ones(shape, dtype=np.float32))
    assert not np.any(top * bottom)
    assert not np.any(left * right)


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (1, 1), (7, 9)])
def test_soft_complement_invariants(shape: tuple[int, int]) -> None:
    top = create_quadrant_mask(shape, "top", 25)
    bottom = create_quadrant_mask(shape, "bottom", 25)
    left = create_quadrant_mask(shape, "left", 25)
    right = create_quadrant_mask(shape, "right", 25)

    np.testing.assert_allclose(top + bottom, 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(left + right, 1.0, rtol=0, atol=1e-6)


def test_axis_transpose_invariant() -> None:
    top = create_quadrant_mask((5, 7), "top", 15)
    left = create_quadrant_mask((7, 5), "left", 15)
    bottom = create_quadrant_mask((5, 7), "bottom", 15)
    right = create_quadrant_mask((7, 5), "right", 15)
    center = create_quadrant_mask((5, 7), "center", 15)
    transposed_center = create_quadrant_mask((7, 5), "center", 15)

    np.testing.assert_allclose(top.T, left, rtol=0, atol=1e-6)
    np.testing.assert_allclose(bottom.T, right, rtol=0, atol=1e-6)
    np.testing.assert_allclose(center.T, transposed_center, rtol=0, atol=1e-6)


@pytest.mark.parametrize("radius", [0, 1, 24])
def test_equivalent_feather_radius_values(radius: int) -> None:
    actual = create_quadrant_mask((32, 48), "center", radius)
    expected_radius = 1 if radius == 0 else 25 if radius == 24 else radius
    expected = create_quadrant_mask((32, 48), "center", expected_radius)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [None, [], np.array(["top"]), 1])
def test_non_string_quadrant_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="string"):
        create_quadrant_mask((4, 6), value, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", "TOP", " top ", "full", "diagonal"])
def test_unknown_quadrant_is_rejected_without_full_image_fallback(value: str) -> None:
    with pytest.raises(ValueError, match="top, bottom, left, right, center"):
        create_quadrant_mask((4, 6), value, 0)


@pytest.mark.parametrize("shape", [(4, 6, 3), np.array([[4, 6]]), [4, 6, 1]])
def test_shape_must_be_exactly_two_dimensions(shape: object) -> None:
    with pytest.raises(ValueError):
        create_quadrant_mask(shape, "top", 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(0, 4), (-1, 4), [True, 4], [4.0, 6]])
def test_invalid_shape_values_are_rejected(shape: object) -> None:
    expected = TypeError if isinstance(shape[0], (bool, float)) else ValueError  # type: ignore[index]
    with pytest.raises(expected):
        create_quadrant_mask(shape, "top", 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [-1, np.int64(-1)])
def test_negative_radius_is_rejected_even_for_empty_tiny_region(radius: object) -> None:
    with pytest.raises(ValueError):
        create_quadrant_mask((1, 1), "center", radius)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [True, np.bool_(False), 2.5, "25"])
def test_invalid_radius_type_is_rejected_even_for_empty_tiny_region(radius: object) -> None:
    with pytest.raises(TypeError):
        create_quadrant_mask((1, 1), "center", radius)  # type: ignore[arg-type]


def test_read_only_shape_is_accepted_and_outputs_do_not_share_memory() -> None:
    shape = np.array([5, 7], dtype=np.int64)
    shape.setflags(write=False)
    before = shape.copy()

    first = create_quadrant_mask(shape, "center", 0)
    second = create_quadrant_mask(shape, "center", 0)

    np.testing.assert_array_equal(shape, before)
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)


@pytest.mark.parametrize("prompt", ["sky", "ground", "center"])
def test_detector_callers_still_use_validated_quadrant_api(prompt: str) -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)

    mask = segment_by_prompt(image, prompt, feather_radius=0)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.float32
    assert np.isfinite(mask).all()
