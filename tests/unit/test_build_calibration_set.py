"""Unit test offline cho bộ sinh synthetic (scripts/build_calibration_set.py)."""

import json

import cv2
import numpy as np
import pytest

from scripts.build_calibration_set import (
    apply_degradation,
    build_variants,
    variant_plan,
    write_variant,
)


@pytest.fixture
def image() -> np.ndarray:
    """Ảnh RGB uint8 có cấu trúc (gradient theo kênh) để các phép biến đổi có tác dụng."""
    rng = np.random.default_rng(0)
    base = np.linspace(40, 200, 64, dtype=np.float32)[None, :, None]
    noise = rng.integers(-10, 10, (48, 64, 3)).astype(np.float32)
    return np.clip(base + noise + np.array([0, 5, -5], np.float32), 0, 255).astype(np.uint8)


def test_variant_plan_names_are_unique_and_clean_has_no_defects():
    plan = variant_plan()
    names = [name for name, _ in plan]
    assert len(names) == len(set(names))
    clean = dict(plan)["clean"]
    assert clean == []


def test_every_variant_keeps_shape_and_dtype(image):
    for name, degradations in variant_plan():
        rng = np.random.default_rng(7)
        out = image
        for deg in degradations:
            out = apply_degradation(out, deg, rng)
        assert out.shape == image.shape, name
        assert out.dtype == np.uint8, name


def test_same_seed_gives_identical_pixels(image):
    first = build_variants(image, "src.jpg", seed=7)
    second = build_variants(image, "src.jpg", seed=7)
    assert [name for name, _, _ in first] == [name for name, _, _ in second]
    for (_, a, pa), (_, b, pb) in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert pa == pb


def test_gaussian_noise_matches_requested_sigma():
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    out = apply_degradation(
        flat, {"type": "gaussian_noise", "sigma": 10.0}, np.random.default_rng(7)
    )
    measured = float(out.astype(np.float32).std())
    assert measured == pytest.approx(10.0, rel=0.1)


def test_poisson_gaussian_noise_depends_on_brightness():
    dark = np.full((128, 128, 3), 10, dtype=np.uint8)
    bright = np.full((128, 128, 3), 240, dtype=np.uint8)
    deg = {"type": "poisson_gaussian_noise", "a": 0.25, "b": 2.0}
    std_dark = apply_degradation(dark, deg, np.random.default_rng(7)).astype(np.float32).std()
    std_bright = apply_degradation(bright, deg, np.random.default_rng(7)).astype(np.float32).std()
    assert std_bright > std_dark * 1.5


def test_gamma_direction_and_expected_defects():
    flat = np.full((8, 8, 3), 128, dtype=np.uint8)
    darker = apply_degradation(flat, {"type": "gamma", "gamma": 1.5}, np.random.default_rng(0))
    brighter = apply_degradation(flat, {"type": "gamma", "gamma": 0.5}, np.random.default_rng(0))
    assert darker.mean() < 128 < brighter.mean()
    entries = dict(variant_plan())
    assert entries["gamma_1.5"] == [{"type": "gamma", "gamma": 1.5}]


def test_haze_lifts_dark_pixels_toward_airlight():
    dark = np.full((8, 8, 3), 20, dtype=np.uint8)
    out = apply_degradation(dark, {"type": "haze", "transmission": 0.6}, np.random.default_rng(0))
    assert out.mean() > 20
    assert out.mean() == pytest.approx(20 * 0.6 + 220 * 0.4, abs=1.0)


def test_build_variants_writes_expected_defects_in_params(image):
    variants = {name: params for name, _, params in build_variants(image, "src.jpg")}
    assert variants["clean"]["expected"]["defects"] == []
    assert variants["noise_s15"]["expected"]["defects"] == [
        {"type": "noise", "region": "full", "severity": 3}
    ]
    assert variants["cct_3000"]["expected"]["defects"][0]["type"] == "color_cast_warm"
    assert variants["cct_8000"]["expected"]["defects"][0]["type"] == "color_cast_cool"
    # mức ám nhẹ (lane color): nhãn mức 1, đúng hướng
    assert variants["cct_5000"]["expected"]["defects"] == [
        {"type": "color_cast_warm", "region": "full", "severity": 1}
    ]
    assert variants["cct_7000"]["expected"]["defects"] == [
        {"type": "color_cast_cool", "region": "full", "severity": 1}
    ]
    assert variants["jpeg_q20"]["expected"]["defects"] == []
    combo = variants["combo_noise_dark"]["expected"]["defects"]
    assert {d["type"] for d in combo} == {"noise", "underexposed"}
    assert variants["noise_s2"]["seed"] == 7
    # params.json phải serialize được và giữ đúng format C5
    json.dumps(variants["combo_blur_noise"])
    assert set(variants["combo_blur_noise"]) == {"source", "seed", "degradations", "expected"}


def test_write_variant_keeps_rgb_order_on_disk(tmp_path, image):
    name, out_image, params = build_variants(image, "src.jpg")[1]
    write_variant(tmp_path, "src", name, out_image, params)
    stem = f"src__{name}"
    loaded = cv2.cvtColor(cv2.imread(str(tmp_path / f"{stem}.png")), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(loaded, out_image)
    saved = json.loads((tmp_path / f"{stem}.params.json").read_text(encoding="utf-8"))
    assert saved["expected"] == params["expected"]
