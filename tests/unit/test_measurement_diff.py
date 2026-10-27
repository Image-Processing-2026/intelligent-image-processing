"""
Unit test cho phân tích khác biệt vòng ≥ 2 (Analysis v2 P7, measurement/diff.py): tác dụng phụ
(halo, nhiễu vùng tối, banding, cháy kênh, bão hòa) và trạng thái lỗi cũ, trên ảnh synthetic với
đúng các thao tác của toolbox.
"""

import cv2
import numpy as np
import pytest

from src.analyzer_evaluator.measurement import measure_image
from src.analyzer_evaluator.measurement.diff import (
    DiffReport,
    _luma,
    compare,
    halo_overshoot,
    side_effects,
)
from src.processing_engine import (
    apply_color_balance,
    apply_denoise,
    apply_gamma,
    apply_sharpen,
)


def _scene(size: int = 512, seed: int = 3) -> np.ndarray:
    """Ảnh có cạnh mạnh (khối tối/sáng), chuyển sắc mịn và màu vừa phải."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.float32)
    ramp = np.linspace(40, 200, size, dtype=np.float32)
    image[:] = ramp[None, :, None]
    for _ in range(12):
        x, y = rng.integers(0, size - 80, 2)
        w, h = rng.integers(30, 80, 2)
        image[y : y + h, x : x + w] = rng.uniform(30, 220, 3)
    return np.clip(image, 0, 255).astype(np.uint8)


def _dark_noisy(size: int = 512, sigma: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = np.full((size, size, 3), 22.0, dtype=np.float32)
    base[: size // 2, : size // 2] = 160.0  # một mảng sáng để ảnh có cạnh
    noisy = base + rng.normal(0, sigma, base.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _kinds(report_or_effects) -> dict:
    effects = (
        report_or_effects.side_effects
        if isinstance(report_or_effects, DiffReport)
        else report_or_effects
    )
    return {effect.kind: effect.severity for effect in effects}


class TestSideEffects:
    def test_identity_has_no_side_effects(self):
        image = _scene()
        assert side_effects(image, image.copy()) == []

    def test_strong_sharpen_creates_halo(self):
        image = cv2.GaussianBlur(_scene(), (0, 0), 1.5)
        effects = _kinds(side_effects(image, apply_sharpen(image, amount=2.0)))
        assert effects.get("halo", 0) >= 1
        luma = _luma(image)
        overshoot = [
            halo_overshoot(luma, _luma(apply_sharpen(image, amount=amount)))
            for amount in (0.3, 1.0, 2.0)
        ]
        assert overshoot == sorted(overshoot) and overshoot[0] < 0.5

    def test_mild_sharpen_is_not_degradation(self):
        image = cv2.GaussianBlur(_scene(), (0, 0), 1.5)
        effects = _kinds(side_effects(image, apply_sharpen(image, amount=0.3)))
        assert max(effects.values(), default=0) <= 1

    def test_gamma_on_dark_noisy_image_raises_dark_noise(self):
        image = _dark_noisy()
        effects = _kinds(side_effects(image, apply_gamma(image, gamma=2.0)))
        assert effects.get("dark_noise", 0) >= 1

    def test_denoise_does_not_raise_dark_noise(self):
        image = _dark_noisy()
        effects = _kinds(side_effects(image, apply_denoise(image, strength=1.0)))
        assert "dark_noise" not in effects

    def test_stretching_a_smooth_gradient_creates_banding(self):
        ramp = np.tile(np.linspace(20, 70, 1024, dtype=np.float32), (256, 1))
        image = np.repeat(ramp[..., None], 3, axis=2).astype(np.uint8)
        stretched = np.clip((image.astype(np.float32) - 20) * 4.5, 0, 255).astype(np.uint8)
        assert _kinds(side_effects(image, stretched)).get("banding", 0) >= 1

    def test_brightening_bright_areas_adds_channel_clipping(self):
        image = np.full((256, 256, 3), 200, dtype=np.uint8)
        image[:, :128] = 90
        brighter = np.clip(image.astype(np.int32) + 60, 0, 255).astype(np.uint8)
        assert _kinds(side_effects(image, brighter)).get("channel_clipping", 0) >= 2

    def test_strong_saturation_boost_is_flagged(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[:, :, 0], image[:, :, 1], image[:, :, 2] = 190, 90, 60  # cam vừa
        boosted = apply_color_balance(
            apply_color_balance(image, saturation_scale=1.5), saturation_scale=1.5
        )
        assert _kinds(side_effects(image, boosted)).get("oversaturation", 0) >= 1

    def test_grayscale_input(self):
        gray = cv2.cvtColor(_scene(), cv2.COLOR_RGB2GRAY)
        assert side_effects(gray, gray.copy()) == []


class TestDefectStatus:
    def test_correct_fix_resolves_underexposure(self):
        dark = (_scene().astype(np.float32) * 0.25).astype(np.uint8)
        fixed = apply_gamma(dark, gamma=2.2)
        report = compare(
            dark,
            fixed,
            defects=[("underexposed", "full")],
            original_measurement=measure_image(dark),
            current_measurement=measure_image(fixed),
        )
        change = report.status_of("underexposed")
        assert change is not None and change.before >= 2
        assert change.status in ("resolved", "improved")

    def test_identity_is_unchanged_and_region_is_tracked(self):
        dark = (_scene().astype(np.float32) * 0.25).astype(np.uint8)
        mask = np.zeros(dark.shape[:2], dtype=np.float32)
        mask[100:400, 100:400] = 1.0
        measured = measure_image(dark)
        report = compare(
            dark,
            dark.copy(),
            {"center": mask},
            defects=[("underexposed", "full"), ("underexposed", "center"), ("noise", "sky")],
            original_measurement=measured,
            current_measurement=measured,
        )
        assert [(c.region, c.status) for c in report.defects] == [
            ("full", "unchanged"),
            ("center", "unchanged"),
        ]  # 'sky' không có mask → không theo dõi được

    def test_darkening_more_is_worse(self):
        dark = (_scene().astype(np.float32) * 0.45).astype(np.uint8)
        darker = apply_gamma(dark, gamma=0.6)
        report = compare(
            dark,
            darker,
            defects=[("underexposed", "full")],
            original_measurement=measure_image(dark),
            current_measurement=measure_image(darker),
        )
        assert report.status_of("underexposed").status == "worse"

    def test_without_measurements_only_side_effects(self):
        image = _scene()
        report = compare(image, image, defects=[("noise", "full")])
        assert report.defects == [] and report.max_side_effect == 0

    @pytest.mark.parametrize("severity", [0, 1, 2])
    def test_max_side_effect(self, severity: int):
        from src.analyzer_evaluator.measurement.diff import SideEffect

        effects = [SideEffect(kind="halo", severity=severity, value=1.0)]
        assert DiffReport(side_effects=effects).max_side_effect == severity
