"""
Unit test (offline) cho src/analyzer_evaluator/metadata.py: đọc EXIF/JPEG/ICC và prior.
JPEG có EXIF giả được tạo ngay trong test bằng Pillow.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageCms
from PIL.TiffImagePlugin import IFDRational

from src.analyzer_evaluator.metadata import (
    ImageMetadata,
    estimate_jpeg_quality,
    metadata_priors,
    read_metadata,
)

_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_SOFTWARE = 0x0131
_EXIF_IFD = 0x8769
_TAG_EXPOSURE_TIME = 0x829A
_TAG_F_NUMBER = 0x829D
_TAG_ISO = 0x8827
_TAG_FLASH = 0x9209
_TAG_FOCAL_35MM = 0xA405


def _base_image(size: int = 64) -> Image.Image:
    """Ảnh RGB có họa tiết để nén JPEG có ý nghĩa."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))


def _jpeg(quality: int = 90, exif: Image.Exif | None = None, icc: bytes | None = None) -> bytes:
    buf = BytesIO()
    kwargs = {"quality": quality}
    if exif is not None:
        kwargs["exif"] = exif.tobytes()
    if icc is not None:
        kwargs["icc_profile"] = icc
    _base_image().save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def _camera_exif(
    *,
    iso: int = 3200,
    exposure: float = 1 / 50,
    f_number: float = 2.0,
    focal_35mm: int = 50,
    flash: int = 1,
    make: str = "Canon",
    model: str = "Canon EOS R6",
    software: str | None = None,
) -> Image.Exif:
    exif = Image.Exif()
    exif[_TAG_MAKE] = make
    exif[_TAG_MODEL] = model
    if software is not None:
        exif[_TAG_SOFTWARE] = software
    sub = exif.get_ifd(_EXIF_IFD)
    sub[_TAG_ISO] = iso
    sub[_TAG_EXPOSURE_TIME] = IFDRational(round(exposure * 1000), 1000)
    sub[_TAG_F_NUMBER] = IFDRational(round(f_number * 10), 10)
    sub[_TAG_FOCAL_35MM] = focal_35mm
    sub[_TAG_FLASH] = flash
    return exif


def test_reads_exif_fields_from_jpeg() -> None:
    data = _jpeg(90, _camera_exif(software="Lightroom 13.2"))

    meta = read_metadata(data)

    assert meta.iso == 3200
    assert meta.exposure_time_s == pytest.approx(0.02)
    assert meta.f_number == pytest.approx(2.0)
    assert meta.focal_length_35mm == 50
    assert meta.flash_fired is True
    assert meta.camera == "Canon EOS R6"
    assert meta.software == "Lightroom 13.2"


def test_camera_name_does_not_repeat_make_when_model_has_it() -> None:
    meta = read_metadata(_jpeg(90, _camera_exif(make="Sony", model="ILCE-7M4")))

    assert meta.camera == "Sony ILCE-7M4"


def test_flash_not_fired_is_false() -> None:
    meta = read_metadata(_jpeg(90, _camera_exif(flash=0)))

    assert meta.flash_fired is False


def test_jpeg_without_exif_reports_only_quality() -> None:
    meta = read_metadata(_jpeg(90))

    assert meta.iso is None
    assert meta.software is None
    assert meta.camera is None
    assert meta.jpeg_quality_estimate == 90


def test_png_without_metadata_is_all_none() -> None:
    buf = BytesIO()
    _base_image().save(buf, format="PNG")

    meta = read_metadata(buf.getvalue())

    assert meta == ImageMetadata()


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\xff\xd8\xff\xe0garbage"])
def test_broken_bytes_never_raise(data: bytes) -> None:
    meta = read_metadata(data)

    assert meta == ImageMetadata()


def test_truncated_jpeg_never_raises() -> None:
    data = _jpeg(90, _camera_exif())

    meta = read_metadata(data[: len(data) // 3])

    assert isinstance(meta, ImageMetadata)


def test_icc_profile_name_is_read() -> None:
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    meta = read_metadata(_jpeg(90, icc=icc))

    assert meta.icc_profile_name
    assert "sRGB" in meta.icc_profile_name


@pytest.mark.parametrize("quality", [30, 60, 90])
def test_jpeg_quality_estimate_within_tolerance(quality: int) -> None:
    meta = read_metadata(_jpeg(quality))

    assert meta.jpeg_quality_estimate is not None
    assert abs(meta.jpeg_quality_estimate - quality) <= 10


def test_estimate_matches_ijg_table_exactly() -> None:
    # Bảng luma IJG ở q=50 (hàng-cột) chính là bảng chuẩn
    base = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ]  # fmt: skip

    assert estimate_jpeg_quality(base) == 50


def _meta(**fields: object) -> ImageMetadata:
    return ImageMetadata(**fields)


def test_no_priors_for_empty_metadata() -> None:
    assert metadata_priors(ImageMetadata()) == []


def test_noise_prior_starts_at_iso_1600() -> None:
    assert metadata_priors(_meta(iso=1599)) == []

    priors = metadata_priors(_meta(iso=1600))

    assert [p.kind for p in priors] == ["noise"]
    assert priors[0].strength == pytest.approx(0.4)


def test_noise_prior_strength_grows_and_caps_at_one() -> None:
    mid = metadata_priors(_meta(iso=3200))[0].strength
    high = metadata_priors(_meta(iso=12800))[0].strength
    capped = metadata_priors(_meta(iso=102400))[0].strength

    assert 0.4 < mid < high
    assert high == pytest.approx(1.0)
    assert capped == pytest.approx(1.0)


def test_motion_blur_when_exposure_slower_than_focal_reciprocal() -> None:
    # 1/35 s với tiêu cự 35mm: đúng ngưỡng, chưa vượt → không có prior
    assert metadata_priors(_meta(exposure_time_s=1 / 35, focal_length_35mm=35)) == []

    priors = metadata_priors(_meta(exposure_time_s=1 / 30, focal_length_35mm=35))

    assert [p.kind for p in priors] == ["motion_blur"]
    assert 0.0 < priors[0].strength <= 1.0


def test_motion_blur_needs_both_exposure_and_focal() -> None:
    assert metadata_priors(_meta(exposure_time_s=1.0)) == []
    assert metadata_priors(_meta(focal_length_35mm=50)) == []


def test_bokeh_prior_at_f28_and_below() -> None:
    assert metadata_priors(_meta(f_number=2.9)) == []

    at_28 = metadata_priors(_meta(f_number=2.8))[0]
    wide = metadata_priors(_meta(f_number=1.4))[0]

    assert at_28.kind == "bokeh"
    assert at_28.strength == pytest.approx(0.4)
    assert wide.strength == pytest.approx(1.0)


def test_no_sharpen_prior_below_quality_60() -> None:
    assert metadata_priors(_meta(jpeg_quality_estimate=60)) == []

    just_below = metadata_priors(_meta(jpeg_quality_estimate=59))[0]
    heavy = metadata_priors(_meta(jpeg_quality_estimate=30))[0]

    assert just_below.kind == "no_sharpen"
    assert just_below.strength == pytest.approx(0.2)
    assert heavy.strength == pytest.approx(1.0)


def test_edited_prior_when_software_present() -> None:
    priors = metadata_priors(_meta(software="Photoshop 25.0"))

    assert [p.kind for p in priors] == ["edited"]
    assert "Photoshop 25.0" in priors[0].reason


def test_reasons_are_vietnamese_text() -> None:
    priors = metadata_priors(_meta(iso=6400, f_number=1.8, software="GIMP"))

    assert all(p.reason for p in priors)
    assert all(0.0 <= p.strength <= 1.0 for p in priors)
