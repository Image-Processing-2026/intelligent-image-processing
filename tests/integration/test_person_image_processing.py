"""Integration test: đưa ảnh chứa người thực (human-pic.jpg) qua toàn bộ Module 2.

Ảnh đầu vào: data/image/human-pic.jpg — người phụ nữ và chó trong vườn hoa (853×1280 RGB).
Các test không yêu cầu model thật dùng monkeypatch để giả lập MediaPipe.
Test đánh dấu ``real_model`` yêu cầu cả model checkpoint lẫn ảnh portrait fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.region_engine.face_detector as face_detector_module
from src.region_engine.controller import (
    RegionRequest,
    RegionResult,
    resolve_region,
)
from src.region_engine.face_detector import FaceDetectionRecord, reset_face_detector
from src.region_engine.mask_utils import blend_regions

# ---------------------------------------------------------------------------
# Ảnh người thực: data/image/human-pic.jpg — người phụ nữ + chó trong vườn hoa
# Kích thước: 853 × 1280, RGB uint8
# ---------------------------------------------------------------------------

_IMAGE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "image" / "human-pic.jpg"
)


def _load_person_image() -> np.ndarray:
    """Nạp ảnh thực từ đĩa, convert sang RGB uint8."""
    from PIL import Image as PILImage

    with PILImage.open(_IMAGE_PATH) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


_PERSON_IMAGE = _load_person_image()
_H, _W = _PERSON_IMAGE.shape[:2]  # 853, 1280

# Bounding box ước lượng khuôn mặt người phụ nữ trong ảnh
# (nằm ở khoảng x=700-900, y=290-500 trong ảnh 853×1280)
_FACE_BBOX = FaceDetectionRecord(x=700, y=290, width=200, height=210, score=0.95)


# ---------------------------------------------------------------------------
# Fixture: backend MediaPipe giả (không cần tflite checkpoint)
# ---------------------------------------------------------------------------


class _FakeMediaPipeBackend:
    """Giả lập MediaPipe backend trả về một khuôn mặt ở vị trí định sẵn."""

    def detect(self, _image: np.ndarray) -> list[FaceDetectionRecord]:
        return [_FACE_BBOX]

    def close(self) -> None:
        pass


class _FakeNoFaceBackend:
    """Giả lập backend không tìm thấy khuôn mặt."""

    def detect(self, _image: np.ndarray) -> list[FaceDetectionRecord]:
        return []

    def close(self) -> None:
        pass


@pytest.fixture()
def fake_face_backend(monkeypatch: pytest.MonkeyPatch):
    """Thay thế factory MediaPipe bằng backend giả, reset sau test."""
    reset_face_detector()
    monkeypatch.setattr(face_detector_module, "_DETECTOR_FACTORY", lambda _path: _FakeMediaPipeBackend())
    yield
    reset_face_detector()


@pytest.fixture()
def fake_no_face_backend(monkeypatch: pytest.MonkeyPatch):
    reset_face_detector()
    monkeypatch.setattr(face_detector_module, "_DETECTOR_FACTORY", lambda _path: _FakeNoFaceBackend())
    yield
    reset_face_detector()


# ===========================================================================
# 1. resolve_region — kind="full"
# ===========================================================================


def test_full_region_on_person_image_returns_ones_mask() -> None:
    """kind='full' phải trả mask toàn 1.0 bất kể nội dung ảnh."""
    result = resolve_region(_PERSON_IMAGE, {"kind": "full"})

    assert isinstance(result, RegionResult)
    assert result.status == "ok"
    assert result.mask.dtype == np.float32
    assert result.mask.shape == (_H, _W)
    np.testing.assert_array_equal(result.mask, np.ones((_H, _W), dtype=np.float32))
    assert result.metadata["backend"] == "geometry"


# ===========================================================================
# 2. resolve_region — kind="bbox"  (bao quanh người)
# ===========================================================================


def test_bbox_region_covers_person_body() -> None:
    """kind='bbox' với toạ độ thân người phải có mask=1 bên trong và =0 ngoài."""
    person_bbox = (220, 80, 420, 420)  # xmin, ymin, xmax, ymax
    result = resolve_region(
        _PERSON_IMAGE,
        {"kind": "bbox", "bbox": list(person_bbox), "feather_radius": 0},
    )

    assert result.status == "ok"
    assert result.mask.shape == (_H, _W)
    # Vùng trong bbox phải là 1.0
    assert np.all(result.mask[80:420, 220:420] == 1.0), "bbox interior must be 1.0"
    # Góc trên trái ngoài bbox phải là 0.0
    assert result.mask[0, 0] == 0.0
    assert result.mask[_H - 1, _W - 1] == 0.0


def test_bbox_region_with_feathering_has_smooth_boundary() -> None:
    """feather_radius > 0 phải tạo ra chuyển tiếp mềm tại biên bbox."""
    result = resolve_region(
        _PERSON_IMAGE,
        {"kind": "bbox", "bbox": [220, 80, 420, 420], "feather_radius": 15},
    )

    assert result.status == "ok"
    # Giá trị trên biên phải nằm giữa 0 và 1 (không phải bước nhảy cứng)
    boundary_col = result.mask[200, 218]  # pixel ngay ngoài biên trái
    assert 0.0 <= float(boundary_col) < 1.0, "boundary pixel must be in transition zone"


# ===========================================================================
# 3. resolve_region — kind="spatial"
# ===========================================================================


@pytest.mark.parametrize("quadrant", ["top", "bottom", "left", "right", "center"])
def test_spatial_region_on_person_image(quadrant: str) -> None:
    """Mỗi spatial quadrant phải tạo ra mask hợp lệ không cần backend."""
    result = resolve_region(
        _PERSON_IMAGE,
        RegionRequest(kind="spatial", quadrant=quadrant, feather_radius=0),
    )

    assert result.status == "ok"
    assert result.mask.dtype == np.float32
    assert result.mask.shape == (_H, _W)
    assert result.metadata["backend"] == "geometry"
    # Mask phải có ít nhất một pixel được chọn
    assert np.any(result.mask > 0.0), f"quadrant '{quadrant}' mask must select some pixels"


def test_spatial_top_and_bottom_are_complementary() -> None:
    """top + bottom phải bao phủ toàn ảnh (với feather_radius=0)."""
    top = resolve_region(_PERSON_IMAGE, RegionRequest(kind="spatial", quadrant="top", feather_radius=0))
    bot = resolve_region(_PERSON_IMAGE, RegionRequest(kind="spatial", quadrant="bottom", feather_radius=0))

    combined = top.mask + bot.mask
    np.testing.assert_array_equal(combined, np.ones((_H, _W), dtype=np.float32))


# ===========================================================================
# 4. resolve_region — kind="face"  (với backend giả)
# ===========================================================================


def test_face_region_detects_one_face_on_person_image(fake_face_backend) -> None:
    """Khi backend trả 1 khuôn mặt, result phải có count=1 và mask khác 0."""
    result = resolve_region(
        _PERSON_IMAGE,
        RegionRequest(kind="face", feather_radius=0, expand_ratio=0.0),
    )

    assert result.status == "ok"
    assert result.metadata["backend"] == "mediapipe"
    assert result.metadata["count"] == 1
    assert len(result.instance_masks) == 1

    # Vùng khuôn mặt (từ _FACE_BBOX) phải được mask
    face_mask = result.mask
    assert face_mask.dtype == np.float32
    assert face_mask.shape == (_H, _W)
    # Tâm khuôn mặt phải có giá trị 1.0
    face_center_y = int(_FACE_BBOX.y + _FACE_BBOX.height // 2)
    face_center_x = int(_FACE_BBOX.x + _FACE_BBOX.width // 2)
    assert face_mask[face_center_y, face_center_x] == 1.0, "face center must be fully masked"
    # Góc trên trái (nền) phải là 0.0
    assert face_mask[0, 0] == 0.0


def test_face_region_no_face_returns_empty_status(fake_no_face_backend) -> None:
    """Khi không phát hiện mặt, status phải là 'empty' và mask toàn 0."""
    result = resolve_region(_PERSON_IMAGE, RegionRequest(kind="face"))

    assert result.status == "empty"
    assert result.metadata["count"] == 0
    assert result.instance_masks == ()
    np.testing.assert_array_equal(result.mask, np.zeros((_H, _W), dtype=np.float32))


def test_face_region_expand_ratio_enlarges_mask(fake_face_backend) -> None:
    """expand_ratio lớn hơn phải tạo mask diện tích lớn hơn."""
    result_tight = resolve_region(
        _PERSON_IMAGE,
        RegionRequest(kind="face", feather_radius=0, expand_ratio=0.0),
    )
    result_expanded = resolve_region(
        _PERSON_IMAGE,
        RegionRequest(kind="face", feather_radius=0, expand_ratio=0.3),
    )

    area_tight = float(np.sum(result_tight.mask))
    area_expanded = float(np.sum(result_expanded.mask))
    assert area_expanded > area_tight, "larger expand_ratio must produce a larger mask area"


# ===========================================================================
# 5. resolve_region — kind="binary_mask"
# ===========================================================================


def test_binary_mask_from_person_region_is_feathered() -> None:
    """Cung cấp binary_mask thủ công (vùng thân người), soft-mask phải hợp lệ."""
    binary = np.zeros((_H, _W), dtype=np.uint8)
    binary[80:420, 220:420] = 255  # vùng người

    result = resolve_region(
        _PERSON_IMAGE,
        RegionRequest(kind="binary_mask", binary_mask=binary, feather_radius=21),
    )

    assert result.status == "ok"
    assert result.mask.dtype == np.float32
    assert result.mask.shape == (_H, _W)
    assert np.all(result.mask >= 0.0) and np.all(result.mask <= 1.0)
    # Tâm vùng người phải gần 1.0
    assert result.mask[250, 320] > 0.9


# ===========================================================================
# 6. blend_regions — ghép ảnh gốc và ảnh đã xử lý theo mask mặt
# ===========================================================================


def test_blend_face_region_replaces_only_face_pixels(fake_face_backend) -> None:
    """Sau blend, vùng nền phải bằng ảnh gốc, vùng mặt bằng ảnh xử lý."""
    processed = np.full_like(_PERSON_IMAGE, [255, 200, 50])  # màu vàng (hiệu ứng giả)

    result = resolve_region(
        _PERSON_IMAGE,
        RegionRequest(kind="face", feather_radius=0, expand_ratio=0.0),
    )
    blended = blend_regions(_PERSON_IMAGE, processed, result.mask)

    assert blended.dtype == np.uint8
    assert blended.shape == _PERSON_IMAGE.shape

    # Tâm khuôn mặt phải là màu đã xử lý
    cy = int(_FACE_BBOX.y + _FACE_BBOX.height // 2)
    cx = int(_FACE_BBOX.x + _FACE_BBOX.width // 2)
    np.testing.assert_array_equal(blended[cy, cx], [255, 200, 50])

    # Góc trên trái (nền) phải giữ nguyên ảnh gốc
    np.testing.assert_array_equal(blended[0, 0], _PERSON_IMAGE[0, 0])


def test_blend_full_mask_returns_fully_processed_image() -> None:
    """Mask toàn 1 (full region) phải cho output = processed hoàn toàn."""
    processed = np.full_like(_PERSON_IMAGE, [100, 150, 200])
    result = resolve_region(_PERSON_IMAGE, {"kind": "full"})
    blended = blend_regions(_PERSON_IMAGE, processed, result.mask)

    np.testing.assert_array_equal(blended, processed)


def test_blend_spatial_top_modifies_only_upper_half() -> None:
    """Spatial mask 'top' chỉ chỉnh sửa nửa trên ảnh, nửa dưới giữ nguyên."""
    processed = np.full_like(_PERSON_IMAGE, [10, 20, 30])
    result = resolve_region(
        _PERSON_IMAGE,
        RegionRequest(kind="spatial", quadrant="top", feather_radius=0),
    )
    blended = blend_regions(_PERSON_IMAGE, processed, result.mask)

    # Nửa trên phải là processed
    np.testing.assert_array_equal(blended[: _H // 2], processed[: _H // 2])
    # Nửa dưới phải là gốc
    np.testing.assert_array_equal(blended[_H // 2 :], _PERSON_IMAGE[_H // 2 :])


# ===========================================================================
# 7. Contract: tất cả mask phải thoả mãn spec Module 2
# ===========================================================================


@pytest.mark.parametrize(
    "request_dict",
    [
        {"kind": "full"},
        {"kind": "bbox", "bbox": [100, 50, 500, 400], "feather_radius": 10},
        {"kind": "spatial", "quadrant": "center", "feather_radius": 25},
        {"kind": "spatial", "quadrant": "left", "feather_radius": 0},
    ],
)
def test_mask_contract_float32_shape_and_range(request_dict: dict) -> None:
    """Mọi mask xuất ra phải: dtype=float32, shape=(H,W), giá trị trong [0,1]."""
    result = resolve_region(_PERSON_IMAGE, request_dict)

    assert result.mask.dtype == np.float32, "mask dtype must be float32"
    assert result.mask.shape == (_H, _W), f"mask shape must be ({_H}, {_W})"
    assert np.all(result.mask >= 0.0), "mask values must be >= 0"
    assert np.all(result.mask <= 1.0), "mask values must be <= 1"
    assert np.isfinite(result.mask).all(), "mask must not contain NaN or Inf"


def test_face_mask_contract_float32_shape_and_range(fake_face_backend) -> None:
    """Mask khuôn mặt phải thoả mãn spec: float32, shape (H,W), [0,1]."""
    result = resolve_region(_PERSON_IMAGE, RegionRequest(kind="face", feather_radius=15))

    assert result.mask.dtype == np.float32
    assert result.mask.shape == (_H, _W)
    assert np.all(result.mask >= 0.0)
    assert np.all(result.mask <= 1.0)
    assert np.isfinite(result.mask).all()


# ===========================================================================
# 8. Real-model test (chỉ chạy khi có checkpoint + ảnh fixture thực)
# ===========================================================================


@pytest.mark.real_model
def test_real_mediapipe_detects_face_in_portrait_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kiểm tra với model MediaPipe thật và ảnh portrait thực.

    Yêu cầu:
    - models/mediapipe/face_detection_full_range.tflite có mặt.
    - tests/fixtures/face_detection/upstream/portrait.jpg có mặt.
    Chạy bằng: pytest -m real_model tests/integration/test_person_image_processing.py
    """
    import json
    from pathlib import Path

    from PIL import Image as PILImage

    from src.region_engine.face_detector import MODEL_PATH_ENV

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "tests" / "fixtures" / "face_detection" / "assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_path = repo_root / manifest["model"]["path"]
    image_path = repo_root / manifest["real_model_cases"]["portrait_full_range"]["image"]
    expected_count = manifest["real_model_cases"]["portrait_full_range"]["expected_count"]

    missing = [str(p) for p in (model_path, image_path) if not p.exists()]
    if missing:
        pytest.fail("real assets missing: " + ", ".join(missing))

    with PILImage.open(image_path) as img:
        portrait = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()

    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    reset_face_detector()

    result = resolve_region(
        portrait,
        RegionRequest(kind="face", feather_radius=20, expand_ratio=0.15),
    )
    reset_face_detector()

    assert result.metadata["count"] == expected_count
    assert result.status == "ok"
    assert result.mask.dtype == np.float32
    assert result.mask.shape == portrait.shape[:2]
    assert np.any(result.mask > 0.0), "face region mask must cover at least some pixels"
