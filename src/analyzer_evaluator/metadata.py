"""
Đọc metadata của ảnh (EXIF, bảng lượng tử JPEG, ICC) và suy ra các prior chẩn đoán.

Chỉ dùng Pillow. Không bao giờ raise: bytes hỏng hoặc thiếu một phần thì các trường tương ứng
là None, phần còn lại vẫn được đọc. Prior là gợi ý cho VLM và chẩn đoán, không phải kết luận.
"""

import logging
import math
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

from PIL import Image, ImageCms
from pydantic import BaseModel

logger = logging.getLogger("img_doctor.analyzer_evaluator.metadata")

# Tag EXIF (IFD0 và Exif sub-IFD)
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_SOFTWARE = 0x0131
_EXIF_IFD = 0x8769
_TAG_EXPOSURE_TIME = 0x829A
_TAG_F_NUMBER = 0x829D
_TAG_ISO_SPEED_RATINGS = 0x8827
_TAG_ISO_SPEED = 0x8833
_TAG_FLASH = 0x9209
_TAG_FOCAL_LENGTH_35MM = 0xA405

# Bảng lượng tử chuẩn của libjpeg (IJG) cho bảng độ sáng, thứ tự hàng-cột (đúng thứ tự Pillow trả về)
_IJG_LUMA_BASE: tuple[int, ...] = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)  # fmt: skip

ISO_NOISE_THRESHOLD = 1600
BOKEH_MAX_F_NUMBER = 2.8
JPEG_NO_SHARPEN_QUALITY = 60
EDITED_PRIOR_STRENGTH = 0.5

PriorKind = Literal["noise", "motion_blur", "bokeh", "no_sharpen", "edited"]


class ImageMetadata(BaseModel):
    """Metadata đọc được từ file ảnh. Mọi trường Optional: không có thì None."""

    iso: Optional[int] = None
    exposure_time_s: Optional[float] = None
    f_number: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    flash_fired: Optional[bool] = None
    camera: Optional[str] = None
    software: Optional[str] = None
    jpeg_quality_estimate: Optional[int] = None  # 1..100, từ bảng lượng tử
    icc_profile_name: Optional[str] = None


class Prior(BaseModel):
    """Một gợi ý chẩn đoán suy ra từ metadata."""

    kind: PriorKind
    strength: float  # 0..1
    reason: str  # tiếng Việt, đưa thẳng vào prompt


def read_metadata(file_bytes: bytes) -> ImageMetadata:
    """Đọc metadata từ bytes của file ảnh; không bao giờ raise (lỗi → các trường None)."""
    try:
        img = Image.open(BytesIO(file_bytes))
    except Exception:
        logger.info("Image metadata unavailable: bytes are not a readable image")
        return ImageMetadata()

    fields: Dict[str, Any] = {}
    with img:
        for section in (_read_exif, _read_jpeg_quality, _read_icc):
            try:
                fields.update(section(img))
            except Exception:
                logger.warning("Skipping image metadata section %s", section.__name__)
    try:
        return ImageMetadata(**fields)
    except Exception:
        logger.warning("Image metadata failed validation; returning empty metadata")
        return ImageMetadata()


def metadata_priors(meta: ImageMetadata) -> List[Prior]:
    """Suy ra prior chẩn đoán từ metadata. Trường thiếu → không sinh prior tương ứng."""
    priors: List[Prior] = []

    if meta.iso is not None and meta.iso >= ISO_NOISE_THRESHOLD:
        # ISO 1600 → 0.4, mỗi lần gấp đôi cộng 0.2, tối đa 1.0
        strength = min(1.0, 0.4 + 0.2 * math.log2(meta.iso / ISO_NOISE_THRESHOLD))
        priors.append(
            Prior(
                kind="noise",
                strength=strength,
                reason=f"ISO {meta.iso} cao: ảnh có khả năng nhiễu hạt.",
            )
        )

    if meta.exposure_time_s and meta.focal_length_35mm:
        # Quy tắc 1/f: phơi chậm hơn 1/tiêu cự (35mm) thì dễ rung máy
        ratio = meta.exposure_time_s * meta.focal_length_35mm
        if ratio > 1.0:
            strength = min(1.0, 0.3 + 0.35 * math.log2(ratio))
            priors.append(
                Prior(
                    kind="motion_blur",
                    strength=strength,
                    reason=(
                        f"Thời gian phơi {meta.exposure_time_s:g}s chậm hơn 1/"
                        f"{meta.focal_length_35mm:g}s (tiêu cự 35mm {meta.focal_length_35mm:g}mm): "
                        "có thể rung máy hoặc mờ chuyển động."
                    ),
                )
            )

    if meta.f_number is not None and meta.f_number <= BOKEH_MAX_F_NUMBER:
        # f/2.8 → 0.4, f/1.4 trở xuống → 1.0
        strength = min(1.0, 0.4 + 0.6 * (BOKEH_MAX_F_NUMBER - meta.f_number) / 1.4)
        priors.append(
            Prior(
                kind="bokeh",
                strength=strength,
                reason=(
                    f"Khẩu độ f/{meta.f_number:g} rộng: nền mờ có thể là chủ ý, "
                    "không nên làm nét đều toàn bộ ảnh."
                ),
            )
        )

    if meta.jpeg_quality_estimate is not None and (
        meta.jpeg_quality_estimate < JPEG_NO_SHARPEN_QUALITY
    ):
        # Càng nén mạnh càng không nên làm sắc nét (khuếch đại artifact khối)
        strength = min(1.0, max(0.2, (JPEG_NO_SHARPEN_QUALITY - meta.jpeg_quality_estimate) / 30))
        priors.append(
            Prior(
                kind="no_sharpen",
                strength=strength,
                reason=(
                    f"Chất lượng JPEG ước lượng {meta.jpeg_quality_estimate} (dưới "
                    f"{JPEG_NO_SHARPEN_QUALITY}): làm sắc nét sẽ khuếch đại artifact nén."
                ),
            )
        )

    if meta.software:
        priors.append(
            Prior(
                kind="edited",
                strength=EDITED_PRIOR_STRENGTH,
                reason=(
                    f"Có ghi chú phần mềm '{meta.software}': ảnh có thể đã qua chỉnh sửa "
                    "trước khi tải lên."
                ),
            )
        )

    return priors


def _read_exif(img: Image.Image) -> Dict[str, Any]:
    """Đọc các trường EXIF (IFD0 và Exif sub-IFD); thiếu tag thì bỏ qua."""
    exif = img.getexif()
    if not exif:
        return {}
    ifd0 = dict(exif)
    sub = exif.get_ifd(_EXIF_IFD)

    fields: Dict[str, Any] = {}
    iso_raw = sub.get(_TAG_ISO_SPEED_RATINGS, sub.get(_TAG_ISO_SPEED))
    if isinstance(iso_raw, (tuple, list)):
        iso_raw = iso_raw[0] if iso_raw else None
    if _positive(iso_raw):
        fields["iso"] = int(iso_raw)
    if _positive(sub.get(_TAG_EXPOSURE_TIME)):
        fields["exposure_time_s"] = float(sub[_TAG_EXPOSURE_TIME])
    if _positive(sub.get(_TAG_F_NUMBER)):
        fields["f_number"] = float(sub[_TAG_F_NUMBER])
    if _positive(sub.get(_TAG_FOCAL_LENGTH_35MM)):
        fields["focal_length_35mm"] = float(sub[_TAG_FOCAL_LENGTH_35MM])
    if sub.get(_TAG_FLASH) is not None:
        fields["flash_fired"] = bool(int(sub[_TAG_FLASH]) & 0x1)

    camera = _camera_name(_clean_text(ifd0.get(_TAG_MAKE)), _clean_text(ifd0.get(_TAG_MODEL)))
    if camera:
        fields["camera"] = camera
    software = _clean_text(ifd0.get(_TAG_SOFTWARE))
    if software:
        fields["software"] = software
    return fields


def _read_jpeg_quality(img: Image.Image) -> Dict[str, Any]:
    """Ước lượng chất lượng JPEG bằng cách khớp bảng lượng tử độ sáng với bảng IJG."""
    if img.format != "JPEG":
        return {}
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return {}
    return {"jpeg_quality_estimate": estimate_jpeg_quality(list(tables[0]))}


def _read_icc(img: Image.Image) -> Dict[str, Any]:
    """Tên hồ sơ ICC (profile description), nếu có."""
    icc = img.info.get("icc_profile")
    if not icc:
        return {}
    name = ImageCms.ImageCmsProfile(BytesIO(icc)).profile.profile_description
    name = (name or "").strip()
    return {"icc_profile_name": name} if name else {}


def estimate_jpeg_quality(table: List[int]) -> int:
    """
    Tìm q ∈ [1, 100] mà bảng IJG tỉ lệ theo q khớp nhất với bảng lượng tử (64 phần tử, hàng-cột).
    Với file do libjpeg ghi, kết quả chính xác; file có bảng tuỳ chỉnh thì chỉ là ước lượng gần nhất.
    """
    best_quality, best_error = 1, math.inf
    for quality in range(1, 101):
        scale = 5000 // quality if quality < 50 else 200 - 2 * quality
        candidate = [min(255, max(1, (base * scale + 50) // 100)) for base in _IJG_LUMA_BASE]
        error = sum(abs(a - b) for a, b in zip(candidate, table))
        if error < best_error:
            best_quality, best_error = quality, error
    return best_quality


def _camera_name(make: Optional[str], model: Optional[str]) -> Optional[str]:
    """Ghép hãng và model; tránh lặp hãng khi model đã bắt đầu bằng tên hãng."""
    if model and make and not model.lower().startswith(make.lower()):
        return f"{make} {model}"
    return model or make or None


def _clean_text(value: Any) -> Optional[str]:
    """Chuỗi EXIF đã bỏ byte null và khoảng trắng thừa; rỗng → None."""
    if value is None:
        return None
    text = str(value).replace("\x00", "").strip()
    return text or None


def _positive(value: Any) -> bool:
    """True nếu giá trị đọc được là số dương (IFDRational, int hoặc float)."""
    try:
        return value is not None and float(value) > 0
    except (TypeError, ValueError):
        return False
