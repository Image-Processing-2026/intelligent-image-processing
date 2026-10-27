"""
Kiểm chứng chéo và hợp nhất chẩn đoán (Analysis v2 P6, plan §Phase 6).

Đầu vào là các GIẢ THUYẾT: lời VLM (Perceive, đã gộp self-consistency) và số đo đã hiệu chỉnh
(severity_from_measurement, C4) trên toàn ảnh và trên từng vùng. Bảng quyết định tất định ghép
hai nguồn, cộng preserve của VLM, prior EXIF và (khi bảng yêu cầu) câu trả lời của một lần gọi
Gemini `verify` với câu hỏi đóng, để ra kết luận cho từng giả thuyết:

| Số đo                     | VLM / verify                 | Kết luận                                   |
|---------------------------|------------------------------|--------------------------------------------|
| severity ≥ 1              | VLM báo lỗi                  | confirmed, severity = số đo (VLM ±1)        |
| ≥ 1, kém tin cậy (*)      | VLM báo lỗi                  | verify: defect → confirmed, severity = VLM  |
| severity ≥ 1              | VLM preserve (chủ ý)         | preserved; severity 3 trên chủ thể → ask_user |
| severity ≥ 1              | VLM không nhắc               | verify: defect → confirmed; còn lại → bỏ    |
| severity 0 (đo được)      | VLM báo lỗi                  | verify: not_present/texture → rejected      |
| không đo được             | VLM báo lỗi                  | confirmed, confidence = tỉ lệ đồng thuận    |
| bất kỳ                    | —                            | ý định người dùng thắng (apply_intent)      |

(*) Ám màu đo bằng đường dự phòng của bộ ước lượng nguồn sáng (`color.illuminant_method` ≥ 1:
cổng C* nới hoặc mọi pixel, contract-changes 9): số đo chỉ cho hướng, không đủ tin để đặt severity.

Lỗi đo được không bao giờ được phát ra một mình khi có VLM (brief fusion mục 11): phải có VLM
đồng ý, verify xác nhận, hoặc bị preserve chặn. Không có VLM (luật dự phòng, không mạng) thì
số đo là nguồn duy nhất: giả thuyết đo có confidence cao được xác nhận với confidence trung bình.

Module này không gọi mạng: `verifier` do bên gọi truyền vào (perception gọi Gemini), nên với
cùng đầu vào và cùng câu trả lời verify, kết quả luôn như nhau. Không sinh pixel (ADR-001).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from src.analyzer_evaluator.measurement import MeasurementReport, RegionStats
from src.analyzer_evaluator.measurement.calibration import severity_from_measurement
from src.analyzer_evaluator.metadata import Prior

from .regions import canonical_region
from .state import Defect, DiagnosisReport, Evidence, IntentProfile, PreserveItem

logger = logging.getLogger(__name__)

Stats = Union[RegionStats, MeasurementReport]
Verdict = Literal["defect", "intentional", "texture", "not_present"]
VERDICTS: Tuple[str, ...] = ("defect", "intentional", "texture", "not_present")

# ---------------------------------------------------------
# Từ vựng và hằng số của bảng quyết định
# ---------------------------------------------------------
# Lỗi toolbox 5 thao tác không sửa được (D2): báo người dùng, sinh contraindication, không lập action
NON_FIXABLE_DEFECTS = frozenset(("motion_blur", "jpeg_artifacts", "haze", "highlight_clipping"))
# Lỗi không sửa được chỉ đáng báo khi rõ (mức ≥ 2): mức nhẹ không có gì để làm, chỉ gây nhiễu
NON_FIXABLE_MIN_SEVERITY = 2
# Ám màu: số đo từ đường dự phòng (`illuminant_method` ≥ 1) kém tin cậy → hỏi verify kể cả khi
# VLM đồng ý, severity theo VLM (lane `color`, contract-changes 9)
COLOR_CAST_TYPES = frozenset(("color_cast_warm", "color_cast_cool", "color_cast_green"))
ILLUMINANT_METHOD_KEY = "illuminant_method"

# Lỗi mâu thuẫn với một đặc điểm cần giữ (VLM đôi khi liệt kê cả hai)
CONFLICTING_DEFECTS: Dict[str, frozenset] = {
    "warm_tone": frozenset(("color_cast_warm",)),
    "cool_tone": frozenset(("color_cast_cool",)),
    "low_key": frozenset(("underexposed", "backlit_subject")),
    "silhouette": frozenset(("underexposed", "backlit_subject")),
    "high_key": frozenset(("overexposed", "highlight_clipping")),
    "film_grain": frozenset(("noise",)),
    "soft_focus": frozenset(("blur", "motion_blur")),
    "muted_colors": frozenset(("undersaturated", "low_contrast", "haze")),
    "vivid_colors": frozenset(("oversaturated",)),
}
# Verify trả "intentional" → đặc điểm cần giữ tương ứng (để planner/preserve guard chặn thao tác)
INTENTIONAL_ASPECT: Dict[str, str] = {
    "underexposed": "low_key",
    "backlit_subject": "silhouette",
    "overexposed": "high_key",
    "highlight_clipping": "high_key",
    "color_cast_warm": "warm_tone",
    "color_cast_cool": "cool_tone",
    "noise": "film_grain",
    "blur": "soft_focus",
    "motion_blur": "soft_focus",
    "oversaturated": "vivid_colors",
    "undersaturated": "muted_colors",
    "low_contrast": "muted_colors",
}
# Hai loại lỗi làm tối chủ thể được coi là cùng một giả thuyết khi ghép VLM với số đo
DARK_FAMILY = frozenset(("underexposed", "backlit_subject"))

# Lỗi đo trên toàn ảnh được đưa thành giả thuyết đo (khi VLM không nhắc). highlight_clipping
# toàn ảnh không có ở đây: trời/đèn cháy sau chủ thể là bình thường (prompt Perceive rev 2);
# cháy sáng chỉ thành giả thuyết đo trên vùng chủ thể.
FULL_MEASURED_TYPES: Tuple[str, ...] = (
    "noise",
    "blur",
    "motion_blur",
    "underexposed",
    "overexposed",
    "low_contrast",
    "color_cast_warm",
    "color_cast_cool",
    "color_cast_green",
    "oversaturated",
    "undersaturated",
    "haze",
    "jpeg_artifacts",
)
# Lỗi đo được trên vùng chủ thể được đưa thành giả thuyết đo (khi VLM không nhắc)
SUBJECT_MEASURED_TYPES: Tuple[str, ...] = ("backlit_subject", "overexposed", "highlight_clipping")
# Lỗi đo được trên RegionStats (chỉ có các key MAP_KEYS: L*, clip, σ nhiễu, mờ, chroma)
REGION_MEASURABLE_TYPES = frozenset(
    ("noise", "blur", "underexposed", "overexposed", "oversaturated", "highlight_clipping")
)
# Lỗi mang tính toàn cục: lời VLM trên một vùng chưa định vị được đo bằng báo cáo toàn ảnh
GLOBAL_TYPES = frozenset(
    (
        "color_cast_warm",
        "color_cast_cool",
        "color_cast_green",
        "jpeg_artifacts",
        "haze",
        "motion_blur",
        "low_contrast",
    )
)
# Vùng là phông/cảnh (không phải chủ thể): không đo ngược sáng, không đo cháy sáng chủ thể
BACKDROP_REGIONS = frozenset(
    (
        "full",
        "background",
        "sky",
        "clouds",
        "sun",
        "sunset",
        "horizon",
        "ground",
        "field",
        "grass",
        "land",
        "floor",
        "foreground",
        "water",
        "sea",
        "ocean",
        "lake",
        "river",
        "beach",
        "sand",
        "snow",
        "mountain",
        "mountains",
        "hill",
        "hills",
        "trees",
        "tree",
        "forest",
        "road",
        "street",
        "wall",
        "city",
        "cityscape",
        "landscape",
        "bokeh",
        "backdrop",
        "light",
        "lights",
        "window",
        "top",
        "bottom",
        "left",
        "right",
        "center",
    )
)
# Chủ thể chiếm gần hết khung thì "so với nền" không còn ý nghĩa
SUBJECT_MAX_AREA = 0.6
SUBJECT_MIN_AREA = 0.001

# Ngược sáng đo được (calibration không khớp được lỗi hai vùng, C4: backlit → (0, 0.0)):
# chủ thể tối hơn nền ít nhất BACKLIT_MIN_GAP_L (L*) và bản thân tối theo thang phơi sáng.
# Khuôn mặt có độ sáng "đúng" tuyệt đối (da ~ L* 60–70) nên dùng mốc riêng FACE_MID_L thay cho
# mức xám giữa. Đây là ngưỡng prior, không khớp trên dữ liệu có nhãn.
BACKLIT_MIN_GAP_L = 20.0
BACKLIT_SEVERITY_L: Tuple[float, float, float] = (12.0, 22.0, 34.0)
FACE_MID_L = 60.0
SUBJECT_MID_L = 50.0
BACKLIT_PRIOR_CONFIDENCE = 0.6

# Confidence "cao" của số đo: đủ để xác nhận một mình khi không có VLM (luật dự phòng)
HIGH_CONFIDENCE = 0.5
# Confidence gán cho lỗi chỉ được một nguồn ủng hộ (số đo hoặc verify)
MEDIUM_CONFIDENCE = 0.6
# Lời VLM không kiểm chứng được (verify không chạy được) trên lỗi đo ra 0: giữ, hạ confidence
UNVERIFIED_VLM_SCALE = 0.5
# Prior EXIF chỉ dịch confidence: confidence ± strength × PRIOR_WEIGHT
PRIOR_WEIGHT = 0.2
PRIOR_NO_SHARPEN_STRENGTH = 0.5
# Self-consistency: dùng giả thuyết bị loại (dưới ngưỡng đồng thuận) làm bằng chứng yếu.
# Mặc định tắt; benchmark bật để đánh giá trên hold-out (brief mục 12).
USE_WEAK_REJECTED = False
USE_WEAK_REJECTED_ENV = "FUSION_WEAK_REJECTED"

# Prior EXIF → loại lỗi bị dịch confidence (dấu: + làm chắc hơn, − làm kém chắc hơn)
PRIOR_EFFECTS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "noise": (("noise", 1.0),),
    "motion_blur": (("motion_blur", 1.0), ("blur", 1.0)),
    "bokeh": (("blur", -1.0),),
    "no_sharpen": (("jpeg_artifacts", 1.0),),
    "edited": (
        ("color_cast_warm", -1.0),
        ("color_cast_cool", -1.0),
        ("color_cast_green", -1.0),
        ("oversaturated", -1.0),
        ("undersaturated", -1.0),
        ("low_contrast", -1.0),
    ),
}

# Câu hỏi đóng cho verify: (mô tả hiện tượng, phương án "intentional", phương án "texture")
VERIFY_PHRASES: Dict[str, Tuple[str, str, str]] = {
    "noise": ("có hạt/nhiễu", "hạt film có chủ ý", "texture thật của bề mặt (vải, cỏ, đá…)"),
    "blur": (
        "mềm/thiếu nét",
        "xóa phông hoặc soft focus có chủ ý",
        "bề mặt vốn mịn, không có cạnh",
    ),
    "motion_blur": ("mờ theo một hướng (rung/chuyển động)", "lia máy/chuyển động có chủ ý", "—"),
    "underexposed": ("tối", "low-key/không khí tối có chủ ý", "—"),
    "overexposed": ("sáng quá, mất chi tiết vùng sáng", "high-key có chủ ý", "—"),
    "backlit_subject": (
        "tối hơn hẳn nền sáng phía sau (ngược sáng)",
        "silhouette/bóng đen có chủ ý",
        "—",
    ),
    "low_contrast": ("phẳng, thiếu tương phản", "tông nhạt/mờ ảo có chủ ý", "—"),
    "color_cast_warm": (
        "ngả vàng/cam",
        "không khí ấm có chủ ý (hoàng hôn, đèn vàng)",
        "màu thật của vật (gỗ, da, cát)",
    ),
    "color_cast_cool": (
        "ngả xanh lam",
        "không khí lạnh có chủ ý (tuyết, giờ xanh)",
        "màu thật của vật (trời, nước)",
    ),
    "color_cast_green": ("ngả xanh lá/tím", "tông màu có chủ ý", "màu thật của vật (cây cỏ)"),
    "oversaturated": ("màu rực quá mức, bệt màu", "màu rực có chủ ý", "màu thật của vật"),
    "undersaturated": ("nhạt màu", "tông nhạt kiểu film có chủ ý", "vật vốn ít màu"),
    "haze": ("mờ sương, bạc màu, thiếu tương phản xa", "sương/khói có chủ ý", "—"),
    "jpeg_artifacts": ("có khối vuông/răng cưa do nén JPEG", "—", "—"),
    "highlight_clipping": ("cháy trắng mất chi tiết", "high-key có chủ ý", "—"),
}

# Contraindication (planner đọc): lỗi → điều cần tránh khi xử lý
CONTRAINDICATIONS: Dict[str, Tuple[Tuple[int, str], ...]] = {
    # Làm nét khuếch đại khối JPEG và tạo viền quanh vệt mờ chuyển động
    "jpeg_artifacts": ((1, "no_sharpen"),),
    "motion_blur": ((2, "no_sharpen"),),
    # Làm sáng toàn ảnh đẩy thêm vùng đã cháy
    "highlight_clipping": ((2, "no_brighten"),),
}

# Trạng thái kết luận của một giả thuyết
Outcome = Literal["confirmed", "preserved", "rejected", "ask_user"]


class VerifyQuestion(BaseModel):
    """Một câu hỏi đóng gửi Gemini verify (plan §Phase 6)."""

    hypothesis_id: str
    defect_type: str
    region: str
    question: str
    crops: List[str] = Field(default_factory=list)


class VerifyAnswer(BaseModel):
    """Câu trả lời verify cho một giả thuyết."""

    hypothesis_id: str
    verdict: Verdict
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""


Verifier = Callable[[List[VerifyQuestion]], Dict[str, VerifyAnswer]]


@dataclass
class Measured:
    """Severity đo được của một (loại lỗi, vùng); None = không đo được."""

    severity: int
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    # True: số đo từ đường dự phòng (ám màu không có pixel trung tính), không tự đặt severity
    low_trust: bool = False


@dataclass
class Hypothesis:
    """Một giả thuyết đang được hợp nhất."""

    id: str
    type: str
    region: str
    measured: Optional[Measured]
    vlm: Optional[Defect] = None
    weak_vlm: Optional[Defect] = None
    prior_shift: float = 0.0

    @property
    def key(self) -> Tuple[str, str]:
        return self.type, self.region


VERIFY_RESPONSE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": ["hypothesis_id", "verdict"],
            },
        }
    },
    "required": ["answers"],
}

VERIFY_PROMPT = """
Bạn là "AI Image Doctor" - BƯỚC KIỂM CHỨNG. Hệ thống đo và bước quan sát đã đưa ra một số giả
thuyết về lỗi kỹ thuật của ảnh. Với MỖI câu hỏi (mã h1, h2, …), nhìn ẢNH TOÀN KHUNG và các CROP
100 % được nêu, rồi trả lời bằng đúng một verdict:
- "defect": lỗi kỹ thuật THẬT, nhìn thấy rõ, người xem bình thường muốn sửa.
- "intentional": hiện tượng có thật nhưng là chủ ý thẩm mỹ của người chụp (giữ nguyên).
- "texture": thứ trông giống lỗi nhưng là bản chất của vật thể/cảnh (texture, màu thật của vật).
- "not_present": không thấy hiện tượng này ở mức đáng kể.
Số đo có thể sai trên nội dung đặc biệt (cỏ, hoàng hôn, tường màu, cảnh đêm); ẢNH là căn cứ chính.
Trời, đèn hay nguồn sáng cháy phía sau chủ thể là bình thường, không phải lỗi. Ảnh đêm, ảnh tối
tâm trạng, hoàng hôn ấm, ảnh sáng nhẹ high-key thường là chủ ý. Chỉ trả "defect" khi chắc chắn.
confidence (0–1) là độ chắc của bạn. reason: một câu tiếng Việt có dấu.
Trả lời đủ mọi mã câu hỏi.
"""


def _weak_rejected_enabled() -> bool:
    """Bật dùng giả thuyết bị loại làm bằng chứng yếu (env ghi đè hằng số module)."""
    raw = os.getenv(USE_WEAK_REJECTED_ENV, "").strip().casefold()
    if raw:
        return raw in ("1", "true", "yes", "on")
    return USE_WEAK_REJECTED


# ---------------------------------------------------------
# Đo một giả thuyết
# ---------------------------------------------------------
def _measured_evidence(defect_type: str, stats: Stats, region: str) -> List[Evidence]:
    """Số đo chính dùng để kết luận (để giải thích trong report)."""
    keys = {
        "noise": "noise.sigma_luma",
        "blur": "sharpness.defocus_sigma",
        "motion_blur": "sharpness.motion_extent_sigma",
        "underexposed": "exposure.mean_l",
        "overexposed": "exposure.mean_l",
        "low_contrast": "exposure.p95_l",
        "color_cast_warm": "color.illuminant_angle_deg",
        "color_cast_cool": "color.illuminant_angle_deg",
        "color_cast_green": "color.illuminant_angle_deg",
        "oversaturated": "color.chroma_p50",
        "undersaturated": "color.chroma_p50",
        "haze": "haze.dark_channel_median",
        "jpeg_artifacts": "compression.blockiness",
        "highlight_clipping": "exposure.largest_clip_blob",
    }
    key = keys.get(defect_type)
    if key is None:
        return []
    if isinstance(stats, RegionStats):
        key = {
            "color.chroma_p50": "color.chroma",
            "exposure.largest_clip_blob": "exposure.clip_high",
        }.get(key, key)
        estimate = stats.values.get(key)
    else:
        section, name = key.split(".", 1)
        estimate = getattr(stats, section, {}).get(name)
    if estimate is None:
        return []
    return [
        Evidence(
            source="measured",
            metric=f"{region}:{key}",
            value=round(float(estimate.value), 3),
            text=f"tin cậy {estimate.confidence:.2f}",
        )
    ]


def _calibrated(
    defect_type: str, stats: Stats, region: str, long_side: Optional[int]
) -> Optional[Measured]:
    """severity_from_measurement (C4); (0, 0.0) = không đo được → None."""
    try:
        severity, confidence = severity_from_measurement(defect_type, stats, long_side=long_side)
    except Exception as exc:  # bảng hiệu chỉnh lỗi không được làm hỏng chẩn đoán
        logger.warning("Calibration failed for %s on %s: %s", defect_type, region, exc)
        return None
    if severity == 0 and confidence <= 0.0:
        return None
    return Measured(
        severity,
        confidence,
        _measured_evidence(defect_type, stats, region),
        low_trust=defect_type in COLOR_CAST_TYPES and _fallback_illuminant(stats),
    )


def _fallback_illuminant(stats: Stats) -> bool:
    """Nguồn sáng ước lượng bằng đường dự phòng (cổng C* nới hoặc mọi pixel).

    Chỉ MeasurementReport có key này (không thuộc MAP_KEYS); RegionStats → False.
    """
    if not isinstance(stats, MeasurementReport):
        return False
    method = stats.color.get(ILLUMINANT_METHOD_KEY)
    return method is not None and float(method.value) >= 1.0


def is_subject_region(region: str) -> bool:
    """Vùng là chủ thể (người, mặt, vật chính) chứ không phải phông/cảnh."""
    return canonical_region(region) not in BACKDROP_REGIONS


def backlit_measurement(
    region: str, stats: RegionStats, background: Optional[RegionStats]
) -> Optional[Measured]:
    """
    Ngược sáng đo được trên một vùng chủ thể: severity theo độ tối của chủ thể so với mốc
    (FACE_MID_L cho mặt, SUBJECT_MID_L cho chủ thể khác), chỉ khi nền sáng hơn ít nhất
    BACKLIT_MIN_GAP_L. Thiếu số đo nền hoặc chủ thể quá lớn/nhỏ → None (không đo được).
    """
    if background is None or not is_subject_region(region):
        return None
    if not SUBJECT_MIN_AREA <= stats.area_fraction <= SUBJECT_MAX_AREA:
        return None
    subject_l = stats.values.get("exposure.mean_l")
    backdrop_l = background.values.get("exposure.mean_l")
    if subject_l is None or backdrop_l is None:
        return None
    gap = backdrop_l.value - subject_l.value
    mid = FACE_MID_L if canonical_region(region) == "face" else SUBJECT_MID_L
    darkness = mid - subject_l.value
    severity = (
        sum(1 for t in BACKLIT_SEVERITY_L if darkness >= t) if gap >= BACKLIT_MIN_GAP_L else 0
    )
    confidence = (
        BACKLIT_PRIOR_CONFIDENCE
        * min(subject_l.confidence, backdrop_l.confidence)
        * min(1.0, stats.mask_confidence + 0.1)
    )
    evidence = [
        Evidence(
            source="measured", metric=f"{region}:exposure.mean_l", value=round(subject_l.value, 1)
        ),
        Evidence(
            source="measured",
            metric="background:exposure.mean_l",
            value=round(backdrop_l.value, 1),
            text=f"nền sáng hơn {gap:.0f} L*",
        ),
    ]
    return Measured(severity, round(confidence, 3), evidence)


def measure_hypothesis(
    defect_type: str,
    region: str,
    measurement: Optional[MeasurementReport],
    region_stats: Dict[str, RegionStats],
    long_side: Optional[int] = None,
) -> Optional[Measured]:
    """
    Severity đo được của một (loại lỗi, vùng), None nếu không đo được:
    - toàn ảnh → MeasurementReport;
    - vùng có RegionStats → số đo của vùng (chỉ các loại REGION_MEASURABLE_TYPES);
    - backlit_subject → so chủ thể với nền ('background');
    - lỗi toàn cục (ám màu, nén, sương…) trên vùng → báo cáo toàn ảnh.
    """
    region = canonical_region(region)
    if defect_type == "backlit_subject":
        stats = region_stats.get(region)
        if stats is None:
            return None
        return backlit_measurement(region, stats, region_stats.get("background"))
    if region == "full" or (defect_type in GLOBAL_TYPES and region not in region_stats):
        if measurement is None:
            return None
        return _calibrated(defect_type, measurement, "full", long_side)
    stats = region_stats.get(region)
    if stats is None or defect_type not in REGION_MEASURABLE_TYPES:
        if defect_type in GLOBAL_TYPES and measurement is not None:
            return _calibrated(defect_type, measurement, "full", long_side)
        return None
    return _calibrated(defect_type, stats, region, long_side)


# ---------------------------------------------------------
# Ghép giả thuyết
# ---------------------------------------------------------
def _same_family(first: str, second: str) -> bool:
    return first == second or {first, second} <= DARK_FAMILY


def _mentions(defect: Defect, defect_type: str, region: str) -> bool:
    """Lời VLM có nói về giả thuyết đo này không (cùng loại/họ; cùng vùng hoặc một bên toàn ảnh)."""
    if not _same_family(defect.type, defect_type):
        return False
    vlm_region = canonical_region(defect.region)
    return vlm_region == region or "full" in (vlm_region, region)


def _preserving_aspect(preserve: Sequence[PreserveItem], defect_type: str, region: str) -> str:
    """Aspect cần giữ mâu thuẫn với lỗi trên vùng này (preserve toàn ảnh áp cho mọi vùng)."""
    for item in preserve:
        item_region = canonical_region(item.region)
        in_scope = item_region in (region, "full") or region == "full"
        if in_scope and defect_type in CONFLICTING_DEFECTS.get(item.aspect, ()):
            return item.aspect
    return ""


def _prior_shift(defect_type: str, priors: Sequence[Prior]) -> float:
    """Độ dịch confidence do prior EXIF (không tạo lỗi mới)."""
    shift = 0.0
    for prior in priors:
        for affected, sign in PRIOR_EFFECTS.get(prior.kind, ()):
            if affected == defect_type:
                shift += sign * PRIOR_WEIGHT * float(prior.strength)
    return shift


def collect_hypotheses(
    report: DiagnosisReport,
    measurement: Optional[MeasurementReport],
    region_stats: Dict[str, RegionStats],
    priors: Sequence[Prior] = (),
    long_side: Optional[int] = None,
    use_weak: bool = False,
    full_image: bool = True,
) -> List[Hypothesis]:
    """
    Tập giả thuyết: mỗi lời VLM (kèm số đo của đúng loại/vùng), rồi các lỗi đo được mà VLM không
    nhắc (toàn ảnh: FULL_MEASURED_TYPES khi full_image; vùng chủ thể: SUBJECT_MEASURED_TYPES).
    Thứ tự tất định: lời VLM theo thứ tự báo, rồi số đo theo thứ tự loại lỗi và vùng.
    """
    hypotheses: List[Hypothesis] = []
    seen: set = set()

    def add(defect_type: str, region: str, vlm: Optional[Defect] = None) -> None:
        region = canonical_region(region)
        if (defect_type, region) in seen:
            return
        seen.add((defect_type, region))
        hypotheses.append(
            Hypothesis(
                id=f"h{len(hypotheses) + 1}",
                type=defect_type,
                region=region,
                measured=measure_hypothesis(
                    defect_type, region, measurement, region_stats, long_side
                ),
                vlm=vlm,
                prior_shift=_prior_shift(defect_type, priors),
            )
        )

    claims = list(report.defects)
    for defect in claims:
        add(defect.type, defect.region, defect)

    candidates: List[Tuple[str, str]] = []
    if measurement is not None and full_image:
        candidates += [(t, "full") for t in FULL_MEASURED_TYPES]
    # Có khuôn mặt thì chỉ đo phơi sáng chủ thể trên mặt: thân người tối (áo, bóng) còn mặt
    # sáng đều không phải ngược sáng (backlit_04, already_good_03 trên bộ thật)
    subject_regions = [
        region
        for region in region_stats
        if region not in ("full", "background") and is_subject_region(region)
    ]
    if "face" in subject_regions:
        subject_regions = ["face"]
    for region in subject_regions:
        candidates += [(t, region) for t in SUBJECT_MEASURED_TYPES]

    weak = list(report.rejected) if use_weak else []
    for defect_type, region in candidates:
        if any(_mentions(claim, defect_type, region) for claim in claims):
            continue
        measured = measure_hypothesis(defect_type, region, measurement, region_stats, long_side)
        if measured is None or measured.severity < 1:
            continue
        if (defect_type, region) in seen:
            continue
        seen.add((defect_type, region))
        hypotheses.append(
            Hypothesis(
                id=f"h{len(hypotheses) + 1}",
                type=defect_type,
                region=region,
                measured=measured,
                weak_vlm=next((d for d in weak if _mentions(d, defect_type, region)), None),
                prior_shift=_prior_shift(defect_type, priors),
            )
        )
    return hypotheses


# ---------------------------------------------------------
# Câu hỏi verify
# ---------------------------------------------------------
def _region_text(region: str) -> str:
    return "toàn ảnh" if region == "full" else f"vùng '{region}'"


def _measured_text(hypothesis: Hypothesis) -> str:
    if hypothesis.measured is None:
        return ""
    values = [
        f"{item.metric.split(':', 1)[-1]} = {item.value:g}"
        for item in hypothesis.measured.evidence
        if item.value is not None
    ]
    return f" (số đo: {', '.join(values)})" if values else ""


def verify_question(hypothesis: Hypothesis, crops: Sequence[str] = ()) -> VerifyQuestion:
    """Câu hỏi đóng cho một giả thuyết, liệt kê các phương án verdict theo loại lỗi."""
    phenomenon, intentional, texture = VERIFY_PHRASES.get(
        hypothesis.type, (hypothesis.type, "chủ ý thẩm mỹ", "bản chất của vật")
    )
    source = "VLM quan sát thấy" if hypothesis.vlm is not None else "Số đo cho thấy"
    options = ["defect = lỗi cần sửa"]
    if intentional != "—":
        options.append(f"intentional = {intentional}")
    if texture != "—":
        options.append(f"texture = {texture}")
    options.append("not_present = không thấy rõ")
    crop_text = f" Xem crop {', '.join(crops)}." if crops else ""
    question = (
        f"{source} {_region_text(hypothesis.region)} {phenomenon}{_measured_text(hypothesis)}."
        f"{crop_text} Đây là gì? ({'; '.join(options)})"
    )
    return VerifyQuestion(
        hypothesis_id=hypothesis.id,
        defect_type=hypothesis.type,
        region=hypothesis.region,
        question=question,
        crops=list(crops),
    )


def parse_verify_answers(data: object) -> Dict[str, VerifyAnswer]:
    """JSON của Gemini verify → câu trả lời theo mã giả thuyết; phần tử sai bị bỏ."""
    answers: Dict[str, VerifyAnswer] = {}
    raw = data.get("answers") if isinstance(data, dict) else None
    for item in raw if isinstance(raw, list) else []:
        try:
            answer = VerifyAnswer(**item)
        except Exception as exc:
            logger.warning("Dropping invalid verify answer %r: %s", item, exc)
            continue
        answers.setdefault(answer.hypothesis_id, answer)
    return answers


def verify_prompt_text(questions: Sequence[VerifyQuestion]) -> str:
    """Khối câu hỏi gửi Gemini verify."""
    lines = ["CÂU HỎI KIỂM CHỨNG:"]
    lines += [f"- {question.hypothesis_id}: {question.question}" for question in questions]
    return "\n".join(lines)


# ---------------------------------------------------------
# Bảng quyết định
# ---------------------------------------------------------
@dataclass
class _Decision:
    outcome: Outcome
    severity: int
    confidence: float
    note: str
    verdict: Optional[VerifyAnswer] = None


def _combined_severity(measured: int, vlm: Defect) -> int:
    """Severity = số đo; VLM chỉ kéo ±1 khi có nêu lý do (bảng P6 dòng 1)."""
    if not vlm.evidence.strip():
        return measured
    return max(1, min(3, measured + max(-1, min(1, vlm.severity - measured))))


def _clip(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _needs_verify(hypothesis: Hypothesis, preserved: str, vlm_mode: bool, use_weak: bool) -> bool:
    """Bảng yêu cầu hỏi verify cho giả thuyết này không."""
    if not vlm_mode or preserved:
        return False
    measured = hypothesis.measured
    if hypothesis.type in NON_FIXABLE_DEFECTS:
        # Lỗi không sửa được ở mức nhẹ bị bỏ dù verify trả lời gì: không tốn câu hỏi
        claimed = hypothesis.vlm.severity if hypothesis.vlm is not None else 0
        if max(claimed, measured.severity if measured else 0) < NON_FIXABLE_MIN_SEVERITY:
            return False
    if hypothesis.vlm is not None:
        return measured is not None and (measured.severity == 0 or measured.low_trust)
    if measured is None or measured.severity < 1:
        return False
    return not (use_weak and hypothesis.weak_vlm is not None)


def _decide(
    hypothesis: Hypothesis,
    preserved: str,
    vlm_mode: bool,
    answer: Optional[VerifyAnswer],
    use_weak: bool,
) -> _Decision:
    """Một dòng của bảng quyết định (tất định theo đầu vào)."""
    measured, vlm = hypothesis.measured, hypothesis.vlm
    m_sev = measured.severity if measured is not None else None
    m_conf = measured.confidence if measured is not None else 0.0
    shift = hypothesis.prior_shift

    # Preserve thắng số đo và lời VLM; nặng trên chủ thể → hỏi người dùng
    if preserved:
        severity = max(m_sev or 0, vlm.severity if vlm is not None else 0)
        if severity >= 3 and is_subject_region(hypothesis.region):
            return _Decision("ask_user", severity, _clip(m_conf + shift), f"preserve {preserved}")
        return _Decision("preserved", severity, _clip(m_conf + shift), f"preserve {preserved}")

    if vlm is not None:
        if m_sev is None:
            # Không đo được: tin lời VLM theo tỉ lệ đồng thuận
            return _Decision(
                "confirmed", max(1, vlm.severity), _clip(vlm.confidence + shift), "vlm, unmeasured"
            )
        if m_sev >= 1 and measured.low_trust and vlm_mode:
            # Số đo dự phòng chỉ cho hướng: verify quyết định, severity theo VLM
            if answer is None:
                return _Decision(
                    "confirmed",
                    max(1, vlm.severity),
                    _clip(vlm.confidence * UNVERIFIED_VLM_SCALE + shift),
                    "vlm + low-trust measured, unverified",
                )
            if answer.verdict == "defect":
                return _Decision(
                    "confirmed",
                    max(1, vlm.severity),
                    _clip(vlm.confidence * answer.confidence + shift),
                    "vlm + low-trust measured, verify defect",
                    answer,
                )
            outcome = "preserved" if answer.verdict == "intentional" else "rejected"
            return _Decision(
                outcome, vlm.severity, _clip(answer.confidence), f"verify {answer.verdict}", answer
            )
        if m_sev >= 1:
            confidence = 1.0 - (1.0 - m_conf) * (1.0 - vlm.confidence)
            return _Decision(
                "confirmed",
                _combined_severity(m_sev, vlm),
                _clip(confidence + shift),
                "vlm + measured agree",
            )
        # Đo được 0 mà VLM báo lỗi: phải kiểm chứng
        if answer is not None:
            if answer.verdict == "defect":
                return _Decision(
                    "confirmed",
                    max(1, vlm.severity),
                    _clip(vlm.confidence * answer.confidence + shift),
                    "vlm, measured 0, verify defect",
                    answer,
                )
            outcome: Outcome = "preserved" if answer.verdict == "intentional" else "rejected"
            return _Decision(
                outcome, vlm.severity, _clip(answer.confidence), f"verify {answer.verdict}", answer
            )
        if vlm_mode:
            # verify không chạy được hoặc bỏ sót câu hỏi: ảnh là căn cứ chính, giữ lời VLM
            return _Decision(
                "confirmed",
                max(1, vlm.severity),
                _clip(vlm.confidence * UNVERIFIED_VLM_SCALE + shift),
                "vlm, measured 0, unverified",
            )
        return _Decision("rejected", vlm.severity, 0.0, "vlm, measured 0, no verify answer")

    # VLM không nhắc: giả thuyết thuần số đo (severity ≥ 1)
    assert m_sev is not None
    if use_weak and hypothesis.weak_vlm is not None:
        weak = hypothesis.weak_vlm
        confidence = 1.0 - (1.0 - m_conf) * (1.0 - weak.confidence)
        return _Decision(
            "confirmed",
            _combined_severity(m_sev, weak),
            _clip(confidence + shift),
            "measured + weak vlm",
        )
    if not vlm_mode:
        if m_conf + shift >= HIGH_CONFIDENCE:
            return _Decision(
                "confirmed",
                m_sev,
                _clip(min(m_conf, MEDIUM_CONFIDENCE) + shift),
                "measured, no vlm",
            )
        return _Decision("rejected", m_sev, _clip(m_conf + shift), "measured, low confidence")
    if answer is not None:
        if answer.verdict == "defect":
            return _Decision(
                "confirmed",
                m_sev,
                _clip(min(m_conf, MEDIUM_CONFIDENCE) * answer.confidence + shift),
                "measured, verify defect",
                answer,
            )
        outcome = "preserved" if answer.verdict == "intentional" else "rejected"
        return _Decision(
            outcome, m_sev, _clip(answer.confidence), f"verify {answer.verdict}", answer
        )
    return _Decision("rejected", m_sev, _clip(m_conf + shift), "measured only, unverified")


def _evidence_for(hypothesis: Hypothesis, decision: _Decision) -> List[Evidence]:
    items: List[Evidence] = []
    if hypothesis.vlm is not None:
        items += hypothesis.vlm.evidence_items
    if hypothesis.measured is not None:
        items += hypothesis.measured.evidence
        items.append(
            Evidence(
                source="measured",
                metric="calibrated_severity",
                value=float(hypothesis.measured.severity),
                text=f"tin cậy {hypothesis.measured.confidence:.2f}",
            )
        )
    if decision.verdict is not None:
        items.append(
            Evidence(
                source="vlm",
                metric="verify",
                value=decision.verdict.confidence,
                text=f"{decision.verdict.verdict}: {decision.verdict.reason}".strip(),
            )
        )
    if hypothesis.prior_shift:
        items.append(Evidence(source="exif", metric="prior_shift", value=hypothesis.prior_shift))
    items.append(Evidence(source="measured", metric="fusion", text=decision.note))
    return items


def _contraindications(defect_type: str, severity: int) -> List[str]:
    return [name for minimum, name in CONTRAINDICATIONS.get(defect_type, ()) if severity >= minimum]


def _evidence_text(hypothesis: Hypothesis, decision: _Decision) -> str:
    if hypothesis.vlm is not None and hypothesis.vlm.evidence:
        return hypothesis.vlm.evidence
    parts = [f"Đo được{_measured_text(hypothesis)}"]
    if decision.verdict is not None and decision.verdict.reason:
        parts.append(f"Kiểm chứng: {decision.verdict.reason}")
    return ". ".join(parts) + "."


def fuse(
    hypotheses: DiagnosisReport,
    measurement: Optional[MeasurementReport],
    region_stats: Dict[str, RegionStats],
    priors: Sequence[Prior] = (),
    intent: Optional[IntentProfile] = None,
    *,
    verifier: Optional[Verifier] = None,
    long_side: Optional[int] = None,
    crops_for: Optional[Callable[[str, str], List[str]]] = None,
    use_weak: Optional[bool] = None,
) -> DiagnosisReport:
    """
    Hợp nhất tất định giả thuyết VLM với số đo đã hiệu chỉnh (bảng quyết định P6).

    Args:
        hypotheses: chẩn đoán của Perceive (VLM đã gộp, hoặc luật dự phòng khi không có VLM):
            defects = lời VLM, rejected = giả thuyết dưới ngưỡng đồng thuận, preserve.
        measurement: tóm tắt đo toàn ảnh; region_stats: số đo theo vùng (kèm 'background').
        priors: prior EXIF (C3), chỉ dịch confidence. intent: ý định người dùng, áp sau cùng.
        verifier: gọi Gemini verify cho các câu hỏi đóng; None → không kiểm chứng được.
        long_side: cạnh dài ảnh gốc để quy độ mờ của vùng (C4).
        crops_for: (loại lỗi, vùng) → nhãn crop nên xem (đưa vào câu hỏi).
        use_weak: dùng giả thuyết bị loại làm bằng chứng yếu; None → theo env/hằng số.

    Returns:
        DiagnosisReport mới: defects = confirmed (kèm confidence, evidence_items, fixable,
        contraindications), rejected, ask_user, preserve (thêm aspect verify xác nhận chủ ý).
    """
    use_weak = _weak_rejected_enabled() if use_weak is None else use_weak
    vlm_mode = hypotheses.source == "vlm"
    preserve = list(hypotheses.preserve)

    if vlm_mode:
        claims = list(hypotheses.defects)
        rule_defects: List[Defect] = []
    else:
        # Luật dự phòng: lỗi toàn ảnh của luật đã là số đo hiệu chỉnh (D4) → giữ nguyên
        claims = []
        rule_defects = [d for d in hypotheses.defects if d.origin != "vlm"]
    working = hypotheses.model_copy(update={"defects": claims})
    # Luật dự phòng đã có lỗi toàn ảnh từ cùng bảng hiệu chỉnh: chỉ thêm giả thuyết theo vùng
    pool = collect_hypotheses(
        working, measurement, region_stats, priors, long_side, use_weak, full_image=vlm_mode
    )
    if not vlm_mode:
        covered = {(d.type, canonical_region(d.region)) for d in rule_defects}
        pool = [
            h
            for h in pool
            if (h.type, h.region) not in covered
            and not any(_mentions(d, h.type, h.region) for d in rule_defects)
        ]

    preserved_by = {h.id: _preserving_aspect(preserve, h.type, h.region) for h in pool}
    asked = [h for h in pool if _needs_verify(h, preserved_by[h.id], vlm_mode, use_weak)]
    answers: Dict[str, VerifyAnswer] = {}
    if asked and verifier is not None:
        questions = [
            verify_question(h, crops_for(h.type, h.region) if crops_for else ()) for h in asked
        ]
        try:
            answers = verifier(questions)
        except Exception as exc:
            logger.warning("Verify call failed (%s); deciding without it.", exc)
            answers = {}

    confirmed: List[Defect] = list(rule_defects)
    rejected: List[Defect] = list(hypotheses.rejected)
    ask_user: List[Defect] = []
    notes: List[str] = []
    for hypothesis in pool:
        decision = _decide(
            hypothesis,
            preserved_by[hypothesis.id],
            vlm_mode,
            answers.get(hypothesis.id),
            use_weak,
        )
        if (
            decision.outcome == "confirmed"
            and hypothesis.type in NON_FIXABLE_DEFECTS
            and decision.severity < NON_FIXABLE_MIN_SEVERITY
        ):
            decision = _Decision(
                "rejected",
                decision.severity,
                decision.confidence,
                f"{decision.note}; minor and not fixable",
                decision.verdict,
            )
        origin = hypothesis.vlm.origin if hypothesis.vlm is not None else "measured"
        defect = Defect(
            type=hypothesis.type,
            region=hypothesis.region,
            severity=max(0, min(3, decision.severity)),
            evidence=_evidence_text(hypothesis, decision),
            origin=origin,
            confidence=decision.confidence,
            evidence_items=_evidence_for(hypothesis, decision),
            fixable=hypothesis.type not in NON_FIXABLE_DEFECTS,
            contraindications=_contraindications(hypothesis.type, decision.severity),
        )
        m = hypothesis.measured
        notes.append(
            f"{hypothesis.id} {hypothesis.type}@{hypothesis.region}: "
            f"measured={'-' if m is None else f'{m.severity}/{m.confidence:.2f}'} "
            f"vlm={'-' if hypothesis.vlm is None else hypothesis.vlm.severity} "
            f"→ {decision.outcome} ({decision.note})"
        )
        if decision.outcome == "confirmed":
            confirmed.append(defect)
        elif decision.outcome == "ask_user":
            ask_user.append(defect)
        else:
            rejected.append(defect)
            aspect = INTENTIONAL_ASPECT.get(hypothesis.type)
            if (
                decision.outcome == "preserved"
                and decision.verdict is not None
                and aspect
                and all((p.aspect, p.region) != (aspect, hypothesis.region) for p in preserve)
            ):
                reason = decision.verdict.reason or "Kiểm chứng: chủ ý thẩm mỹ."
                preserve.append(
                    PreserveItem(aspect=aspect, region=hypothesis.region, reason=reason)
                )

    _apply_report_priors(confirmed, priors)
    fused = hypotheses.model_copy(
        update={
            "defects": confirmed,
            "rejected": rejected,
            "ask_user": ask_user,
            "preserve": preserve,
            "fusion_notes": notes,
        }
    )
    if intent is not None:
        from .intent import apply_intent  # intent → perception → fusion: tránh import vòng

        fused = apply_intent(fused, intent)
    return fused


def _apply_report_priors(defects: List[Defect], priors: Sequence[Prior]) -> None:
    """Prior no_sharpen đủ mạnh (JPEG nén nặng) → contraindication cho mọi lỗi đã xác nhận."""
    if not any(p.kind == "no_sharpen" and p.strength >= PRIOR_NO_SHARPEN_STRENGTH for p in priors):
        return
    for defect in defects:
        if "no_sharpen" not in defect.contraindications:
            defect.contraindications.append("no_sharpen")


def drop_preserved_defects(report: DiagnosisReport) -> List[Defect]:
    """Bỏ lỗi mâu thuẫn với preserve trên cùng vùng (hoặc preserve toàn ảnh): chủ ý thắng."""
    kept: List[Defect] = []
    for defect in report.defects:
        conflicts = {
            item.aspect
            for item in report.preserve
            if canonical_region(item.region) in (canonical_region(defect.region), "full")
            and defect.type in CONFLICTING_DEFECTS[item.aspect]
        }
        if conflicts:
            logger.info(
                "Dropping defect '%s' on '%s': conflicts with preserved %s.",
                defect.type,
                defect.region,
                sorted(conflicts),
            )
            continue
        kept.append(defect)
    return kept


def plan_contraindications(report: Optional[DiagnosisReport]) -> List[str]:
    """Hợp các contraindication của lỗi đã xác nhận (planner đọc), theo thứ tự xuất hiện."""
    names: List[str] = []
    for defect in report.defects if report is not None else []:
        for name in defect.contraindications:
            if name not in names:
                names.append(name)
    return names
