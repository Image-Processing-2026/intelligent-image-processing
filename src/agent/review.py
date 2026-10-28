"""
Cổng review ảnh sẽ trả về (R-08) và kết cục cuối của lượt chạy (R-01).

`review_candidate` cho Gemini xem ảnh gốc, ảnh trước vòng cuối và ẢNH SẼ TRẢ (sau rollback nếu
có), kèm crop 100 % ghép cặp của vùng đã đổi và crop nghi vấn. Review chỉ là người nhận xét:
`decide_node` vẫn là bộ điều phối tất định. Mọi lỗi (không key, timeout, parse, bằng chứng
thiếu) đều cho `UNVERIFIED`, không bao giờ cho `ACCEPTED`.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .state import (
    DiagnosisReport,
    ExecutedAction,
    HistoryItem,
    IntentProfile,
    Outcome,
    ReviewFinding,
    ReviewResult,
    SideEffectCheck,
)
from .vlm_diagnostician import (
    DEFAULT_GEMINI_MODEL,
    _is_retryable_gemini_error,
    call_gemini_json,
    to_vlm_image,
)

logger = logging.getLogger(__name__)

# Bật/tắt cổng review. Unit test tắt mặc định (tests/conftest.py); test review tự bật lại.
REVIEW_ENABLED = True
# None = dùng GEMINI_MODEL (hoặc mặc định); biến môi trường REVIEW_MODEL ghi đè
REVIEW_MODEL: Optional[str] = None
REVIEW_MODEL_ENV = "REVIEW_MODEL"
# Tạm thời, chưa hiệu chỉnh (BL-10)
REVIEW_TIMEOUT_MS = 60_000
# Ngân sách lời gọi Gemini cho một lượt review: lần đầu + một lần thử lại lỗi tạm thời
REVIEW_MAX_CALLS = 2
PROMPT_VERSION = "review-v1"

# Crop ghép cặp (gốc | kết quả) quanh vùng thay đổi lớn nhất. Tạm thời, chưa hiệu chỉnh (BL-10)
CROP_SIZE = 384
MAX_CHANGE_CROPS = 2
# Chênh lệch trung bình (thang 0..255) tối thiểu để coi một cửa sổ là "đã đổi"
MIN_CHANGE_LEVEL = 2.0
# Pixel tối của ảnh gốc (L* thang OpenCV 0..255) và mức nâng tối thiểu để tính là black-lift
SUSPECT_DARK_L = 64
SUSPECT_MIN_LIFT = 12
FULL_FRAME_ID = "FULL"

SIDE_EFFECT_SUSPECT_KINDS = frozenset({"black_lift"})

VERDICTS = ("ACCEPT", "REVISE", "REJECT", "UNCERTAIN")
FINDING_ISSUES = (
    "black_lift",
    "halo",
    "color_shift",
    "banding",
    "noise_boost",
    "over_smooth",
    "over_exposed",
    "under_corrected",
    "preserve_violated",
    "artifact",
    "other",
)

REVIEW_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "goal": {"type": "string"},
        "preserve": {"type": "array", "items": {"type": "string"}},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "crop_id": {"type": "string"},
                    "issue": {"type": "string", "enum": list(FINDING_ISSUES)},
                    "severity": {"type": "integer", "minimum": 0, "maximum": 3},
                    "text": {"type": "string"},
                },
                "required": ["crop_id", "issue", "severity", "text"],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "goal", "preserve", "findings", "confidence", "reason"],
}

REVIEW_PROMPT = """
Bạn là người duyệt ảnh độc lập của "AI Image Doctor". Một pipeline xử lý ảnh bằng thuật toán
cổ điển (không sinh ảnh) vừa chỉnh một bức ảnh. Nhiệm vụ của bạn là nhìn ẢNH SẼ TRẢ CHO NGƯỜI
DÙNG và nói thật nó có đạt mục tiêu mà không phá hỏng gì không. Bạn KHÔNG lập kế hoạch sửa.

Bạn nhận: ẢNH GỐC, (có thể) ẢNH TRƯỚC VÒNG CUỐI, ẢNH SẼ TRẢ (nhãn FULL), các CROP 100 % ghép
cặp "gốc | kết quả" của vùng đã đổi nhiều nhất, và có thể có CROP NGHI VẤN. Mỗi ảnh có nhãn
(FULL, A, B, …). Kèm theo là ý định người dùng, lỗi đã chẩn đoán ban đầu, các thao tác thực
sự đã chạy và cảnh báo tác dụng phụ đo được.

Trả về JSON:
- goal: mục tiêu chỉnh sửa của ảnh này, một câu.
- preserve: các đặc điểm cần giữ nguyên (phong cách, vùng tối chủ ý, màu da…).
- findings: mỗi nhận xét PHẢI có crop_id là nhãn ảnh dùng làm bằng chứng (FULL, A, B, …),
  issue lấy từ danh sách cho phép, severity 0–3, text ngắn (một câu).
- verdict:
  - ACCEPT: mục tiêu đạt rõ và không thấy hư hại đáng kể.
  - REVISE: hướng chỉnh đúng nhưng mục tiêu chưa đạt hoặc còn tác dụng phụ nhẹ. Cần findings.
  - REJECT: ảnh trả về tệ hơn ảnh trước/gốc (hư hại rõ). Cần findings.
  - UNCERTAIN: không đủ bằng chứng để kết luận. ĐỪNG đoán; đây là đáp án hợp lệ.
- confidence: mức tự tin 0–1 (chỉ để ghi nhận).
- reason: lý do ngắn gọn, không suy luận dài.
Chỉ nêu điều quan sát được trong ảnh gửi kèm. Không bịa vùng không có trong crop.
""".strip()


@dataclass
class ReviewContext:
    """Bối cảnh review: ý định, lỗi ban đầu và thao tác thực sự đã chạy."""

    intent: Optional[IntentProfile] = None
    defects: List[Dict[str, Any]] = field(default_factory=list)
    executed: List[ExecutedAction] = field(default_factory=list)
    side_effects: Optional[SideEffectCheck] = None
    iterations: int = 0


@dataclass(frozen=True)
class ReviewCrop:
    """Một crop gửi reviewer: ảnh ghép cặp gốc | kết quả."""

    crop_id: str
    box: Tuple[int, int, int, int]  # pixel (x0, y0, x1, y1)
    description: str
    image: np.ndarray


def image_sha256(image: np.ndarray) -> str:
    """sha256 của ảnh (kèm shape) dùng để chứng minh review xem đúng ảnh được trả."""
    array = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def same_image(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    """True nếu hai ảnh giống hệt từng pixel."""
    return a is not None and b is not None and bool(np.array_equal(a, b))


def _gray(image: np.ndarray) -> np.ndarray:
    """Ảnh xám float32 (thang 0..255) từ RGB hoặc xám."""
    if image.ndim == 2 or image.shape[2] == 1:
        return image.reshape(image.shape[0], image.shape[1]).astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _window_box(cx: int, cy: int, shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Khung CROP_SIZE tâm (cx, cy), kẹp trong ảnh (ảnh nhỏ hơn thì lấy cả ảnh)."""
    height, width = shape[:2]
    size_x, size_y = min(CROP_SIZE, width), min(CROP_SIZE, height)
    x0 = int(np.clip(cx - size_x // 2, 0, width - size_x))
    y0 = int(np.clip(cy - size_y // 2, 0, height - size_y))
    return x0, y0, x0 + size_x, y0 + size_y


def _pair(
    original: np.ndarray, candidate: np.ndarray, box: Tuple[int, int, int, int]
) -> np.ndarray:
    """Ghép crop ảnh gốc (trái) và ảnh kết quả (phải) cùng khung, độ phân giải 100 %."""
    x0, y0, x1, y1 = box
    return np.concatenate([original[y0:y1, x0:x1], candidate[y0:y1, x0:x1]], axis=1)


def _change_peaks(original: np.ndarray, candidate: np.ndarray) -> List[Tuple[int, int]]:
    """
    Tâm các cửa sổ có thay đổi trung bình lớn nhất giữa ảnh gốc và ảnh trả (tối đa
    MAX_CHANGE_CROPS, không chồng nhau). Ảnh gần như không đổi → [].
    """
    delta = np.abs(_gray(candidate) - _gray(original))
    window = max(8, CROP_SIZE // 2)
    smooth = cv2.boxFilter(delta, -1, (window, window), normalize=True)
    peaks: List[Tuple[int, int]] = []
    for _ in range(MAX_CHANGE_CROPS):
        _, high, _, location = cv2.minMaxLoc(smooth)
        if high < MIN_CHANGE_LEVEL:
            break
        cx, cy = location
        peaks.append((cx, cy))
        x0, y0, x1, y1 = _window_box(cx, cy, smooth.shape)
        smooth[y0:y1, x0:x1] = 0.0
    return peaks


def _suspect_center(original: np.ndarray, candidate: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Tâm vùng nghi black-lift: pixel TỐI của ảnh gốc bị nâng sáng nhiều nhất. None nếu không có
    pixel nào thỏa (ảnh không có vùng tối bị chỉnh).
    """
    before, after = _gray(original), _gray(candidate)
    lifted = ((before < SUSPECT_DARK_L) & (after - before > SUSPECT_MIN_LIFT)).astype(np.float32)
    if float(lifted.sum()) <= 0.0:
        return None
    window = max(8, CROP_SIZE // 4)
    smooth = cv2.boxFilter(lifted, -1, (window, window), normalize=True)
    _, high, _, location = cv2.minMaxLoc(smooth)
    return location if high > 0.0 else None


def select_review_crops(
    original: np.ndarray, candidate: np.ndarray, side_effects: Optional[SideEffectCheck] = None
) -> List[ReviewCrop]:
    """
    Crop ghép cặp để reviewer soi ở 100 %: vùng đổi nhiều nhất (nhãn A, B) rồi crop nghi vấn
    (nhãn kế tiếp) khi phép đo tác dụng phụ đã báo black-lift. Thiếu dữ liệu thì bỏ qua.
    """
    if original.shape != candidate.shape:
        return []
    crops: List[ReviewCrop] = []
    for cx, cy in _change_peaks(original, candidate):
        box = _window_box(cx, cy, original.shape)
        crop_id = chr(ord("A") + len(crops))
        crops.append(
            ReviewCrop(
                crop_id,
                box,
                "vùng thay đổi nhiều nhất (trái: ảnh gốc, phải: ảnh sẽ trả)",
                _pair(original, candidate, box),
            )
        )
    flagged = side_effects is not None and any(
        effect.get("kind") in SIDE_EFFECT_SUSPECT_KINDS
        for effect in (*side_effects.previous, *side_effects.original)
    )
    center = _suspect_center(original, candidate) if flagged else None
    if center is not None:
        box = _window_box(center[0], center[1], original.shape)
        crops.append(
            ReviewCrop(
                chr(ord("A") + len(crops)),
                box,
                "vùng nghi bị nâng đen (black-lift) (trái: ảnh gốc, phải: ảnh sẽ trả)",
                _pair(original, candidate, box),
            )
        )
    return crops


def _context_text(context: ReviewContext, crops: Sequence[ReviewCrop], has_previous: bool) -> str:
    """Khối văn bản bối cảnh gửi reviewer (tiếng Việt, đủ ngắn)."""
    lines = [f"Số vòng đã chạy: {context.iterations}."]
    intent = context.intent
    if intent is not None:
        if intent.keep:
            lines.append("Người dùng muốn GIỮ: " + ", ".join(str(item) for item in intent.keep))
        if intent.fix:
            lines.append("Người dùng muốn SỬA: " + ", ".join(intent.fix))
        if intent.notes:
            lines.append(f"Ghi chú người dùng: {intent.notes}")
    if context.defects:
        lines.append("Lỗi chẩn đoán ban đầu:")
        lines += [
            f"- {d.get('type')} @ {d.get('region')} (severity {d.get('severity')})"
            for d in context.defects
        ]
    if context.executed:
        lines.append("Thao tác THỰC SỰ đã chạy (không phải kế hoạch):")
        for item in context.executed:
            state = f"BỎ QUA ({item.skip_reason})" if item.skipped else (item.backend or "ok")
            fallback = f", fallback: {item.fallback_reason}" if item.fallback_reason else ""
            lines.append(
                f"- {item.operation} trên {item.region_id} {item.parameters} "
                f"[{state}{fallback}, strength_scale={item.strength_scale}]"
            )
    if context.side_effects is not None and context.side_effects.status == "ok":
        found = [*context.side_effects.previous, *context.side_effects.original]
        if found:
            lines.append("Tác dụng phụ đo được: " + ", ".join(str(e.get("kind")) for e in found))
    elif context.side_effects is not None and context.side_effects.status == "unavailable":
        lines.append("Phép đo tác dụng phụ không chạy được: hãy tự soi kỹ các crop.")
    labels = [FULL_FRAME_ID] + [crop.crop_id for crop in crops]
    lines.append("Nhãn hợp lệ cho crop_id: " + ", ".join(labels))
    lines.append("Có ảnh trước vòng cuối." if has_previous else "Không có ảnh trước vòng cuối.")
    return "\n".join(lines)


def _build_contents(
    original: np.ndarray,
    previous: Optional[np.ndarray],
    candidate: np.ndarray,
    crops: Sequence[ReviewCrop],
    context: ReviewContext,
) -> List[Any]:
    """Nội dung đa phương thức gửi reviewer."""
    has_previous = previous is not None and not same_image(previous, original)
    contents: List[Any] = [_context_text(context, crops, has_previous)]
    contents += ["ẢNH GỐC:", to_vlm_image(original)]
    if has_previous and previous is not None:
        contents += ["ẢNH TRƯỚC VÒNG CUỐI:", to_vlm_image(previous)]
    contents += [f"ẢNH SẼ TRẢ ({FULL_FRAME_ID}):", to_vlm_image(candidate)]
    for crop in crops:
        contents += [f"Crop {crop.crop_id}: {crop.description}", to_vlm_image(crop.image)]
    return contents


def _model_name() -> str:
    """Model dùng cho review: REVIEW_MODEL (env) > REVIEW_MODEL > GEMINI_MODEL > mặc định."""
    return (
        os.getenv(REVIEW_MODEL_ENV)
        or REVIEW_MODEL
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )


def parse_review(data: Any, valid_ids: Sequence[str]) -> Tuple[ReviewResult, Optional[str]]:
    """
    Phản hồi VLM → ReviewResult (chưa gắn sha/model/latency). REVISE/REJECT mà không có finding
    nào trỏ tới crop hợp lệ bị hạ về UNCERTAIN, và trả kèm lý do hạ cấp.
    """
    if not isinstance(data, dict):
        raise ValueError("review response is not a JSON object")
    verdict = str(data.get("verdict") or "").upper()
    if verdict not in VERDICTS:
        raise ValueError(f"review verdict {verdict!r} is not allowed")
    findings = [
        ReviewFinding(
            crop_id=str(item.get("crop_id") or ""),
            issue=str(item.get("issue") or ""),
            severity=int(np.clip(int(item.get("severity", 1)), 0, 3)),
            text=str(item.get("text") or ""),
        )
        for item in data.get("findings") or []
        if isinstance(item, dict)
    ]
    grounded = [item for item in findings if item.crop_id in valid_ids]
    downgrade: Optional[str] = None
    if verdict in ("REVISE", "REJECT") and not grounded:
        downgrade = f"{verdict} has no finding that points at a sent image"
        verdict = "UNCERTAIN"
    result = ReviewResult(
        status="completed",
        verdict=verdict,
        goal=str(data.get("goal") or ""),
        preserve=[str(item) for item in data.get("preserve") or []],
        findings=grounded,
        confidence=float(np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0)),
        reason=str(data.get("reason") or ""),
    )
    return result, downgrade


def review_candidate(
    original: np.ndarray,
    previous: Optional[np.ndarray],
    candidate: np.ndarray,
    context: Optional[ReviewContext] = None,
) -> ReviewResult:
    """
    Review ảnh SẼ TRẢ (`candidate`) bằng Gemini. Không bao giờ ném lỗi: không bật, không có key,
    lỗi gọi/parse đều trả ReviewResult không có verdict ACCEPT, để bên gọi ra UNVERIFIED.
    Tối đa REVIEW_MAX_CALLS lời gọi; chỉ thử lại lỗi tạm thời (5xx/timeout).
    """
    context = context or ReviewContext()
    result = ReviewResult(
        candidate_sha256=image_sha256(candidate),
        model=_model_name(),
        prompt_version=PROMPT_VERSION,
    )
    if not REVIEW_ENABLED:
        result.reason = "review disabled"
        return result
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        result.status = "unavailable"
        result.reason = "no GEMINI_API_KEY; the returned image was not reviewed"
        return result

    started = time.monotonic()
    try:
        crops = select_review_crops(original, candidate, context.side_effects)
        contents = _build_contents(original, previous, candidate, crops, context)
        valid_ids = [FULL_FRAME_ID] + [crop.crop_id for crop in crops]
    except Exception as exc:
        logger.warning("Review input could not be built: %s", exc)
        result.status = "unavailable"
        result.error = str(exc)
        result.reason = "review input could not be built"
        return result

    for attempt in range(1, max(1, REVIEW_MAX_CALLS) + 1):
        try:
            data = call_gemini_json(
                api_key,
                REVIEW_PROMPT,
                contents,
                REVIEW_RESPONSE_SCHEMA,
                model=result.model,
                timeout_ms=REVIEW_TIMEOUT_MS,
            )
            parsed, downgrade = parse_review(data, valid_ids)
        except Exception as exc:
            if attempt < REVIEW_MAX_CALLS and _is_retryable_gemini_error(exc):
                logger.warning("Review call failed (%s); retrying once.", exc)
                continue
            logger.warning("Review failed: %s", exc)
            result.status = "unavailable"
            result.error = f"{type(exc).__name__}: {exc}"
            result.reason = "review call failed"
            break
        parsed.candidate_sha256 = result.candidate_sha256
        parsed.model = result.model
        parsed.prompt_version = PROMPT_VERSION
        if downgrade:
            parsed.error = downgrade
            parsed.reason = parsed.reason or downgrade
        result = parsed
        break
    result.latency_ms = int((time.monotonic() - started) * 1000)
    return result


def review_context_from_state(
    history: Sequence[HistoryItem], intent: Optional[IntentProfile]
) -> ReviewContext:
    """
    Bối cảnh review từ lịch sử: lỗi của chẩn đoán đầu tiên, và thao tác/tác dụng phụ của vòng
    cuối đã được giữ lại (vòng bị rollback không tạo nên ảnh được trả).
    """
    first = next((item.diagnosis for item in history if item.diagnosis is not None), None)
    kept = [item for item in history if not item.rolled_back]
    last = kept[-1] if kept else None
    return ReviewContext(
        intent=intent,
        defects=_defect_dicts(first),
        executed=list(last.executed) if last is not None else [],
        side_effects=last.side_effect_check if last is not None else None,
        iterations=len(history),
    )


def _defect_dicts(diagnosis: Optional[DiagnosisReport]) -> List[Dict[str, Any]]:
    """Các lỗi (loại, vùng, severity) của chẩn đoán, dạng dict gọn cho prompt."""
    if diagnosis is None:
        return []
    return [
        {"type": d.type, "region": d.region, "severity": d.severity}
        for d in diagnosis.defects
        if d.severity >= 1
    ]


def wanted_fix(history: Sequence[HistoryItem], intent: Optional[IntentProfile]) -> bool:
    """True nếu lượt chạy có ý sửa ảnh: người dùng yêu cầu, hoặc có lỗi/thao tác thật sự."""
    if intent is not None and (intent.fix or any(edit.op == "add_defect" for edit in intent.edits)):
        return True
    for item in history:
        if item.plan.actions:
            return True
        if item.diagnosis is not None and item.diagnosis.actionable_defects:
            return True
    return False


def derive_outcome(
    *,
    original: np.ndarray,
    returned: np.ndarray,
    is_synthetic: bool,
    reference_met: bool,
    history: Sequence[HistoryItem],
    intent: Optional[IntentProfile],
    review: Optional[ReviewResult],
) -> Outcome:
    """
    Kết cục cuối của ảnh ĐƯỢC TRẢ (R-01). Chỉ ACCEPTED khi có người xác nhận: review ACCEPT
    cho ảnh thật, hoặc đạt ngưỡng tham chiếu cho ảnh synthetic có ground truth. Hết vòng hay
    điểm heuristic cao không đủ để ACCEPTED.
    """
    sha = image_sha256(returned)
    if same_image(returned, original):
        if wanted_fix(history, intent):
            return Outcome(
                status="KEPT_ORIGINAL",
                candidate_sha256=sha,
                reason="the returned image is identical to the original; no edit survived",
            )
        return Outcome(status="NO_CHANGE", candidate_sha256=sha, reason="no defect needed fixing")
    if is_synthetic:
        if reference_met:
            return Outcome(
                status="ACCEPTED",
                accepted_by="reference",
                candidate_sha256=sha,
                reason="reference metrics reach the PSNR/SSIM targets",
            )
        return Outcome(
            status="NOT_MET",
            candidate_sha256=sha,
            reason="reference metrics are below the PSNR/SSIM targets",
        )
    if review is None or review.status != "completed" or review.verdict is None:
        detail = review.reason if review is not None and review.reason else "no review"
        return Outcome(
            status="UNVERIFIED",
            candidate_sha256=sha,
            reason=f"the returned image was not verified by a reviewer ({detail})",
        )
    if review.verdict == "REJECT":
        return Outcome(
            status="UNVERIFIED",
            candidate_sha256=sha,
            reason="the reviewer rejected the last result; the previous image is returned",
        )
    if review.verdict == "REVISE":
        return Outcome(
            status="NOT_MET",
            candidate_sha256=sha,
            reason=review.reason or "the reviewer says the goal is not met yet",
        )
    if review.verdict == "ACCEPT" and review.candidate_sha256 == sha:
        return Outcome(
            status="ACCEPTED",
            accepted_by="review",
            candidate_sha256=sha,
            reason=review.reason or "accepted by the reviewer",
        )
    # UNCERTAIN, hoặc review xem ảnh khác với ảnh được trả
    return Outcome(
        status="UNVERIFIED",
        candidate_sha256=sha,
        reason=review.reason or "the reviewer could not decide",
    )
