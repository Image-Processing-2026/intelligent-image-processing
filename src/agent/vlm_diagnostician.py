"""
Chẩn đoán bệnh lý ảnh bằng mô hình đa phương thức Gemini (VLM Diagnostician).
Kết hợp chỉ số kỹ thuật từ Module 1 và thị giác máy tính để xây dựng kế hoạch điều trị.
"""

import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import ValidationError

try:
    import httpx
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    httpx = None
    genai = None
    genai_types = None

from .knowledge import (
    diagnose_from_metrics,
    format_context,
    plan_actions_from_knowledge,
    retrieve,
)
from .planner import (
    ALLOWED_OPERATIONS,
    PARAMETER_BOUNDS,
    REGION_FIELD_BOUNDS,
    clamp_region_fields,
    normalize_defect_ref,
)
from .state import (
    DiagnosisReport,
    HistoryItem,
    PlanSource,
    RegionOperation,
    TreatmentPlan,
)

logger = logging.getLogger(__name__)

# Model mặc định; ghi đè bằng biến môi trường GEMINI_MODEL
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# Ảnh gửi VLM giữ nguyên độ phân giải: thu nhỏ trung bình hóa nhiễu, che mờ nhẹ và lỗi nén.
# Giữ request dưới giới hạn inline 20 MB theo tài liệu Gemini (base64 +33%) → ảnh thô ≤ 14 MB.
VLM_INLINE_IMAGE_BUDGET = 14 * 1024 * 1024
# Ảnh PNG vượt phần ngân sách của nó → JPEG chất lượng cao, không lấy mẫu con màu (4:4:4)
VLM_JPEG_QUALITY = 95
# Gọi lại khi server Gemini lỗi tạm thời (500 INTERNAL, 503 quá tải); lỗi khác thì fallback ngay
GEMINI_RETRYABLE_CODES = frozenset({500, 503})
GEMINI_MAX_ATTEMPTS = 3
# Thời gian chờ trước lần thử thứ n+1 = GEMINI_RETRY_BASE_DELAY * 2**(n-1) giây (1s, 2s)
GEMINI_RETRY_BASE_DELAY = 1.0
# Timeout mỗi lần gọi (ms); SDK mặc định không có timeout nên một request treo sẽ chờ vô hạn
GEMINI_TIMEOUT_MS = 90_000

SYSTEM_PROMPT = """
Bạn là "AI Image Doctor" - một chuyên gia chẩn đoán và xử lý ảnh theo phương pháp kinh điển (Classical Image Processing).
Nhiệm vụ của bạn:
1. Nhìn ảnh và đọc các chỉ số kỹ thuật đo lường (độ sáng, tương phản, nhiễu, độ mờ).
2. Chẩn đoán vấn đề ở từng vùng cụ thể (ví dụ: bầu trời bị chói, khuôn mặt bị tối, nền bị nhiễu).
3. Đề xuất kế hoạch điều trị chỉ sử dụng các công cụ trong Toolbox hợp lệ:
   - denoise (parameters: method in ['gaussian', 'median', 'bilateral', 'nlm'], strength in [0.1, 2.0])
   - gamma_correct (parameters: gamma in [0.5, 2.5]. LƯU Ý: gamma > 1.0 để làm sáng ảnh, gamma < 1.0 để làm tối ảnh)
   - clahe (parameters: clip_limit in [1.0, 4.0])
   - sharpen (parameters: method in ['unsharp_mask', 'laplacian'], amount in [0.2, 2.0])
   - color_correct (parameters: saturation_scale in [0.5, 1.5], temperature_shift in [-1.0, 1.0])
4. Tuân thủ thứ tự ưu tiên: Luôn khử nhiễu TRƯỚC KHI làm nét (denoise before sharpen).
5. Chọn vùng xử lý cho mỗi action bằng "region_type" (có thể bỏ trống để hệ thống tự suy ra từ target_prompt):
   - "full": toàn bộ ảnh (target_prompt: "full").
   - "face": khuôn mặt người (target_prompt: "face"). Tùy chọn face_mode in ['bbox', 'oval'] (mặc định 'bbox'); với gamma_correct/clahe hệ thống luôn dùng mask bám khuôn mặt, không dùng hộp chữ nhật.
   - "spatial": một phần cố định của khung hình, kèm quadrant in ['top', 'bottom', 'left', 'right', 'center'].
   - "semantic": một đối tượng cụ thể, target_prompt là danh từ ngắn bằng tiếng Anh (ví dụ: "sky", "person", "dog").
     Tùy chọn instance_selection in ['all', 'largest'] (mặc định 'all').
   - Tùy chọn feather_radius in [5, 50] (độ mềm viền mask, mặc định 15).
   - KHÔNG trả về bbox, binary_mask hay tọa độ pixel.
6. Từ vòng 2 trở đi bạn nhận HAI ảnh: ẢNH GỐC (trước mọi xử lý) và ẢNH HIỆN TẠI (cần chẩn đoán).
   So sánh hai ảnh để thấy các vòng trước đã thay đổi gì; chỉ lập kế hoạch cho ẢNH HIỆN TẠI.
   Nếu ẢNH HIỆN TẠI đã đạt chất lượng tốt, trả về "actions": [] (không xử lý thêm).
7. Nếu có CHẨN ĐOÁN (giai đoạn 1): chỉ lập kế hoạch cho các defect có severity >= 1,
   dùng region của defect làm target_prompt và type của defect làm detected_issue;
   cường độ tham số tỉ lệ với severity (1 nhẹ tay, 3 mạnh tay).
   Được phép chỉnh một VÙNG CON của vùng defect (ví dụ defect ngược sáng ở "person" thì chỉnh
   riêng "face" với face_mode "oval", để tóc, áo và nền không bị kéo theo). Khi đó BẮT BUỘC điền
   "defect_ref" dạng "type@region" của defect gốc (ví dụ "backlit_subject@person") để hệ thống
   theo dõi đúng lỗi; action trên chính vùng của defect hoặc toàn ảnh có thể bỏ trống defect_ref.
   Defect có "fixable": false (motion_blur, jpeg_artifacts, haze, highlight_clipping) là lỗi
   toolbox KHÔNG sửa được: không lập action cho chúng, chỉ dùng để tránh làm nặng thêm.
   Tôn trọng "contraindications" của defect: "no_sharpen" → không dùng sharpen;
   "no_brighten" → không gamma_correct > 1.0 trên toàn ảnh. Thao tác vi phạm sẽ bị loại bỏ.
   Nếu có "diff" (vòng ≥ 2): đó là so sánh đo đạc với ảnh gốc (trạng thái lỗi cũ, tác dụng phụ
   như halo, nhiễu vùng tối tăng); tránh lặp lại thao tác đã gây tác dụng phụ.
   Dùng region_metrics (số đo thật theo vùng) để chọn tham số cho từng vùng.
   TUYỆT ĐỐI giữ nguyên các đặc điểm trong "preserve"; thao tác vi phạm sẽ bị hệ thống loại bỏ.
8. Nếu có TRI THỨC CHUYÊN MÔN: tham số trong playbook đã được hiệu chỉnh bằng đo đạc trên chính
   toolbox này và đã chọn sẵn theo mức độ của ca hiện tại. Áp dụng ĐỦ các bước của công thức
   phù hợp (ví dụ khử nhiễu VÀ nâng sáng) với đúng giá trị đã chọn; chỉ lệch khi thấy lý do cụ
   thể trên ảnh và phải nêu lý do đó. Tuân thủ các mục "Tránh", và nêu id playbook đã dựa vào
   trong reasoning (ví dụ: "[dark-region]").
9. Nếu có Ý ĐỊNH NGƯỜI DÙNG: đó là ưu tiên cao nhất. Không làm mất điều người dùng muốn giữ,
   phải xử lý điều người dùng muốn sửa, và chọn cường độ hợp với phong cách họ mong muốn.
10. Nếu có KINH NGHIỆM TỪ CA TƯƠNG TỰ: học từ phản hồi cũ của chính người dùng này. Ví dụ ca
   giống trước đây bị chê "tối quá" thì lần này nâng sáng mạnh tay hơn. Không chép nguyên
   phác đồ cũ: ảnh hiện tại vẫn là căn cứ chính.

Viết "reasoning" bằng tiếng Việt có dấu. Trả về kết quả dưới định dạng JSON thuần túy theo cấu trúc:
{
  "iteration": 1,
  "reasoning": "Giải thích lý do lựa chọn thuật toán và tham số...",
  "actions": [
    {
      "region_id": "sky",
      "target_prompt": "sky",
      "region_type": "semantic",
      "detected_issue": "overexposed",
      "defect_ref": "overexposed@sky",
      "operation": "gamma_correct",
      "parameters": {"gamma": 0.8},
      "order": 1
    }
  ]
}
"""


def _build_plan_response_schema() -> Dict[str, Any]:
    """
    Dựng JSON Schema cho structured output của Gemini từ chính bảng ràng buộc của planner,
    để schema không bao giờ lệch khỏi toolbox (ALLOWED_OPERATIONS, PARAMETER_BOUNDS).
    Mỗi operation là một biến thể riêng (anyOf) với đúng tham số của nó, bắt buộc điền đủ:
    schema phẳng gộp mọi tham số khiến VLM điền nhầm tên (ví dụ 'amount' thay cho 'strength').
    Schema chỉ định hướng VLM; planner vẫn kẹp lại mọi giá trị.
    """

    def _enum(field: str) -> Dict[str, Any]:
        """Schema enum cho một trường vùng, từ REGION_FIELD_BOUNDS (bỏ giá trị None)."""
        allowed = [value for value in REGION_FIELD_BOUNDS[field]["allowed"] if value is not None]
        return {"type": "string", "enum": allowed}

    def _parameter_schema(constraint: Dict[str, Any]) -> Dict[str, Any]:
        """Schema JSON cho một tham số: enum hoặc số trong [min, max]."""
        if "allowed" in constraint:
            return {"type": "string", "enum": list(constraint["allowed"])}
        return {"type": "number", "minimum": constraint["min"], "maximum": constraint["max"]}

    feather = REGION_FIELD_BOUNDS["feather_radius"]
    region_properties = {
        "region_id": {"type": "string"},
        "target_prompt": {"type": "string"},
        "region_type": _enum("region_type"),
        "quadrant": _enum("quadrant"),
        "face_mode": _enum("face_mode"),
        "instance_selection": {"type": "string", "enum": ["all", "largest"]},
        "feather_radius": {"type": "integer", "minimum": feather["min"], "maximum": feather["max"]},
        "detected_issue": {"type": "string"},
        "defect_ref": {"type": "string"},
    }
    variants = [
        {
            "type": "object",
            "properties": {
                **region_properties,
                "operation": {"type": "string", "enum": [operation]},
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: _parameter_schema(constraint) for name, constraint in bounds.items()
                    },
                    "required": list(bounds),
                },
            },
            "required": ["region_id", "target_prompt", "detected_issue", "operation", "parameters"],
        }
        for operation, bounds in sorted(PARAMETER_BOUNDS.items())
        if operation in ALLOWED_OPERATIONS
    ]
    return {
        "type": "object",
        "properties": {
            "iteration": {"type": "integer"},
            "reasoning": {"type": "string"},
            "actions": {"type": "array", "items": {"anyOf": variants}},
        },
        "required": ["reasoning", "actions"],
    }


PLAN_RESPONSE_SCHEMA: Dict[str, Any] = _build_plan_response_schema()


def _plan_from_vlm_json(data: Any, iteration: int) -> TreatmentPlan:
    """
    Dựng TreatmentPlan từ JSON của VLM, kẹp trường vùng trước khi validate.
    Mỗi action được dựng riêng: một action sai bị loại kèm cảnh báo,
    không làm rỗng cả kế hoạch (rỗng = SHIP âm thầm).
    Nếu VLM đề xuất action nhưng tất cả đều sai → ValueError để gọi hàm chuyển sang rule-based.
    """
    if not isinstance(data, dict):
        raise ValueError("VLM response is not a JSON object")
    raw_actions = data.get("actions") or []
    if not isinstance(raw_actions, list):
        raise ValueError("VLM response 'actions' is not a list")

    actions: List[RegionOperation] = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            logger.warning("Dropping VLM action %d: not a JSON object.", index)
            continue
        fields = clamp_region_fields(raw_action)
        # face_mode VLM bỏ trống chỉ được bù default: bỏ khỏi fields để không vào model_fields_set
        # (executor dựa vào đó để biết face_mode có do kế hoạch chỉ định hay không)
        if raw_action.get("face_mode") is None:
            fields.pop("face_mode", None)
        fields["defect_ref"] = normalize_defect_ref(fields.get("defect_ref"))
        # Schema tham số là hợp của mọi operation: VLM có thể điền null hoặc tham số của
        # operation khác (ví dụ gamma trong denoise) → chỉ giữ tham số thuộc operation này
        if isinstance(fields.get("parameters"), dict):
            own = PARAMETER_BOUNDS.get(str(fields.get("operation", "")).strip().lower())
            fields["parameters"] = {
                key: value
                for key, value in fields["parameters"].items()
                if value is not None and (own is None or key in own)
            }
        try:
            actions.append(RegionOperation(**fields))
        except ValidationError as exc:
            logger.warning(
                "Dropping invalid VLM action %d: %s", index, exc.errors(include_input=False)
            )

    if raw_actions and not actions:
        raise ValueError(f"all {len(raw_actions)} VLM actions were invalid")
    reasoning = data.get("reasoning")
    return TreatmentPlan(
        iteration=iteration,
        reasoning=reasoning if isinstance(reasoning, str) else "",
        actions=actions,
    )


def _build_history_feedback(history: Optional[List[HistoryItem]]) -> str:
    """
    Tổng hợp lịch sử các vòng lặp trước thành đoạn văn bản phản hồi
    để đưa vào prompt gửi VLM, giúp VLM hiểu ngữ cảnh liên vòng.
    """
    if not history:
        return ""

    feedback_parts = ["\n--- PHẢN HỒI TỪ CÁC VÒNG TRƯỚC ---"]
    recent_history = history[-2:] if len(history) > 2 else history

    for item in recent_history:
        actions_summary = []
        if item.plan and item.plan.actions:
            for action in item.plan.actions:
                params_str = ", ".join(f"{k}={v}" for k, v in action.parameters.items())
                actions_summary.append(
                    f"  - [{action.operation}] trên vùng '{action.region_id}' ({params_str})"
                )

        actions_text = (
            "\n".join(actions_summary) if actions_summary else "  (Không có thao tác nào)"
        )

        # Trích xuất delta metrics nếu có
        metrics_before = item.metrics_before or {}
        metrics_after = item.metrics_after or {}
        delta_info = []
        for key in [
            "brightness_mean",
            "contrast_std",
            "noise_variance",
            "sharpness_laplacian_var",
        ]:
            before_val = metrics_before.get(key)
            after_val = metrics_after.get(key)
            if before_val is not None and after_val is not None:
                try:
                    delta = float(after_val) - float(before_val)
                    direction = "tăng" if delta > 0 else "giảm"
                    delta_info.append(
                        f"  - {key}: {float(before_val):.2f} → {float(after_val):.2f} ({direction} {abs(delta):.2f})"
                    )
                except (ValueError, TypeError):
                    pass

        delta_text = "\n".join(delta_info) if delta_info else "  (Không có dữ liệu delta)"
        reasoning = item.plan.reasoning if item.plan else ""
        diagnosis_text = (
            f"Chẩn đoán: {item.diagnosis.summary}\n"
            if item.diagnosis and item.diagnosis.summary
            else ""
        )

        feedback_parts.append(
            f"\n🔄 Vòng {item.iteration}:\n"
            f"{diagnosis_text}"
            f"Các thao tác đã thực hiện:\n{actions_text}\n"
            f"Biến thiên chỉ số kỹ thuật:\n{delta_text}\n"
            f"Quyết định: {item.decision}\n"
            f"Lý do: {reasoning}"
        )

    feedback_parts.append(
        "\n⚠️ Dựa trên lịch sử trên, hãy TINH CHỈNH nhẹ hoặc chuyển sang xử lý vấn đề chưa được giải quyết. "
        "KHÔNG lặp lại cùng thao tác với cùng tham số nếu chỉ số không cải thiện."
    )

    return "\n".join(feedback_parts)


def _rule_based_plan(
    metrics: Dict[str, Any],
    iteration: int,
    source: PlanSource,
    reasoning: str,
    diagnosis: Optional[DiagnosisReport] = None,
) -> TreatmentPlan:
    """
    Kế hoạch dự phòng không cần mạng: chạy công thức của các playbook card auto_apply khớp với
    chẩn đoán (bổ sung lỗi từ chỉ số Module 1). Dùng khi không có GEMINI_API_KEY hoặc khi lời
    gọi/parse Gemini thất bại, để lỗi API không bị biến thành plan rỗng (= SHIP âm thầm).
    """
    notes: List[str] = []
    actions, card_ids = plan_actions_from_knowledge(diagnosis, metrics, notes=notes)
    if card_ids:
        reasoning = f"{reasoning} Playbook: {', '.join(card_ids)}."
    if notes:
        reasoning = f"{reasoning} {' '.join(notes)}"
    return TreatmentPlan(
        iteration=iteration,
        reasoning=reasoning,
        actions=[RegionOperation(**action) for action in actions],
        source=source,
        knowledge=card_ids,
    )


def to_vlm_image(image: np.ndarray) -> Image.Image:
    """Chuyển ảnh sang PIL ở độ phân giải gốc (không thu nhỏ) để gửi VLM."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def _encode_image(image: Image.Image, budget: int) -> Tuple[bytes, str]:
    """
    Mã hóa ảnh gửi Gemini: PNG không mất mát nếu vừa ngân sách byte, ngược lại JPEG
    chất lượng VLM_JPEG_QUALITY (4:4:4). Trả về (bytes, mime_type).
    """
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    if buffer.tell() <= budget:
        return buffer.getvalue(), "image/png"
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=VLM_JPEG_QUALITY, subsampling=0)
    if buffer.tell() > budget:
        logger.warning(
            "Image %dx%d is %.1f MB as JPEG, over the %.1f MB inline budget; "
            "the request may be slow or rejected.",
            image.width,
            image.height,
            buffer.tell() / 2**20,
            budget / 2**20,
        )
    return buffer.getvalue(), "image/jpeg"


def _encode_contents(contents: List[Any]) -> List[Any]:
    """
    Thay mỗi ảnh PIL trong nội dung bằng Part đã mã hóa, chung ngân sách
    VLM_INLINE_IMAGE_BUDGET. Ảnh nhỏ (crop, overlay) được mã hóa trước; phần ngân sách còn lại
    chia đều cho các ảnh lớn hơn, nên crop 512 px không làm ảnh toàn khung mất ngân sách.
    """
    images = [index for index, part in enumerate(contents) if isinstance(part, Image.Image)]
    if not images:
        return contents
    encoded: List[Any] = list(contents)
    remaining = VLM_INLINE_IMAGE_BUDGET
    by_size = sorted(images, key=lambda index: contents[index].width * contents[index].height)
    for position, index in enumerate(by_size):
        budget = remaining // (len(by_size) - position)
        data, mime_type = _encode_image(contents[index], budget)
        remaining = max(0, remaining - len(data))
        encoded[index] = genai_types.Part.from_bytes(data=data, mime_type=mime_type)
    return encoded


def image_parts(
    image: np.ndarray, iteration: int, original_image: Optional[np.ndarray]
) -> List[Any]:
    """Phần ảnh của nội dung gửi VLM; từ vòng 2 gửi kèm ảnh gốc để so sánh trước/sau."""
    parts: List[Any] = []
    if iteration > 1 and original_image is not None:
        parts += [
            "ẢNH GỐC (trước mọi xử lý):",
            to_vlm_image(original_image),
            "ẢNH HIỆN TẠI (cần chẩn đoán):",
        ]
    parts.append(to_vlm_image(image))
    return parts


# Trường DiagnosisReport không đưa vào prompt Plan
_PLAN_DIAGNOSIS_EXCLUDE = {
    "source",
    "iteration",
    "measurement",
    "region_stats",
    "subject_boxes",
    "rejected",
    "ask_user",
    "fusion_notes",
}


def _build_contents(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int,
    history: Optional[List[HistoryItem]],
    original_image: Optional[np.ndarray],
    diagnosis: Optional[DiagnosisReport] = None,
    knowledge_text: str = "",
) -> List[Any]:
    """
    Dựng nội dung đa phương thức gửi Gemini (giai đoạn Plan): chỉ số kỹ thuật,
    chẩn đoán giai đoạn 1 (nếu có), tri thức truy xuất, phản hồi các vòng trước và ảnh.
    """
    prompt = f"Chỉ số kỹ thuật hiện tại:\n{json.dumps(metrics, indent=2, default=str)}\n"
    if diagnosis is not None:
        # Bỏ số đo v2 (dài, đã tóm vào region_metrics) và bbox subject: prompt Plan cấm bbox
        diagnosis_json = diagnosis.model_dump(mode="json", exclude=_PLAN_DIAGNOSIS_EXCLUDE)
        prompt += (
            "CHẨN ĐOÁN (giai đoạn 1, kèm region_metrics đo thực tế theo vùng):\n"
            f"{json.dumps(diagnosis_json, indent=2, ensure_ascii=False)}\n"
        )
    if knowledge_text:
        prompt += f"{knowledge_text}\n"
    prompt += f"Vòng lặp: {iteration}{_build_history_feedback(history)}"
    return [prompt, *image_parts(image, iteration, original_image)]


def _parse_json_text(text: Optional[str]) -> Any:
    """Parse JSON từ phản hồi VLM; vẫn chịu được khối ```json``` dù đã bật structured output."""
    if not text:
        raise ValueError("VLM response has no text")
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """
    Lỗi tạm thời thì đáng gọi lại: lỗi server (google.genai ServerError mang thuộc tính
    `code`) hoặc request quá GEMINI_TIMEOUT_MS.
    """
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return True
    return isinstance(exc, TimeoutError) or getattr(exc, "code", None) in GEMINI_RETRYABLE_CODES


def call_gemini_json(
    api_key: str,
    system_prompt: str,
    contents: List[Any],
    schema: Dict[str, Any],
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> Any:
    """
    Gọi Gemini với structured output (JSON theo schema) và trả về JSON đã parse.
    Dùng chung cho giai đoạn Perceive, Plan và Review; mọi lỗi được ném ra để bên gọi fallback.
    temperature: None = mặc định của model; Perceive self-consistency dùng 0.7.
    model / timeout_ms: None = GEMINI_MODEL (hoặc mặc định) / GEMINI_TIMEOUT_MS; Review ghi đè.
    Lỗi 500/503 và timeout được gọi lại tối đa GEMINI_MAX_ATTEMPTS lần, thời gian chờ tăng gấp đôi.
    """
    if genai is None or genai_types is None:
        raise RuntimeError("google-genai is not installed")

    client = genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=timeout_ms or GEMINI_TIMEOUT_MS),
    )
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_json_schema=schema,
        # Không khai báo tool nào → tắt AFC để SDK không cảnh báo/không tự gọi hàm
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        # Ngân sách token ảnh lớn nhất: model tự lấy mẫu lại ảnh theo ngân sách này
        media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        temperature=temperature,
    )
    contents = _encode_contents(contents)
    model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            break
        except Exception as exc:
            if attempt == GEMINI_MAX_ATTEMPTS or not _is_retryable_gemini_error(exc):
                raise
            delay = GEMINI_RETRY_BASE_DELAY * 2 ** (attempt - 1)
            logger.warning(
                "Gemini call failed with %s (attempt %d/%d); retrying in %.1fs.",
                getattr(exc, "code", None) or type(exc).__name__,
                attempt,
                GEMINI_MAX_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
    return _parse_json_text(response.text)


def diagnose_and_plan(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,
    diagnosis: Optional[DiagnosisReport] = None,
    intent_text: str = "",
    experience_text: str = "",
) -> TreatmentPlan:
    """
    Giai đoạn Plan: gọi Gemini lập kế hoạch điều trị dựa trên chỉ số và chẩn đoán.
    intent_text: khối 'Ý ĐỊNH NGƯỜI DÙNG' (intent.intent_prompt), đặt trước tri thức.
    experience_text: khối 'KINH NGHIỆM TỪ CA TƯƠNG TỰ' (memory.experience_prompt).
    - Không có GEMINI_API_KEY → kế hoạch rule-based (source="rule_based").
    - Lỗi SDK/mạng/parse → kế hoạch rule-based (source="vlm_fallback"), không trả plan rỗng.
    - VLM trả "actions": [] hợp lệ → plan rỗng thật sự (ảnh đã tốt).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _rule_based_plan(
            metrics,
            iteration,
            source="rule_based",
            reasoning="Chế độ Fallback Rule-Based: Điều chỉnh dựa trên ngưỡng thống kê kỹ thuật.",
            diagnosis=diagnosis,
        )

    try:
        context = retrieve(diagnosis or diagnose_from_metrics(metrics, iteration), metrics)
        knowledge_text = "\n".join(
            text for text in (intent_text, experience_text, format_context(context)) if text
        )
        contents = _build_contents(
            image, metrics, iteration, history, original_image, diagnosis, knowledge_text
        )
        data = call_gemini_json(api_key, SYSTEM_PROMPT, contents, PLAN_RESPONSE_SCHEMA)
        plan = _plan_from_vlm_json(data, iteration)
        plan.knowledge = context.ids
        return plan

    except Exception as exc:
        # Lỗi API không được biến thành plan rỗng (= SHIP âm thầm) → dùng luật dự phòng
        logger.warning("Gemini diagnosis failed (%s); using the rule-based fallback plan.", exc)
        return _rule_based_plan(
            metrics,
            iteration,
            source="vlm_fallback",
            reasoning=(
                "Gọi Gemini thất bại, chuyển sang kế hoạch Rule-Based dựa trên ngưỡng "
                "thống kê kỹ thuật."
            ),
            diagnosis=diagnosis,
        )
