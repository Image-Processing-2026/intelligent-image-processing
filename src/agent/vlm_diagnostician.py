"""
Chẩn đoán bệnh lý ảnh bằng mô hình đa phương thức Gemini (VLM Diagnostician).
Kết hợp chỉ số kỹ thuật từ Module 1 và thị giác máy tính để xây dựng kế hoạch điều trị.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import ValidationError

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from .planner import ALLOWED_OPERATIONS, PARAMETER_BOUNDS, REGION_FIELD_BOUNDS, clamp_region_fields
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
# Cạnh dài tối đa của ảnh gửi VLM: đủ để chẩn đoán, giảm token và độ trễ
VLM_MAX_SIDE = 1024

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
   - "face": khuôn mặt người (target_prompt: "face"). Tùy chọn face_mode in ['bbox', 'oval'] (mặc định 'bbox').
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
   Dùng region_metrics (số đo thật theo vùng) để chọn tham số cho từng vùng.
   TUYỆT ĐỐI giữ nguyên các đặc điểm trong "preserve"; thao tác vi phạm sẽ bị hệ thống loại bỏ.

Viết "reasoning" bằng tiếng Việt. Trả về kết quả dưới định dạng JSON thuần túy theo cấu trúc:
{
  "iteration": 1,
  "reasoning": "Giải thích lý do lựa chọn thuật toán và tham số...",
  "actions": [
    {
      "region_id": "sky",
      "target_prompt": "sky",
      "region_type": "semantic",
      "detected_issue": "overexposed",
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
    Schema chỉ định hướng VLM; planner vẫn kẹp lại mọi giá trị.
    """
    parameter_properties: Dict[str, Any] = {}
    for bounds in PARAMETER_BOUNDS.values():
        for name, constraint in bounds.items():
            if "allowed" in constraint:
                previous = parameter_properties.get(name, {}).get("enum", [])
                merged = sorted(set(previous) | set(constraint["allowed"]))
                parameter_properties[name] = {"type": "string", "enum": merged}
            else:
                parameter_properties[name] = {
                    "type": "number",
                    "minimum": constraint["min"],
                    "maximum": constraint["max"],
                }

    def _enum(field: str) -> Dict[str, Any]:
        allowed = [value for value in REGION_FIELD_BOUNDS[field]["allowed"] if value is not None]
        return {"type": "string", "enum": allowed}

    feather = REGION_FIELD_BOUNDS["feather_radius"]
    action_schema = {
        "type": "object",
        "properties": {
            "region_id": {"type": "string"},
            "target_prompt": {"type": "string"},
            "region_type": _enum("region_type"),
            "quadrant": _enum("quadrant"),
            "face_mode": _enum("face_mode"),
            "instance_selection": {"type": "string", "enum": ["all", "largest"]},
            "feather_radius": {
                "type": "integer",
                "minimum": feather["min"],
                "maximum": feather["max"],
            },
            "detected_issue": {"type": "string"},
            "operation": {"type": "string", "enum": sorted(ALLOWED_OPERATIONS)},
            "parameters": {"type": "object", "properties": parameter_properties},
        },
        "required": ["region_id", "target_prompt", "detected_issue", "operation", "parameters"],
    }
    return {
        "type": "object",
        "properties": {
            "iteration": {"type": "integer"},
            "reasoning": {"type": "string"},
            "actions": {"type": "array", "items": action_schema},
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
        # Structured output có thể trả null cho tham số không dùng → bỏ để planner gán default
        if isinstance(fields.get("parameters"), dict):
            fields["parameters"] = {
                key: value for key, value in fields["parameters"].items() if value is not None
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


def _full_image_action(issue: str, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Dựng một action rule-based áp dụng trên toàn ảnh."""
    return {
        "region_id": "full_image",
        "target_prompt": "full",
        "region_type": "full",
        "detected_issue": issue,
        "operation": operation,
        "parameters": parameters,
    }


def _region_action(
    region: str, issue: str, operation: str, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Dựng một action rule-based trên một vùng; executor tự suy ra loại vùng (D2)."""
    return {
        "region_id": region,
        "target_prompt": region,
        "region_type": None,
        # Viền mềm rộng để vùng được chỉnh sáng hòa vào nền, không lộ quầng
        "feather_radius": 30,
        "detected_issue": issue,
        "operation": operation,
        "parameters": parameters,
    }


def _region_exposure_actions(
    metrics: Dict[str, Any], diagnosis: Optional[DiagnosisReport]
) -> List[Dict[str, Any]]:
    """
    Luật theo vùng: chỉnh sáng riêng một vùng (ví dụ khuôn mặt ngược sáng) khi độ sáng
    toàn cục không cùng chiều lỗi, để không chỉnh sáng hai lần lên cùng một vùng.
    """
    if diagnosis is None:
        return []
    brightness_level = metrics.get("brightness_level")
    actions: List[Dict[str, Any]] = []
    for defect in diagnosis.actionable_defects:
        if defect.region == "full":
            continue
        if defect.type in ("underexposed", "backlit_subject"):
            if brightness_level == "underexposed":
                continue
            gamma = 1.6 if defect.severity >= 3 else 1.4
            actions.append(
                _region_action(defect.region, defect.type, "gamma_correct", {"gamma": gamma})
            )
        elif defect.type == "overexposed" and brightness_level != "overexposed":
            actions.append(
                _region_action(defect.region, defect.type, "gamma_correct", {"gamma": 0.8})
            )
    return actions


def _rule_based_plan(
    metrics: Dict[str, Any],
    iteration: int,
    source: PlanSource,
    reasoning: str,
    diagnosis: Optional[DiagnosisReport] = None,
) -> TreatmentPlan:
    """
    Kế hoạch dự phòng dựa trên ngưỡng phân loại của Module 1 (không cần mạng).
    Dùng khi không có GEMINI_API_KEY hoặc khi lời gọi/parse Gemini thất bại,
    để lỗi API không bị biến thành plan rỗng (= SHIP âm thầm).
    Nếu có chẩn đoán, thêm luật chỉnh sáng theo vùng từ số đo thực tế.
    """
    noise_level = metrics.get("noise_level")
    actions: List[Dict[str, Any]] = []

    if noise_level in ("medium", "severe"):
        actions.append(
            _full_image_action("high_noise", "denoise", {"method": "bilateral", "strength": 1.0})
        )

    if metrics.get("brightness_level") == "underexposed":
        actions.append(_full_image_action("underexposed", "gamma_correct", {"gamma": 1.3}))
    elif metrics.get("brightness_level") == "overexposed":
        actions.append(_full_image_action("overexposed", "gamma_correct", {"gamma": 0.8}))
    elif metrics.get("contrast_level") == "low":
        actions.append(_full_image_action("low_contrast", "clahe", {"clip_limit": 2.0}))

    actions += _region_exposure_actions(metrics, diagnosis)

    # Cố ý KHÔNG có luật sharpen/color_cast: đo trên data/real, sharpen làm giảm điểm chất lượng
    # và ám ấm thường là chủ ý (hoàng hôn, đồ ăn). Cần hiểu ngữ cảnh → để VLM/KB quyết định.
    for order, action in enumerate(actions, start=1):
        action["order"] = order

    return TreatmentPlan(
        iteration=iteration,
        reasoning=reasoning,
        actions=[RegionOperation(**action) for action in actions],
        source=source,
    )


def to_vlm_image(image: np.ndarray) -> Image.Image:
    """Thu nhỏ ảnh (giữ tỉ lệ) để cạnh dài không vượt VLM_MAX_SIDE rồi chuyển sang PIL."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > VLM_MAX_SIDE:
        scale = VLM_MAX_SIDE / longest
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


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


def _build_contents(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int,
    history: Optional[List[HistoryItem]],
    original_image: Optional[np.ndarray],
    diagnosis: Optional[DiagnosisReport] = None,
) -> List[Any]:
    """
    Dựng nội dung đa phương thức gửi Gemini (giai đoạn Plan): chỉ số kỹ thuật,
    chẩn đoán giai đoạn 1 (nếu có), phản hồi các vòng trước và ảnh.
    """
    prompt = f"Chỉ số kỹ thuật hiện tại:\n{json.dumps(metrics, indent=2, default=str)}\n"
    if diagnosis is not None:
        diagnosis_json = diagnosis.model_dump(mode="json", exclude={"source", "iteration"})
        prompt += (
            "CHẨN ĐOÁN (giai đoạn 1, kèm region_metrics đo thực tế theo vùng):\n"
            f"{json.dumps(diagnosis_json, indent=2, ensure_ascii=False)}\n"
        )
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


def call_gemini_json(
    api_key: str, system_prompt: str, contents: List[Any], schema: Dict[str, Any]
) -> Any:
    """
    Gọi Gemini với structured output (JSON theo schema) và trả về JSON đã parse.
    Dùng chung cho giai đoạn Perceive và Plan; mọi lỗi được ném ra để bên gọi fallback.
    """
    if genai is None or genai_types is None:
        raise RuntimeError("google-genai is not installed")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_json_schema=schema,
            # Không khai báo tool nào → tắt AFC để SDK không cảnh báo/không tự gọi hàm
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return _parse_json_text(response.text)


def diagnose_and_plan(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,
    diagnosis: Optional[DiagnosisReport] = None,
) -> TreatmentPlan:
    """
    Giai đoạn Plan: gọi Gemini lập kế hoạch điều trị dựa trên chỉ số và chẩn đoán.
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
        contents = _build_contents(image, metrics, iteration, history, original_image, diagnosis)
        data = call_gemini_json(api_key, SYSTEM_PROMPT, contents, PLAN_RESPONSE_SCHEMA)
        return _plan_from_vlm_json(data, iteration)

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
