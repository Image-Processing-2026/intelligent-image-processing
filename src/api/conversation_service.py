"""Hội thoại và phương án bền vững, tách biệt hoàn toàn với thực thi ảnh."""

import json
import os
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from src.agent.fusion import plan_contraindications
from src.agent.graph import plan_treatment
from src.agent.planner import (
    PARAMETER_BOUNDS,
    apply_contraindications,
    apply_preserve_guard,
    sanitize_actions,
)
from src.agent.refine import apply_adjustments, parse_feedback
from src.agent.state import DiagnosisReport, RegionOperation, TreatmentPlan
from src.agent.user_analysis import apply_edits, parse_edits
from src.agent.vlm_diagnostician import call_gemini_json


class ProposalAction(RegionOperation):
    """Giới hạn thao tác hội thoại trong toolbox thực thi."""

    operation: Literal["denoise", "gamma_correct", "clahe", "sharpen", "color_correct"]
    detected_issue: str = "user_request"


class ConversationReply(BaseModel):
    """Đầu ra hội thoại được kiểm tra trước khi lưu vào phiên."""

    reply: str
    changes_plan: bool = False
    needs_clarification: bool = False
    source_mode: Literal["rebuild", "incremental"] = "rebuild"
    source_reason: str = "Dựng lại từ ảnh gốc để tránh chồng hiệu ứng."
    actions: list[ProposalAction] = Field(default_factory=list)
    requirements: list[str] | None = None


def update_proposal(
    previous: dict[str, Any],
    message: str,
    diagnosis: DiagnosisReport,
    original: np.ndarray,
    current: np.ndarray | None,
    base_actions: list[RegionOperation],
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], DiagnosisReport, str]:
    """Đọc trạng thái và ảnh liên quan; chỉ tạo phương án, không gọi executor."""
    actions = [RegionOperation.model_validate(a) for a in previous.get("actions", [])]
    if "actions" not in previous:
        actions = base_actions or plan_treatment(original, {}, diagnosis).actions
    labels = {
        "denoise": "giảm nhiễu",
        "gamma_correct": "điều chỉnh độ sáng",
        "clahe": "cải thiện tương phản",
        "sharpen": "tăng độ nét",
        "color_correct": "điều chỉnh màu sắc",
    }
    steps = "; ".join(f"{labels.get(a.operation, a.operation)} tại {a.region_id}" for a in actions)
    reply = (
        (diagnosis.summary + " Đề xuất: " + steps + ". Các thay đổi này chưa được áp dụng.")
        if steps
        else (
            diagnosis.summary
            + " Hiện chưa có thao tác cần áp dụng. Bạn có thể trao đổi thêm về mục tiêu chỉnh ảnh."
        )
    )
    changed = "actions" not in previous
    ready = previous.get("status") != "applied"
    mode = previous.get("source_mode", "rebuild")
    reason = previous.get(
        "source_reason", "Dùng ảnh gốc để áp dụng đầy đủ phương án và tránh chồng hiệu ứng."
    )
    requirements = list(previous.get("requirements", []))
    if message:
        ready = True
        key = os.getenv("GEMINI_API_KEY")
        if key:
            context = {
                "diagnosis": diagnosis.model_dump(mode="json"),
                "proposal": previous,
                "base_actions": [a.model_dump() for a in base_actions],
                "messages": messages[-30:],
                "message": message,
                "has_result": current is not None,
                "parameter_bounds": PARAMETER_BOUNDS,
            }
            contents: list[Any] = [json.dumps(context, ensure_ascii=False)]
            # Ảnh đầu vào thật giúp hiểu phản hồi mà không render thử trong lúc chat.
            from PIL import Image

            contents.append(Image.fromarray(original))
            if current is not None:
                contents.append(Image.fromarray(current))
            result = ConversationReply.model_validate(
                call_gemini_json(
                    key,
                    "Bạn là trợ lý xử lý ảnh. Trả lời bằng tiếng Việt tự nhiên. Đọc diagnosis, phương án, yêu cầu và hội thoại. "
                    "Câu hỏi giải thích không đổi phương án. Nếu mơ hồ hãy hỏi lại. Không nói đã sửa ảnh khi chỉ đề xuất. "
                    "Giữ requirements đã thống nhất, cập nhật khi user thay ý. Chỉ dùng actions trong schema với tham số hợp lệ. "
                    "source_mode chỉ rebuild hoặc incremental. Rebuild: toàn bộ actions từ ảnh gốc, giữ chỉnh sửa đã chấp nhận. "
                    "Incremental: chỉ actions bổ sung trên kết quả hiện tại. Giảm/hoàn tác hiệu ứng cũ phải rebuild. "
                    "Chọn incremental khi chỉ bổ sung và giữ kết quả. Nếu changes_plan false thì giữ phương án. "
                    "Nếu thay đổi, actions là phương án đầy đủ theo source_mode đã chọn. Không thêm công cụ ngoài schema.",
                    contents,
                    ConversationReply.model_json_schema(),
                )
            )
            reply = result.reply
            ready = not result.needs_clarification
            if result.requirements is not None:
                requirements = result.requirements
            if result.changes_plan:
                if result.source_mode not in ("rebuild", "incremental") or (
                    result.source_mode == "incremental" and current is None
                ):
                    raise ValueError("Invalid proposal source mode")
                actions = sanitize_actions(result.actions)
                mode, reason = result.source_mode, result.source_reason
                changed = True
            if current is None and result.changes_plan:
                edits, _ = parse_edits(message, diagnosis)
                diagnosis = apply_edits(diagnosis, edits)
        else:
            edits, _ = parse_edits(message, diagnosis)
            adjustments, _ = parse_feedback(message, original, current, actions)
            if current is None and edits:
                diagnosis = apply_edits(diagnosis, edits)
                actions = plan_treatment(original, {}, diagnosis).actions
                changed = True
            elif adjustments:
                actions, notes = apply_adjustments(actions or base_actions, adjustments)
                reply = "Phương án tiếp theo: " + "; ".join(notes) + ". Chưa áp dụng lên ảnh."
                changed = True
                mode = "rebuild"
            else:
                ready = False
                reply = "Chế độ hiện tại chưa hiểu đủ yêu cầu này. Bạn hãy nói rõ vùng ảnh và điều muốn thay đổi; hệ thống chưa xử lý ảnh."
        if changed and not key and edits:
            reply = "Mình đã cập nhật phân tích và phương án theo góp ý. Các thay đổi đang chờ bạn bấm nút xử lý."
        if changed:
            requirements = list(dict.fromkeys([*requirements, message]))
    plan = apply_preserve_guard(
        TreatmentPlan(reasoning=reply, actions=sanitize_actions(actions)), diagnosis.preserve
    )
    plan = apply_contraindications(plan, plan_contraindications(diagnosis))
    revision = int(previous.get("revision", 0)) + 1
    proposal = {
        **previous,
        "revision": revision,
        "actions": [a.model_dump(mode="json") for a in plan.actions],
        "diagnosis": diagnosis.model_dump(mode="json"),
        "requirements": requirements,
        "source_mode": mode,
        "source_reason": reason,
        "reply": reply,
        "status": "ready"
        if ready and (changed or previous.get("status") != "applied")
        else "needs_clarification",
    }
    return proposal, diagnosis, reply
