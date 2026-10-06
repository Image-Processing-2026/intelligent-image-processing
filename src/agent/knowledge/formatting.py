"""
Định dạng tri thức truy xuất thành khối 'TRI THỨC CHUYÊN MÔN' cho prompt giai đoạn Plan.
Tham số được chọn sẵn theo severity của chính ca đang xét (cùng quy tắc với rule engine),
để VLM không phải tự suy ra giá trị nào ứng với mức nào.
"""

from typing import List

from .engine import param_value, step_targets
from .models import ParamSpec, RecipeStep
from .retriever import CardMatch, KnowledgeContext


def _range_hint(spec: ParamSpec) -> str:
    if not isinstance(spec, dict) or len(spec) < 2:
        return ""
    steps = ", ".join(f"mức {severity}→{value}" for severity, value in sorted(spec.items()))
    return f" [{steps}]"


def _describe_step(step: RecipeStep, match: CardMatch) -> List[str]:
    lines: List[str] = []
    for region, severity, _ in step_targets(step, match):
        params = ", ".join(
            f"{name}={param_value(spec, severity)}{_range_hint(spec)}"
            for name, spec in step.params.items()
        )
        where = "toàn ảnh" if region == "full" else f"vùng '{region}'"
        feather = f", feather_radius={step.feather_radius}" if step.feather_radius else ""
        lines.append(f"{step.operation} trên {where} (mức {severity}): {params}{feather}")
    return lines


def format_context(context: KnowledgeContext) -> str:
    """Khối 'TRI THỨC CHUYÊN MÔN' đưa vào prompt giai đoạn Plan; rỗng nếu không có gì."""
    if not context.cards and not context.principles:
        return ""
    lines = ["TRI THỨC CHUYÊN MÔN (đã lọc theo chẩn đoán; tham số đã chọn theo mức độ của ca này):"]
    for match in context.cards:
        card = match.card
        bound = ", ".join(f"{d.type}@{d.region} (mức {d.severity})" for d in match.bindings)
        lines.append(f"[{card.id}] {card.title} — áp dụng cho: {bound or ', '.join(card.scenes)}")
        steps = [text for step in card.recipe for text in _describe_step(step, match)]
        if steps:
            lines.append("  Công thức: " + "; rồi ".join(steps))
        for item in card.avoid:
            lines.append(f"  Tránh: {item}")
        if card.rationale:
            lines.append(f"  Vì sao: {card.rationale}")
    for principle in context.principles:
        lines.append(f"Nguyên lý — {principle.title}: {principle.text}")
    return "\n".join(lines)
