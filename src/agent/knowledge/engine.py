"""
Rule engine offline dựa trên Knowledge Base.
- diagnose_from_metrics: chẩn đoán từ phân loại của Module 1 (không cần mạng).
- with_metric_defects: bổ sung lỗi đo từ chỉ số toàn cục vào một chẩn đoán có sẵn.
- instantiate: biến công thức của các card auto_apply thành action cho executor.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..planner import OPERATION_ORDER
from ..state import Defect, DiagnosisReport, PlanSource
from .models import KnowledgeBase, ParamSpec, ParamValue, RecipeStep
from .retriever import CardMatch, match_cards

# Mức độ mặc định khi card không gắn với lỗi nào (card theo loại cảnh)
DEFAULT_SEVERITY = 2
DEFAULT_FEATHER = 15


def diagnose_from_metrics(
    metrics: Dict[str, Any], iteration: int = 1, source: PlanSource = "rule_based"
) -> DiagnosisReport:
    """Chẩn đoán từ phân loại của Module 1 (không cần mạng); luôn đo thêm vùng khuôn mặt."""
    defects = _metric_defects(metrics)
    listed = ", ".join(f"{d.type} (mức {d.severity})" for d in defects) or "không có lỗi rõ rệt"
    return DiagnosisReport(
        iteration=iteration,
        subjects=["face"],
        defects=defects,
        summary=f"Chẩn đoán Rule-Based từ chỉ số toàn cục: {listed}.",
        source=source,
    )


def _metric_defects(metrics: Dict[str, Any]) -> List[Defect]:
    """Lỗi toàn ảnh suy ra từ các mức phân loại của Module 1."""
    defects: List[Defect] = []

    def add(defect_type: str, severity: int, evidence: str) -> None:
        defects.append(
            Defect(type=defect_type, severity=severity, evidence=evidence, origin="rule")
        )

    noise_severity = {"low": 1, "medium": 2, "severe": 3}.get(metrics.get("noise_level") or "")
    if noise_severity:
        add("noise", noise_severity, f"noise_level={metrics['noise_level']}")

    brightness = float(metrics.get("brightness_mean", 128.0))
    if metrics.get("brightness_level") == "underexposed":
        add("underexposed", 3 if brightness < 40 else 2, f"brightness_mean={brightness:.1f}")
    elif metrics.get("brightness_level") == "overexposed":
        add("overexposed", 3 if brightness > 215 else 2, f"brightness_mean={brightness:.1f}")

    if metrics.get("contrast_level") == "low":
        contrast = float(metrics.get("contrast_std", 40.0))
        add("low_contrast", 2 if contrast < 25 else 1, f"contrast_std={contrast:.1f}")

    blur_severity = {"mild_blur": 1, "severe_blur": 2}.get(metrics.get("blur_level") or "")
    if blur_severity:
        add("blur", blur_severity, f"blur_level={metrics['blur_level']}")

    cast_type = {
        "warm": "color_cast_warm",
        "cool": "color_cast_cool",
        "greenish": "color_cast_green",
    }.get(metrics.get("color_cast") or "")
    if cast_type:
        add(cast_type, 1, f"color_cast={metrics['color_cast']}")
    return defects


def with_metric_defects(diagnosis: DiagnosisReport, metrics: Dict[str, Any]) -> DiagnosisReport:
    """
    Bản sao chẩn đoán có thêm lỗi toàn ảnh từ chỉ số Module 1 mà chẩn đoán chưa nêu.
    Dùng cho kế hoạch dự phòng: số đo là bằng chứng bổ sung; preserve của chẩn đoán vẫn chặn
    các card tương ứng (ví dụ low_key chặn card làm sáng toàn ảnh).
    """
    present = {(defect.type, defect.region) for defect in diagnosis.defects}
    extra = [defect for defect in _metric_defects(metrics) if (defect.type, "full") not in present]
    return diagnosis.model_copy(update={"defects": [*diagnosis.defects, *extra]})


def param_value(spec: ParamSpec, severity: int) -> ParamValue:
    """Giá trị theo severity: khóa đúng mức, nếu không có thì mức thấp hơn gần nhất."""
    if not isinstance(spec, dict):
        return spec
    lower = [key for key in spec if key <= severity]
    return spec[max(lower)] if lower else spec[min(spec)]


def _action(step: RecipeStep, region: str, severity: int, issue: str) -> Dict[str, Any]:
    params = {name: param_value(spec, severity) for name, spec in step.params.items()}
    action: Dict[str, Any] = {
        "region_id": "full_image" if region == "full" else region,
        "target_prompt": region,
        # Vùng cụ thể → None để executor tự suy ra loại vùng (D2)
        "region_type": "full" if region == "full" else None,
        "feather_radius": step.feather_radius or DEFAULT_FEATHER,
        "detected_issue": issue,
        "operation": step.operation,
        "parameters": params,
    }
    if step.face_mode:
        action["face_mode"] = step.face_mode
    return action


def step_targets(step: RecipeStep, match: CardMatch) -> List[Tuple[str, int, str]]:
    """(vùng, severity, lỗi) cho từng nơi bước công thức được áp dụng."""
    if step.region != "$defect":
        severity = max((d.severity for d in match.bindings), default=DEFAULT_SEVERITY)
        issue = match.bindings[0].type if match.bindings else match.card.id
        return [(step.region, severity, issue)]
    targets: Dict[str, Tuple[int, str]] = {}
    for defect in match.bindings:
        previous = targets.get(defect.region)
        if previous is None or defect.severity > previous[0]:
            targets[defect.region] = (defect.severity, defect.type)
    return [(region, severity, issue) for region, (severity, issue) in targets.items()]


def instantiate(
    matches: Sequence[CardMatch], auto_only: bool = True
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Sinh action từ công thức của các card (mặc định chỉ card auto_apply).
    Card priority cao hơn thắng khi hai card cùng (operation, vùng). Action được sắp theo
    thứ tự thực thi an toàn của planner.

    Returns:
        (actions, card_ids): action dạng dict cho RegionOperation và id các card đã dùng.
    """
    ordered = sorted(matches, key=lambda m: (m.card.priority, m.score), reverse=True)
    actions: List[Dict[str, Any]] = []
    used: List[str] = []
    seen: set = set()
    for match in ordered:
        if auto_only and not match.card.auto_apply:
            continue
        for step in match.card.recipe:
            for region, severity, issue in step_targets(step, match):
                key = (step.operation, region)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(_action(step, region, severity, issue))
                if match.card.id not in used:
                    used.append(match.card.id)
    actions.sort(key=lambda action: OPERATION_ORDER.get(action["operation"], 99))
    for order, action in enumerate(actions, start=1):
        action["order"] = order
    return actions, used


def plan_actions_from_knowledge(
    diagnosis: Optional[DiagnosisReport],
    metrics: Dict[str, Any],
    kb: Optional[KnowledgeBase] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Kế hoạch offline: chẩn đoán (+ lỗi từ chỉ số) → card khớp → action."""
    effective = (
        with_metric_defects(diagnosis, metrics)
        if diagnosis is not None
        else diagnose_from_metrics(metrics)
    )
    return instantiate(match_cards(effective, metrics, kb))
