"""
Rule engine offline dựa trên Knowledge Base.
- diagnose_from_metrics: chẩn đoán từ phân loại của Module 1 (không cần mạng).
- with_metric_defects: bổ sung lỗi đo từ chỉ số toàn cục vào một chẩn đoán có sẵn.
- instantiate: biến công thức của các card auto_apply thành action cho executor.
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from ..planner import OPERATION_ORDER
from ..regions import is_person_region
from ..state import Defect, DiagnosisReport, PlanSource
from .models import KnowledgeBase, ParamSpec, ParamValue, RecipeStep
from .retriever import CardMatch, face_located, match_cards

logger = logging.getLogger(__name__)

# Mức độ mặc định khi card không gắn với lỗi nào (card theo loại cảnh)
DEFAULT_SEVERITY = 2
# Lỗi lấy severity hiệu chỉnh từ TechnicalMetrics.defect_severity, theo thứ tự báo cáo
CALIBRATED_DEFECTS = (
    "noise",
    "underexposed",
    "overexposed",
    "low_contrast",
    "blur",
    "color_cast_warm",
    "color_cast_cool",
    "color_cast_green",
)
DEFAULT_FEATHER = 15
# Tạm thời, chưa hiệu chỉnh (BL-10): có thêm một thao tác nhẹ trên chủ thể người bên cạnh thao
# tác trên mặt hay không. Mặc định tắt; R-09 so sánh các cấu hình rồi mới chọn.
BACKLIT_LIGHT_PERSON_ENABLED = False
# Tạm thời, chưa hiệu chỉnh (BL-10): thao tác nhẹ trên người dùng bấy nhiêu phần độ lệch gamma
BACKLIT_LIGHT_PERSON_SCALE = 0.4


class StepTarget(NamedTuple):
    """Một nơi bước công thức được áp dụng: vùng, mức, lỗi và cách nhắm tới lỗi đó."""

    region: str
    severity: int
    issue: str
    defect_ref: Optional[str] = None
    face_mode: Optional[str] = None
    param_scale: float = 1.0
    note: str = ""


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
    """Lỗi toàn ảnh suy ra từ Module 1.

    Có `defect_severity` (severity theo bảng hiệu chỉnh P2, cùng nguồn với các *_level — quyết
    định D4) thì dùng thẳng; chỉ số cũ không có trường này thì suy từ các mức phân loại.
    """
    defects: List[Defect] = []

    def add(defect_type: str, severity: int, evidence: str) -> None:
        """Thêm một lỗi toàn ảnh suy ra từ chỉ số Module 1."""
        defects.append(
            Defect(type=defect_type, severity=severity, evidence=evidence, origin="rule")
        )

    calibrated = metrics.get("defect_severity")
    if calibrated:
        version = metrics.get("calibration_version") or "?"
        for defect_type in CALIBRATED_DEFECTS:
            severity = int(calibrated.get(defect_type, 0))
            if severity >= 1:
                add(defect_type, severity, f"{defect_type}={severity} (calibration {version})")
        return defects

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


def _scaled(operation: str, params: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """Giảm cường độ về phía trung hòa (hiện chỉ gamma: 1 + (gamma − 1) × scale)."""
    if scale == 1.0 or operation != "gamma_correct" or "gamma" not in params:
        return params
    return {**params, "gamma": round(1.0 + (float(params["gamma"]) - 1.0) * scale, 3)}


def _action(step: RecipeStep, target: StepTarget) -> Dict[str, Any]:
    """Dựng action dạng dict (cho RegionOperation) từ một bước công thức tại một vùng."""
    region = target.region
    params = {name: param_value(spec, target.severity) for name, spec in step.params.items()}
    action: Dict[str, Any] = {
        "region_id": "full_image" if region == "full" else region,
        "target_prompt": region,
        # Vùng cụ thể → None để executor tự suy ra loại vùng (D2)
        "region_type": "full" if region == "full" else ("face" if region == "face" else None),
        "feather_radius": step.feather_radius or DEFAULT_FEATHER,
        "detected_issue": target.issue,
        "operation": step.operation,
        "parameters": _scaled(step.operation, params, target.param_scale),
    }
    if target.defect_ref:
        action["defect_ref"] = target.defect_ref
    face_mode = target.face_mode or step.face_mode
    if face_mode:
        action["face_mode"] = face_mode
    return action


def step_targets(step: RecipeStep, match: CardMatch, has_face: bool = False) -> List[StepTarget]:
    """
    Các nơi bước công thức được áp dụng. has_face: Perceive đã định vị được khuôn mặt; dùng cho
    bước face_when_person (lỗi ở chủ thể người → nhắm riêng mặt, hoặc giảm một bậc nếu không có
    mặt).
    """
    if step.region != "$defect":
        severity = max((d.severity for d in match.bindings), default=DEFAULT_SEVERITY)
        issue = match.bindings[0].type if match.bindings else match.card.id
        return [StepTarget(step.region, severity, issue)]
    found: Dict[str, Tuple[int, str]] = {}
    for defect in match.bindings:
        previous = found.get(defect.region)
        if previous is None or defect.severity > previous[0]:
            found[defect.region] = (defect.severity, defect.type)
    targets: List[StepTarget] = []
    for region, (severity, issue) in found.items():
        ref = f"{issue}@{region}"
        if not (step.face_when_person and is_person_region(region)):
            targets.append(StepTarget(region, severity, issue, ref))
        elif has_face:
            note = f"[{match.card.id}] Đã định vị khuôn mặt: chỉnh riêng 'face' (oval) cho {ref}."
            targets.append(StepTarget("face", severity, issue, ref, "oval", note=note))
            if BACKLIT_LIGHT_PERSON_ENABLED:
                scale = BACKLIT_LIGHT_PERSON_SCALE
                targets.append(StepTarget(region, severity, issue, ref, None, scale))
        else:
            lowered = max(1, severity - 1)
            note = (
                f"[{match.card.id}] Không định vị được khuôn mặt trong '{region}': giữ recipe "
                f"trên '{region}', giảm một bậc (mức {severity} → {lowered}) cho {ref}."
            )
            targets.append(StepTarget(region, lowered, issue, ref, note=note))
    return targets


def instantiate(
    matches: Sequence[CardMatch],
    auto_only: bool = True,
    has_face: bool = False,
    notes: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Sinh action từ công thức của các card (mặc định chỉ card auto_apply).
    Card priority cao hơn thắng khi hai card cùng (operation, vùng). Action được sắp theo
    thứ tự thực thi an toàn của planner.
    has_face: Perceive đã định vị được khuôn mặt (xem step_targets). notes: nếu truyền vào, nhận
    ghi chú lý do chọn vùng (nhắm mặt, hoặc giảm một bậc vì không thấy mặt).

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
            for target in step_targets(step, match, has_face):
                key = (step.operation, target.region)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(_action(step, target))
                if target.note:
                    logger.info(target.note)
                    if notes is not None and target.note not in notes:
                        notes.append(target.note)
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
    notes: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Kế hoạch offline: chẩn đoán (+ lỗi từ chỉ số) → card khớp → action."""
    effective = (
        with_metric_defects(diagnosis, metrics)
        if diagnosis is not None
        else diagnose_from_metrics(metrics)
    )
    return instantiate(
        match_cards(effective, metrics, kb), has_face=face_located(effective), notes=notes
    )
