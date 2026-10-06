"""
Kiểm tra tính hợp lệ của kế hoạch điều trị (Plan Validator).
Đảm bảo Agent chỉ sử dụng các công cụ trong Toolbox được phép và đúng thứ tự ưu tiên.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .state import PreserveItem, RegionOperation, TreatmentPlan

logger = logging.getLogger(__name__)

ALLOWED_OPERATIONS: Set[str] = {
    "denoise",
    "gamma_correct",
    "clahe",
    "sharpen",
    "color_correct",
}

# Bảng ràng buộc tham số an toàn cho từng operation
PARAMETER_BOUNDS: Dict[str, Dict[str, Any]] = {
    "denoise": {
        "strength": {"min": 0.1, "max": 2.0, "default": 1.0},
        "method": {"allowed": ["gaussian", "median", "bilateral", "nlm"], "default": "bilateral"},
    },
    "gamma_correct": {
        "gamma": {"min": 0.5, "max": 2.5, "default": 1.2},
    },
    "clahe": {
        "clip_limit": {"min": 1.0, "max": 4.0, "default": 2.0},
    },
    "sharpen": {
        "amount": {"min": 0.2, "max": 2.0, "default": 1.0},
        "method": {"allowed": ["unsharp_mask", "laplacian"], "default": "unsharp_mask"},
    },
    "color_correct": {
        "saturation_scale": {"min": 0.5, "max": 1.5, "default": 1.0},
        "temperature_shift": {"min": -1.0, "max": 1.0, "default": 0.0},
    },
}


# Bảng ràng buộc các trường vùng (Module 2) mà VLM có thể đề xuất.
# Kẹp ở đây thay vì ge/le trong Pydantic để một giá trị sai không xóa cả kế hoạch.
REGION_FIELD_BOUNDS: Dict[str, Dict[str, Any]] = {
    # min=5: ADR-003 yêu cầu mask luôn được làm mềm, không cho VLM tắt feather
    "feather_radius": {"min": 5, "max": 50, "default": 15, "integer": True},
    "expand_ratio": {"min": 0.0, "max": 1.0, "default": 0.15},
    "num_faces": {"min": 1, "max": 10, "default": 4, "integer": True},
    "box_threshold": {"min": 0.0, "max": 1.0, "default": 0.35},
    "text_threshold": {"min": 0.0, "max": 1.0, "default": 0.25},
    "nms_iou_threshold": {"min": 0.0, "max": 1.0, "default": 0.8},
    # bbox/binary_mask chỉ dùng trong tiến trình, VLM không cung cấp được → suy ra (D2)
    "region_type": {"allowed": [None, "full", "face", "spatial", "semantic"], "default": None},
    "quadrant": {
        "allowed": [None, "top", "bottom", "left", "right", "center", "giữa"],
        "default": None,
    },
    # sam_refined chưa được bật trong bản build này → dùng bbox
    "face_mode": {"allowed": ["bbox", "oval"], "default": "bbox"},
    "merge_policy": {"allowed": ["max"], "default": "max"},
    "instance_selection": {"allowed": ["all", "largest", "index"], "default": "all"},
}

# Các trường chỉ hợp lệ trong tiến trình (ndarray, toạ độ pixel); luôn bỏ khỏi đầu ra VLM.
IN_PROCESS_REGION_FIELDS: tuple[str, ...] = ("bbox", "binary_mask")


def _clamp_number(field: str, raw_val: Any, constraint: Dict[str, Any]) -> float | int:
    """Ép kiểu số và kẹp về [min, max]; giá trị không phải số hữu hạn → default."""
    try:
        val = float(raw_val)
    except (TypeError, ValueError):
        val = math.nan
    if isinstance(raw_val, bool) or not math.isfinite(val):
        logger.warning(
            "Invalid value '%s' for region.%s. Resetting to default %s.",
            raw_val,
            field,
            constraint["default"],
        )
        return constraint["default"]
    clamped_val = max(constraint["min"], min(constraint["max"], val))
    if constraint.get("integer"):
        clamped_val = int(round(clamped_val))
    if clamped_val != val:
        logger.warning(
            "Clamped region.%s from %s to %s (bounds: [%s, %s]).",
            field,
            raw_val,
            clamped_val,
            constraint["min"],
            constraint["max"],
        )
    return clamped_val


def _clamp_instance_index(fields: Dict[str, Any]) -> None:
    """instance_index chỉ hợp lệ khi instance_selection='index' và là số nguyên >= 0."""
    raw_index = fields.get("instance_index")
    if fields["instance_selection"] != "index":
        if raw_index is not None:
            logger.warning(
                "Dropping region.instance_index=%s because instance_selection is '%s'.",
                raw_index,
                fields["instance_selection"],
            )
        fields["instance_index"] = None
        return
    index: Optional[int] = None
    if not isinstance(raw_index, bool):
        try:
            value = float(raw_index)
            if math.isfinite(value) and value >= 0 and value == int(value):
                index = int(value)
        except (TypeError, ValueError):
            index = None
    if index is None:
        logger.warning(
            "Invalid region.instance_index '%s'. Resetting instance_selection to 'all'.",
            raw_index,
        )
        fields["instance_selection"] = "all"
    fields["instance_index"] = index


def clamp_region_fields(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Kẹp các trường vùng của một action về khoảng an toàn (tương tự clamp_parameters).
    Nhận dict thô từ VLM hoặc từ RegionOperation; các khóa khác được giữ nguyên.
    - Số vượt biên → kéo về min/max; không phải số → default.
    - Enum không hợp lệ → default (region_type → None để executor suy ra, D2).
    - bbox/binary_mask luôn bị đặt về None vì VLM không thể cung cấp hợp lệ.
    - Trường bị thiếu → bổ sung giá trị default.
    """
    clamped = dict(fields)

    for field, constraint in REGION_FIELD_BOUNDS.items():
        raw_val = clamped.get(field)
        if "allowed" in constraint:
            val = raw_val.strip().casefold() if isinstance(raw_val, str) else raw_val
            if val not in constraint["allowed"]:
                # Trường bị bỏ trống (None) là hợp lệ → gán default, không cảnh báo
                if raw_val is not None:
                    logger.warning(
                        "Invalid value '%s' for region.%s. Resetting to default '%s'.",
                        raw_val,
                        field,
                        constraint["default"],
                    )
                val = constraint["default"]
            clamped[field] = val
        elif raw_val is None:
            clamped[field] = constraint["default"]
        else:
            clamped[field] = _clamp_number(field, raw_val, constraint)

    _clamp_instance_index(clamped)

    for field in IN_PROCESS_REGION_FIELDS:
        if clamped.get(field) is not None:
            logger.warning("Dropping region.%s supplied by the plan (in-process only).", field)
        clamped[field] = None

    return clamped


def clamp_parameters(operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kẹp (clamp) tham số của một thao tác về khoảng an toàn.
    Nếu tham số vượt biên → kéo về giá trị min/max.
    Nếu tham số enum không hợp lệ → gán về default.
    Nếu tham số bị thiếu → bổ sung giá trị default.
    """
    bounds = PARAMETER_BOUNDS.get(operation, {})
    clamped = dict(params) if params else {}

    for param_name, constraint in bounds.items():
        if "allowed" in constraint:
            # Validate enum
            val = clamped.get(param_name)
            if val not in constraint["allowed"]:
                if param_name in clamped:
                    logger.warning(
                        "Invalid enum value '%s' for %s.%s. Resetting to default '%s'.",
                        val,
                        operation,
                        param_name,
                        constraint["default"],
                    )
                clamped[param_name] = constraint["default"]
        elif "min" in constraint and "max" in constraint:
            # Clamp numeric
            raw_val = clamped.get(param_name)
            if raw_val is None:
                clamped[param_name] = constraint["default"]
            else:
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    val = constraint["default"]
                clamped_val = max(constraint["min"], min(constraint["max"], val))
                if clamped_val != val:
                    logger.warning(
                        "Clamped %s.%s from %s to %s (bounds: [%s, %s]).",
                        operation,
                        param_name,
                        raw_val,
                        clamped_val,
                        constraint["min"],
                        constraint["max"],
                    )
                clamped[param_name] = clamped_val

    return clamped


def validate_and_sort_plan(plan: TreatmentPlan) -> TreatmentPlan:
    """
    Xác thực kế hoạch điều trị:
    1. Loại bỏ các thao tác không nằm trong Toolbox.
    2. Kẹp (clamp) tham số và các trường vùng về khoảng an toàn.
    3. Sắp xếp thứ tự ưu tiên (khử nhiễu -> cân bằng sáng -> CLAHE -> làm nét -> chỉnh màu).
    """
    valid_actions: List[RegionOperation] = []

    priority_map = {
        "denoise": 10,
        "gamma_correct": 20,
        "clahe": 30,
        "sharpen": 40,
        "color_correct": 50,
    }

    for action in plan.actions:
        op_name = action.operation.lower().strip()
        if op_name in ALLOWED_OPERATIONS:
            # Kẹp tham số về khoảng an toàn
            action.parameters = clamp_parameters(op_name, action.parameters)
            # Kẹp các trường vùng (Module 2) và bỏ bbox/binary_mask
            region_fields = {
                name: getattr(action, name)
                for name in (*REGION_FIELD_BOUNDS, "instance_index", *IN_PROCESS_REGION_FIELDS)
            }
            for name, value in clamp_region_fields(region_fields).items():
                setattr(action, name, value)
            # Gán lại độ ưu tiên mặc định nếu chưa được sắp xếp
            calculated_priority = priority_map.get(op_name, 99)
            action.order = calculated_priority
            valid_actions.append(action)

    # Sắp xếp danh sách hành động theo thứ tự an toàn
    valid_actions.sort(key=lambda x: x.order)
    plan.actions = valid_actions
    return plan


# ---------------------------------------------------------
# Preserve guard: giữ các đặc điểm thẩm mỹ có chủ ý (giai đoạn Perceive)
# ---------------------------------------------------------
# (operation, tham số, điều kiện vi phạm, giá trị trung hòa); tham số None → bỏ cả action
_PreserveRule = Tuple[str, Optional[str], Optional[Callable[[float], bool]], Optional[float]]

PRESERVE_RULES: Dict[str, List[_PreserveRule]] = {
    "warm_tone": [("color_correct", "temperature_shift", lambda v: v < 0, 0.0)],
    "cool_tone": [("color_correct", "temperature_shift", lambda v: v > 0, 0.0)],
    "muted_colors": [("color_correct", "saturation_scale", lambda v: v > 1, 1.0)],
    "vivid_colors": [("color_correct", "saturation_scale", lambda v: v < 1, 1.0)],
    "low_key": [("gamma_correct", "gamma", lambda v: v > 1, 1.0), ("clahe", None, None, None)],
    "high_key": [("gamma_correct", "gamma", lambda v: v < 1, 1.0)],
    "silhouette": [("gamma_correct", "gamma", lambda v: v > 1, 1.0), ("clahe", None, None, None)],
    "film_grain": [("denoise", None, None, None)],
    "soft_focus": [("sharpen", None, None, None)],
}
# Aspect về kết cấu: preserve trên toàn ảnh áp dụng cho mọi vùng
_TEXTURE_ASPECTS = frozenset(("film_grain", "soft_focus"))
# Giá trị không tác dụng của từng operation có tham số trung hòa được
_NEUTRAL_PARAMETERS: Dict[str, Dict[str, float]] = {
    "gamma_correct": {"gamma": 1.0},
    "color_correct": {"saturation_scale": 1.0, "temperature_shift": 0.0},
}


def _action_region(action: RegionOperation) -> str:
    """Tên vùng của action để so với preserve; vùng toàn ảnh → 'full'."""
    # Import muộn: executor kéo theo Module 2/3, planner cần nhẹ khi import
    from .executor import _infer_region_kind

    if _infer_region_kind(action) == "full":
        return "full"
    return action.target_prompt.strip().casefold()


def _in_scope(action_region: str, item: PreserveItem) -> bool:
    """
    Preserve toàn ảnh áp dụng cho action toàn ảnh (và mọi vùng nếu là aspect kết cấu);
    preserve một vùng áp dụng cho action trên vùng đó và action toàn ảnh (vì cũng tác động lên nó).
    """
    if item.region == "full":
        return action_region == "full" or item.aspect in _TEXTURE_ASPECTS
    return action_region in (item.region, "full")


def _is_neutral(action: RegionOperation) -> bool:
    """True nếu mọi tham số tác dụng của action đều ở giá trị trung hòa (action vô hiệu)."""
    neutral = _NEUTRAL_PARAMETERS.get(action.operation)
    if neutral is None:
        return False
    return all(
        abs(float(action.parameters.get(name, value)) - value) < 1e-6
        for name, value in neutral.items()
    )


def apply_preserve_guard(plan: TreatmentPlan, preserve: Sequence[PreserveItem]) -> TreatmentPlan:
    """
    Chặn tất định các thao tác phá vỡ đặc điểm cần giữ: trung hòa tham số vi phạm
    (ví dụ temperature_shift < 0 khi giữ warm_tone) hoặc bỏ cả action (ví dụ denoise khi
    giữ film_grain). Action trở thành vô hiệu sau khi trung hòa cũng bị bỏ.
    """
    if not preserve:
        return plan

    kept: List[RegionOperation] = []
    notes: List[str] = []
    for action in plan.actions:
        region = _action_region(action)
        dropped = False
        for item in preserve:
            if not _in_scope(region, item):
                continue
            for operation, parameter, violates, neutral in PRESERVE_RULES[item.aspect]:
                if operation != action.operation:
                    continue
                if parameter is None:
                    dropped = True
                    notes.append(f"bỏ {action.operation} trên '{region}' để giữ {item.aspect}")
                    continue
                value = action.parameters.get(parameter)
                if value is not None and violates is not None and violates(float(value)):
                    action.parameters[parameter] = neutral
                    notes.append(
                        f"đặt {action.operation}.{parameter}={neutral} trên '{region}' "
                        f"để giữ {item.aspect}"
                    )
        if not dropped and _is_neutral(action):
            dropped = True
        if dropped:
            logger.warning("Preserve guard removed '%s' on region '%s'.", action.operation, region)
            continue
        kept.append(action)

    plan.actions = kept
    if notes:
        plan.reasoning = f"{plan.reasoning}\n[Giữ chủ ý thẩm mỹ] " + "; ".join(notes) + "."
    return plan
