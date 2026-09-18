"""
Kiểm tra tính hợp lệ của kế hoạch điều trị (Plan Validator).
Đảm bảo Agent chỉ sử dụng các công cụ trong Toolbox được phép và đúng thứ tự ưu tiên.
"""

import logging
from typing import Any, Dict, List, Set

from .state import RegionOperation, TreatmentPlan

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
    2. Kẹp (clamp) tham số về khoảng an toàn.
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
            # Gán lại độ ưu tiên mặc định nếu chưa được sắp xếp
            calculated_priority = priority_map.get(op_name, 99)
            action.order = calculated_priority
            valid_actions.append(action)

    # Sắp xếp danh sách hành động theo thứ tự an toàn
    valid_actions.sort(key=lambda x: x.order)
    plan.actions = valid_actions
    return plan
