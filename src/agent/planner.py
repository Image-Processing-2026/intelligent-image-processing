"""
Kiểm tra tính hợp lệ của kế hoạch điều trị (Plan Validator).
Đảm bảo Agent chỉ sử dụng các công cụ trong Toolbox được phép và đúng thứ tự ưu tiên.
"""

from typing import List, Set

from .state import RegionOperation, TreatmentPlan

ALLOWED_OPERATIONS: Set[str] = {
    "denoise",
    "gamma_correct",
    "clahe",
    "sharpen",
    "color_correct",
}


def validate_and_sort_plan(plan: TreatmentPlan) -> TreatmentPlan:
    """
    Xác thực kế hoạch điều trị:
    1. Loại bỏ các thao tác không nằm trong Toolbox.
    2. Sắp xếp thứ tự ưu tiên (khử nhiễu -> cân bằng sáng -> CLAHE -> làm nét -> chỉnh màu).
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
            # Gán lại độ ưu tiên mặc định nếu chưa được sắp xếp
            calculated_priority = priority_map.get(op_name, 99)
            action.order = calculated_priority
            valid_actions.append(action)

    # Sắp xếp danh sách hành động theo thứ tự an toàn
    valid_actions.sort(key=lambda x: x.order)
    plan.actions = valid_actions
    return plan
