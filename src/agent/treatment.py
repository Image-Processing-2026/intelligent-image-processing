"""
Phác đồ điều trị: tập thao tác agent thực sự giữ lại qua các vòng lặp.

Tách ra từ module phiên bản cũ (đã xóa) vì bộ nhớ ca bệnh, API và giao diện vẫn cần phác đồ
để lưu và render lại kết quả.
"""

from typing import List, Sequence

from .state import HistoryItem, RegionOperation


def treatment_recipe(history: Sequence[HistoryItem]) -> List[RegionOperation]:
    """Các thao tác đã thực sự giữ lại qua vòng lặp (bỏ vòng bị rollback), theo thứ tự."""
    recipe: List[RegionOperation] = []
    for item in history:
        if item.rolled_back or not item.plan:
            continue
        recipe += [action.model_copy(deep=True) for action in item.plan.actions]
    return recipe
