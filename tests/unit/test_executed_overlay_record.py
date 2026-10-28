"""Overlay mask đã thực thi được ghi cùng run process; thiếu mask trong cache thì bỏ qua."""

import numpy as np

from src.agent.state import ExecutedAction, HistoryItem
from src.api.workspace_service import _executed_overlay_artifact


class _Entry:
    def __init__(self, executed: list) -> None:
        self.executed = executed


def test_no_executed_actions_returns_none() -> None:
    state = {"original_image": np.zeros((8, 8, 3), np.uint8)}
    assert _executed_overlay_artifact(lambda *a: 1, state, [_Entry([])]) is None  # type: ignore[arg-type,return-value]


def test_mask_evicted_from_cache_is_skipped_without_error() -> None:
    action = ExecutedAction(region_id="face", operation="gamma_correct", mask_hash="deadbeef")
    state = {"original_image": np.zeros((8, 8, 3), np.uint8)}
    calls: list = []
    result = _executed_overlay_artifact(
        lambda *a: calls.append(a),
        state,
        [_Entry([action])],  # type: ignore[arg-type,return-value]
    )
    assert result is None and calls == []


def test_history_item_default_has_no_executed() -> None:
    assert HistoryItem.model_fields["executed"].default_factory() == []  # type: ignore[misc]
