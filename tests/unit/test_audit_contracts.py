"""Hợp đồng dữ liệu của đợt sửa audit (bước 0): giá trị mặc định và plumbing rỗng."""

import numpy as np

from src.agent.graph import initial_doctor_state, run_pipeline
from src.agent.state import (
    ExecutedAction,
    HistoryItem,
    Outcome,
    RegionOperation,
    ReviewResult,
    SideEffectCheck,
    TreatmentPlan,
)


def test_contract_defaults() -> None:
    """Các kiểu mới có mặc định trung tính, hành vi cũ không đổi."""
    op = RegionOperation(
        region_id="full", target_prompt="full", detected_issue="noise", operation="denoise"
    )
    assert op.defect_ref is None
    assert SideEffectCheck().status == "not_run"
    assert ReviewResult().status == "not_run"
    assert ReviewResult().verdict is None
    assert ExecutedAction(region_id="full", operation="denoise").strength_scale == 1.0
    assert Outcome(status="NO_CHANGE").accepted_by is None

    item = HistoryItem(
        iteration=1,
        plan=TreatmentPlan(reasoning="x"),
        metrics_before={},
        metrics_after={},
        eval_score={},
        decision="SHIP",
    )
    assert item.executed == []
    assert item.side_effect_check.status == "not_run"
    assert item.review is None


def test_initial_state_has_audit_fields() -> None:
    state = initial_doctor_state(np.zeros((8, 8, 3), np.uint8))
    assert state["executed_actions"] == []
    assert state["outcome"] is None
    assert state["review"] is None


def test_pipeline_history_carries_audit_fields() -> None:
    """History của pipeline offline luôn serialize được kèm side_effect_check/executed."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 60, (96, 96, 3), dtype=np.uint8)
    result = run_pipeline(image, max_iterations=1)
    for item in result["history"]:
        dumped = item.model_dump()
        assert "executed" in dumped and "side_effect_check" in dumped
