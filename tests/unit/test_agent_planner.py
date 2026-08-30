"""
Unit tests for Module 4: Agent & Plan Validator.
"""

from src.agent.planner import validate_and_sort_plan
from src.agent.state import RegionOperation, TreatmentPlan


def test_validate_and_sort_plan():
    """Kiểm tra việc lọc công cụ ngoài danh mục và sắp xếp độ ưu tiên."""
    raw_plan = TreatmentPlan(
        iteration=1,
        reasoning="Test plan",
        actions=[
            RegionOperation(
                region_id="1",
                target_prompt="full",
                detected_issue="blur",
                operation="sharpen",
                order=1,
            ),
            RegionOperation(
                region_id="2",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                order=2,
            ),
            RegionOperation(
                region_id="3",
                target_prompt="full",
                detected_issue="invalid",
                operation="deep_fake_beautify",  # Invalid operation
                order=3,
            ),
        ],
    )

    validated = validate_and_sort_plan(raw_plan)

    # Thao tác ngoài danh mục phải bị loại bỏ
    assert len(validated.actions) == 2
    # Thao tác khử nhiễu phải được sắp xếp trước làm nét (denoise before sharpen)
    assert validated.actions[0].operation == "denoise"
    assert validated.actions[1].operation == "sharpen"
