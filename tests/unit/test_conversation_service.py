"""Kiểm tra phân biệt hỏi đáp, chỉnh phương án và chọn ảnh mà không render."""

import numpy as np
import pytest

from src.agent.state import DiagnosisReport
from src.api import conversation_service as service


def test_question_keeps_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Một câu hỏi giải thích giữ nguyên danh sách thao tác."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-key")
    monkeypatch.setattr(
        service,
        "call_gemini_json",
        lambda *args: {
            "reply": "Tăng sáng để thấy chủ thể rõ hơn.",
            "changes_plan": False,
        },
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    actions = [
        {
            "region_id": "full",
            "target_prompt": "full",
            "detected_issue": "dark",
            "operation": "gamma_correct",
            "parameters": {"gamma": 1.2},
        }
    ]
    previous = {"revision": 1, "status": "ready", "actions": actions, "requirements": ["giữ nền"]}
    proposal, _, reply = service.update_proposal(
        previous, "vì sao tăng sáng?", DiagnosisReport(), image, None, [], []
    )
    assert proposal["actions"][0]["parameters"]["gamma"] == 1.2
    assert proposal["requirements"] == ["giữ nền"]
    assert "chủ thể" in reply


@pytest.mark.parametrize("mode", ["rebuild", "incremental"])
def test_model_selects_source_without_processing(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Cả dựng lại và chỉnh tiếp đều chỉ sinh kế hoạch chờ nút bấm."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-key")
    monkeypatch.setattr(
        service,
        "call_gemini_json",
        lambda *args: {
            "reply": "Sẽ điều chỉnh thêm độ sáng.",
            "changes_plan": True,
            "source_mode": mode,
            "source_reason": "Theo yêu cầu hiện tại",
            "actions": [
                {
                    "region_id": "full",
                    "target_prompt": "full",
                    "operation": "gamma_correct",
                    "parameters": {"gamma": 1.1},
                }
            ],
        },
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    before = image.copy()
    proposal, _, _ = service.update_proposal(
        {"revision": 1, "actions": [], "status": "applied"},
        "sáng hơn",
        DiagnosisReport(),
        image,
        image,
        [],
        [],
    )
    assert proposal["source_mode"] == mode
    assert proposal["status"] == "ready"
    np.testing.assert_array_equal(image, before)
