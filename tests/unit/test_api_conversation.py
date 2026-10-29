"""Kiểm tra chat không render và thực thi phương án đúng phiên bản."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agent.session import SessionResult
from src.agent.state import DiagnosisReport
from src.api import workspace
from tests.unit.test_api_analysis_chat import BASE, client, session  # noqa: F401, F811


@pytest.fixture(autouse=True)
def offline_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tách kiểm thử orchestration khỏi model phân vùng và phép đo chậm."""
    monkeypatch.setattr(
        workspace,
        "start_session",
        lambda *args, **kwargs: SessionResult(
            session_id="offline-test",
            status="needs_input",
            diagnosis=DiagnosisReport(source="rule_based"),
        ),
    )


def test_conversation_then_execute(client: TestClient, session: dict[str, Any]) -> None:  # noqa: F811
    """Nhiều lượt chat không tạo treatment; chỉ click mới tạo kết quả."""
    url = f"{BASE}/{session['id']}"
    response = client.post(url + "/chat", json={"revision": 0})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["treatments"] == []
    proposal = body["conversation"]
    assert proposal["status"] == "ready"
    stale = client.post(url + "/execute", json={"revision": 42, "idempotency_key": "stale"})
    assert stale.status_code == 409
    payload = {"revision": proposal["revision"], "idempotency_key": "first"}
    result = client.post(url + "/execute", json=payload)
    assert result.status_code == 200, result.text
    assert len(result.json()["treatments"]) == 1
    retry = client.post(url + "/execute", json=payload)
    assert retry.status_code == 200
    assert len(retry.json()["treatments"]) == 1
    revision = result.json()["conversation"]["revision"]
    chat = client.post(url + "/chat", json={"revision": revision, "message": "ấm hơn một chút"})
    assert chat.status_code == 200, chat.text
    assert len(chat.json()["treatments"]) == 1
