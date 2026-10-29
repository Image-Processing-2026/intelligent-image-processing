"""
Kiểm thử API Workspace (Tạo phiên với/không có user_prompt và clarify_again).
Chạy offline hoàn toàn với SQLite tạm thời và pipeline rule-based.
"""

from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import workspace
from src.api.db import Database
from src.api.main import app

PASSWORD = "correct horse"
BASE = "/api/v1/workspace/sessions"


def _png() -> bytes:
    """Tạo ảnh PNG 48x64 RGB cho kiểm thử."""
    rng = np.random.default_rng(0)
    image = rng.integers(10, 50, (48, 64, 3)).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(app_db: Database) -> TestClient:
    """Khởi tạo TestClient và đăng ký người dùng mặc định."""
    test_client = TestClient(app)
    response = test_client.post(
        "/api/v1/auth/register", json={"username": "alice", "password": PASSWORD}
    )
    assert response.status_code == 201
    return test_client


def _create(client: TestClient, prompt: str | None = None) -> Any:
    """Helper tạo phiên với prompt tùy chọn."""
    data: Dict[str, str] = {"max_iterations": "1"}
    if prompt is not None:
        data["user_prompt"] = prompt
    return client.post(
        BASE,
        files={"file": ("dark.png", _png(), "image/png")},
        data=data,
    )


def _defects(diagnosis: Dict[str, Any]) -> List[str]:
    """Trích xuất danh sách defect dạng type@region từ payload diagnosis."""
    return [f"{d['type']}@{d.get('region', 'full')}" for d in diagnosis.get("defects", [])]


def test_create_session_without_user_prompt(client: TestClient) -> None:
    """Tạo phiên không kèm user_prompt chỉ sinh diagnosis và questions, không có draft_edits."""
    response = _create(client, None)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "awaiting_answers"
    assert [m["kind"] for m in body["messages"]] == ["diagnosis", "questions"]
    assert body["draft_edits"] == []


def test_create_session_with_blank_user_prompt(client: TestClient) -> None:
    """Tạo phiên với user_prompt toàn khoảng trắng coi như không có prompt."""
    response = _create(client, "   ")
    assert response.status_code == 201, response.text
    body = response.json()
    assert [m["kind"] for m in body["messages"]] == ["diagnosis", "questions"]
    assert body["draft_edits"] == []


def test_create_session_with_user_prompt_parsed(client: TestClient) -> None:
    """
    Tạo phiên kèm user_prompt hợp lệ:
    - Tin nhắn đầu tiên là user_prompt (role=user).
    - Có message diagnosis, questions và sau đó là analysis_update (assistant).
    - draft_edits được parse nạp sẵn.
    """
    response = _create(client, "  mặt còn tối  ")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "awaiting_answers"

    kinds = [m["kind"] for m in body["messages"]]
    assert kinds == ["user_prompt", "diagnosis", "questions", "analysis_update"]

    first = body["messages"][0]
    assert first["role"] == "user"
    assert first["content"]["text"] == "mặt còn tối"
    assert first["content"]["original_artifact_id"] == body["original"]["id"]

    [edit] = body["draft_edits"]
    assert (edit["op"], edit["type"], edit["region"]) == ("add_defect", "underexposed", "face")
    assert "underexposed@face" in _defects(body["draft_diagnosis"])

    update = body["messages"][-1]["content"]
    assert update["from_prompt"] is True
    assert update["unparsed"] is False


def test_create_session_with_unparsed_user_prompt(client: TestClient) -> None:
    """user_prompt không nhận diện được edit sẽ gắn cờ unparsed và nối vào notes khi trả lời."""
    body = _create(client, "bạn ơi giúp mình với").json()
    assert body["draft_edits"] == []
    assert body["messages"][-1]["content"]["unparsed"] is True

    answered = client.post(f"{BASE}/{body['id']}/answers", json={"answers": {}, "notes": ""})
    assert answered.status_code == 200, answered.text
    answers = [m for m in answered.json()["messages"] if m["kind"] == "answers"][0]["content"]
    assert "giúp mình" in answers["notes"]


def test_create_session_user_prompt_too_long(client: TestClient) -> None:
    """user_prompt vượt quá giới hạn 1000 ký tự phải trả mã lỗi 422 validation."""
    response = _create(client, "x" * 1001)
    assert response.status_code == 422
    assert "user_prompt" in response.json()["error"]["fields"]


def test_clarify_again_reparses_stored_user_prompt(client: TestClient) -> None:
    """Hỏi ý lại (clarify) sẽ làm mới chẩn đoán và tự động parse lại user_prompt ban đầu."""
    sid = _create(client, "mặt còn tối").json()["id"]
    client.post(f"{BASE}/{sid}/analysis-notes", json={"message": "giữ tông ấm"})

    body = client.post(f"{BASE}/{sid}/clarify").json()
    # Edit thêm sau bị xóa, chỉ còn edit parse từ prompt ban đầu
    assert [(e["op"], e.get("type")) for e in body["draft_edits"]] == [
        ("add_defect", "underexposed")
    ]
    assert [m["kind"] for m in body["messages"]].count("user_prompt") == 1
    assert body["messages"][-1]["kind"] == "analysis_update"


def test_user_prompt_survives_failed_diagnosis(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nếu chẩn đoán lần đầu thất bại, user_prompt vẫn được lưu và phục hồi khi clarify."""
    real_start = workspace.start_session

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("perceive exploded")

    monkeypatch.setattr(workspace, "start_session", boom)
    body = _create(client, "mặt còn tối").json()
    assert body["status"] == "failed"
    assert [m["kind"] for m in body["messages"]][0] == "user_prompt"

    monkeypatch.setattr(workspace, "start_session", real_start)
    retried = client.post(f"{BASE}/{body['id']}/clarify")
    assert retried.status_code == 200, retried.text
    assert [e["type"] for e in retried.json()["draft_edits"]] == ["underexposed"]


def _force_running(app_db: Database, session_id: str, updated_at: str) -> None:
    """Đặt phiên về 'running' với mốc cập nhật cho trước, như phiên bị bỏ rơi."""
    from src.api.models import ProcessingSession

    with app_db.sessionmaker() as db:
        item = db.get(ProcessingSession, session_id)
        item.status = "running"
        item.updated_at = updated_at
        db.commit()


def test_orphaned_running_session_is_failed_and_can_clarify_again(
    client: TestClient, app_db: Database
) -> None:
    """Phiên 'running' không còn luồng nào xử lý (quá hạn) không được chạy mãi."""
    session_id = _create(client, None).json()["id"]
    _force_running(app_db, session_id, "2020-01-01T00:00:00+00:00")

    listed = client.get(BASE).json()
    assert [s["status"] for s in listed if s["id"] == session_id] == ["failed"]
    body = client.get(f"{BASE}/{session_id}").json()
    assert body["status"] == "failed"
    assert body["messages"][-1]["content"]["code"] == "interrupted"
    assert client.post(f"{BASE}/{session_id}/clarify").status_code == 200


def test_recent_or_active_running_session_is_left_alone(
    client: TestClient, app_db: Database
) -> None:
    """Phiên vừa chuyển 'running' (còn trong thời gian chờ) hoặc đang có luồng xử lý thì giữ nguyên."""
    from src.api.models import utc_now
    from src.api.workspace_service import active_session

    session_id = _create(client, None).json()["id"]
    _force_running(app_db, session_id, utc_now())
    assert client.get(f"{BASE}/{session_id}").json()["status"] == "running"

    _force_running(app_db, session_id, "2020-01-01T00:00:00+00:00")
    with active_session(session_id):
        assert client.get(f"{BASE}/{session_id}").json()["status"] == "running"
    assert client.get(f"{BASE}/{session_id}").json()["status"] == "failed"
