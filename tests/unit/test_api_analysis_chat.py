"""
Test chat bổ sung phân tích (T5): POST analysis-notes, DELETE analysis-edits, truyền edit vào
/answers, overlay vùng, sai trạng thái, giới hạn 20 edit và cột draft_edits cho CSDL cũ.
Pipeline rule-based thật trên ảnh nhỏ (không API key, không mạng).
"""

import sqlite3
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import workspace
from src.api.db import Database, close_db, init_db
from src.api.main import app
from src.api.workspace_schemas import MAX_DRAFT_EDITS

PASSWORD = "correct horse"
BASE = "/api/v1/workspace/sessions"


def _png() -> bytes:
    rng = np.random.default_rng(0)
    image = rng.integers(10, 50, (48, 64, 3)).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(app_db: Database) -> TestClient:
    test_client = TestClient(app)
    response = test_client.post(
        "/api/v1/auth/register", json={"username": "alice", "password": PASSWORD}
    )
    assert response.status_code == 201
    return test_client


@pytest.fixture
def session(client: TestClient) -> Dict[str, Any]:
    response = client.post(
        BASE,
        files={"file": ("dark.png", _png(), "image/png")},
        data={"max_iterations": "1", "num_variants": "1"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "awaiting_answers"
    return body


def _note(client: TestClient, sid: str, message: str):
    return client.post(f"{BASE}/{sid}/analysis-notes", json={"message": message})


def _defects(diagnosis: Dict[str, Any]) -> List[str]:
    return [f"{d['type']}@{d['region']}" for d in diagnosis["defects"]]


# ---------------- analysis-notes ----------------
def test_note_adds_edit_and_updates_draft(client: TestClient, session: Dict[str, Any]) -> None:
    response = _note(client, session["id"], "mặt còn tối")
    assert response.status_code == 200, response.text
    body = response.json()

    [edit] = body["draft_edits"]
    assert (edit["op"], edit["type"], edit["region"]) == ("add_defect", "underexposed", "face")
    assert "underexposed@face" in _defects(body["draft_diagnosis"])
    added = [d for d in body["draft_diagnosis"]["defects"] if d["region"] == "face"]
    assert added[0]["origin"] == "user"
    # Chẩn đoán gốc không bị sửa
    assert "underexposed@face" not in _defects(body["diagnosis"])

    kinds = [m["kind"] for m in body["messages"]]
    assert kinds[-2:] == ["analysis_note", "analysis_update"]
    update = body["messages"][-1]["content"]
    assert update["source"] == "rule" and update["unparsed"] is False
    assert update["edits"] == [edit]


def test_unparsed_message_is_kept_and_flagged(client: TestClient, session: Dict[str, Any]) -> None:
    response = _note(client, session["id"], "bạn ơi giúp mình với")
    assert response.status_code == 200
    body = response.json()
    assert body["draft_edits"] == []
    update = body["messages"][-1]["content"]
    assert update["unparsed"] is True and update["edits"] == []


def test_note_requires_awaiting_answers(client: TestClient, session: Dict[str, Any]) -> None:
    sid = session["id"]
    done = client.post(f"{BASE}/{sid}/answers", json={"answers": {}, "notes": ""})
    assert done.status_code == 200 and done.json()["status"] == "done"
    response = _note(client, sid, "mặt còn tối")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_state"


def test_note_validation_and_ownership(
    client: TestClient, session: Dict[str, Any], app_db: Database
) -> None:
    sid = session["id"]
    assert _note(client, sid, "   ").status_code == 422
    assert (
        client.post(f"{BASE}/{sid}/analysis-notes", json={"message": "x" * 1001}).status_code == 422
    )

    other = TestClient(app)
    other.post("/api/v1/auth/register", json={"username": "bob", "password": PASSWORD})
    assert _note(other, sid, "mặt còn tối").status_code == 404
    assert other.delete(f"{BASE}/{sid}/analysis-edits/0").status_code == 404


def test_edit_limit_is_enforced(client: TestClient, session: Dict[str, Any]) -> None:
    sid = session["id"]
    for _ in range(MAX_DRAFT_EDITS):
        assert _note(client, sid, "nền nhiễu").status_code == 200
    response = _note(client, sid, "nền nhiễu")
    assert response.status_code == 422
    assert client.get(f"{BASE}/{sid}").json()["draft_edits"].__len__() == MAX_DRAFT_EDITS


# ---------------- analysis-edits ----------------
def test_delete_edit_removes_it_from_draft(client: TestClient, session: Dict[str, Any]) -> None:
    sid = session["id"]
    _note(client, sid, "mặt còn tối")
    _note(client, sid, "giữ tông ấm")
    response = client.delete(f"{BASE}/{sid}/analysis-edits/0")
    assert response.status_code == 200
    body = response.json()
    assert [e["op"] for e in body["draft_edits"]] == ["keep"]
    assert "underexposed@face" not in _defects(body["draft_diagnosis"])

    assert client.delete(f"{BASE}/{sid}/analysis-edits/5").status_code == 404
    assert client.delete(f"{BASE}/{sid}/analysis-edits/-1").status_code == 404


# ---------------- /answers ----------------
def test_answers_pass_edits_to_pipeline(client: TestClient, session: Dict[str, Any]) -> None:
    sid = session["id"]
    _note(client, sid, "mặt còn tối")
    _note(client, sid, "bạn ơi giúp mình với")
    response = client.post(f"{BASE}/{sid}/answers", json={"answers": {}, "notes": "ghi chú"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "done"

    first = body["process"]["history"][0]["diagnosis"]
    assert "underexposed@face" in _defects(first)
    user_defect = [d for d in first["defects"] if d["region"] == "face"][0]
    assert user_defect["origin"] == "user"

    answers = [m for m in body["messages"] if m["kind"] == "answers"][0]["content"]
    assert answers["edits"][0]["type"] == "underexposed"
    assert "ghi chú" in answers["notes"] and "giúp mình" in answers["notes"]


def test_answers_forward_edits_to_answer_session(
    client: TestClient, session: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: Dict[str, Any] = {}

    def fake(thread_id: str, answers: Dict[str, str], notes: str, edits: Any = None) -> Any:
        seen.update(answers=answers, notes=notes, edits=edits)
        raise RuntimeError("stop here")

    monkeypatch.setattr(workspace, "answer_session", fake)
    sid = session["id"]
    _note(client, sid, "mặt còn tối")
    assert (
        client.post(f"{BASE}/{sid}/answers", json={"answers": {}, "notes": ""}).status_code == 500
    )
    assert [e["type"] for e in seen["edits"]] == ["underexposed"]


# ---------------- overlay vùng ----------------
def test_session_has_region_overlay(client: TestClient, session: Dict[str, Any]) -> None:
    assert session["region_overlay_url"]
    image = client.get(session["region_overlay_url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"


def test_overlay_is_rebuilt_when_edit_adds_region(
    client: TestClient, session: Dict[str, Any]
) -> None:
    sid = session["id"]
    before = {a["id"] for a in session["artifacts"] if a["kind"] == "region_overlay"}
    body = _note(client, sid, "bầu trời bị cháy").json()
    overlay = [a for a in body["artifacts"] if a["kind"] == "region_overlay"]
    assert len(overlay) == 1 and {a["id"] for a in overlay} == before  # cùng artifact, đổi nội dung
    assert client.get(body["region_overlay_url"]).status_code == 200
    assert isinstance(body["regions"], list)


def test_overlay_failure_does_not_block_session(
    app_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.api.workspace_service as service

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("overlay exploded")

    monkeypatch.setattr(service, "diagnosis_region_overlay", boom)
    test_client = TestClient(app)
    test_client.post("/api/v1/auth/register", json={"username": "carol", "password": PASSWORD})
    response = test_client.post(
        BASE,
        files={"file": ("dark.png", _png(), "image/png")},
        data={"max_iterations": "1", "num_variants": "1"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_answers" and body["region_overlay_url"] is None


# ---------------- clarify lại ----------------
def test_clarify_again_clears_edits(client: TestClient, session: Dict[str, Any]) -> None:
    sid = session["id"]
    _note(client, sid, "mặt còn tối")
    _note(client, sid, "bạn ơi giúp mình với")
    response = client.post(f"{BASE}/{sid}/clarify")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["draft_edits"] == []
    assert "underexposed@face" not in _defects(body["draft_diagnosis"])
    # Tin nhắn không hiểu của lần phân tích trước không còn nối vào notes
    answered = client.post(f"{BASE}/{sid}/answers", json={"answers": {}, "notes": ""}).json()
    answers = [m for m in answered["messages"] if m["kind"] == "answers"][0]["content"]
    assert "giúp mình" not in answers["notes"]


# ---------------- CSDL cũ ----------------
def test_old_database_gets_draft_edits_column(app_db: Database) -> None:
    settings = app_db.settings
    close_db()
    connection = sqlite3.connect(settings.db_path)
    connection.execute("ALTER TABLE processing_sessions DROP COLUMN draft_edits")
    connection.commit()
    connection.close()

    reopened = init_db(settings)
    with reopened.engine.connect() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(processing_sessions)")}
    assert "draft_edits" in columns
