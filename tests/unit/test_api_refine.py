"""
Test tinh chỉnh theo góp ý và lịch sử phiên bản phác đồ (S5): refine tạo phiên bản mới với
điều chỉnh, ghi góp ý vào bộ nhớ ca bệnh, quay về bản trước, quyền sở hữu, lỗi refine. Phiên
được tạo bằng pipeline rule-based thật trên ảnh nhỏ (offline, góp ý được hiểu bằng luật).
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.agent import memory as case_memory
from src.agent.memory import CaseMemory
from src.api import workspace
from src.api.db import Database, close_db, init_db
from src.api.main import app
from src.api.models import Run
from tests.unit.test_api_flow import _answer, _create, _login


@pytest.fixture
def client(app_db: Database) -> TestClient:
    test_client = TestClient(app)
    _login(test_client, "alice")
    return test_client


@pytest.fixture
def done_session(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    monkeypatch.setattr(case_memory, "_MEMORY", CaseMemory(tmp_path / "cases.sqlite"))
    created = _create(client)
    body = _answer(client, created["id"], style="natural").json()
    assert body["status"] == "done"
    return body


def _refine(client: TestClient, sid: str, feedback: str, **extra: Any):
    return client.post(
        f"/api/v1/workspace/sessions/{sid}/refine", json={"feedback": feedback, **extra}
    )


def test_refine_creates_new_version_with_adjustments(
    client: TestClient, done_session: dict
) -> None:
    sid = done_session["id"]
    [v1] = done_session["treatments"]
    response = _refine(client, sid, "da hơi vàng")
    assert response.status_code == 200, response.text
    body = response.json()

    treatments = {t["version"]: t for t in body["treatments"]}
    v2 = treatments[2]
    assert v2["source"] == "refine" and v2["parent_id"] == v1["id"]
    assert v2["feedback_text"] == "da hơi vàng"
    assert v2["refine_source"] == "rules"
    assert any(a["kind"] == "cooler" for a in v2["adjustments"])
    assert v2["notes"]
    assert body["active_treatment_id"] == v2["id"]

    artifacts = {a["id"]: a for a in body["artifacts"]}
    assert (artifacts[v2["artifact_id"]]["width"], artifacts[v2["artifact_id"]]["height"]) == (
        64,
        48,
    )
    kinds = [m["kind"] for m in body["messages"]]
    assert kinds[-2:] == ["feedback", "refine_result"]
    assert body["messages"][-2]["content"]["text"] == "da hơi vàng"
    assert body["messages"][-1]["content"]["version"] == 2

    case = case_memory.get_case_memory().get(done_session["process"]["case_id"])
    assert case.feedback_texts == ["da hơi vàng"]


def test_refine_chains_on_active_version_and_activate_goes_back(
    client: TestClient, done_session: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = done_session["id"]
    [v1] = done_session["treatments"]
    v2 = _refine(client, sid, "sáng hơn").json()
    v3 = _refine(client, sid, "thêm tương phản").json()
    by_version = {t["version"]: t for t in v3["treatments"]}
    assert by_version[3]["parent_id"] == by_version[2]["id"]
    assert v2["active_treatment_id"] == by_version[2]["id"]

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("activate must not run the pipeline")

    monkeypatch.setattr(workspace, "refine", forbidden)
    back = client.post(f"/api/v1/workspace/sessions/{sid}/treatments/{v1['id']}/activate")
    assert back.status_code == 200
    assert back.json()["active_treatment_id"] == v1["id"]

    # Góp ý tiếp áp vào bản đang dùng (v1), không phải bản mới nhất
    monkeypatch.undo()
    from_v1 = _refine(client, sid, "bớt nhiễu").json()
    newest = max(from_v1["treatments"], key=lambda t: t["version"])
    assert newest["version"] == 4 and newest["parent_id"] == v1["id"]


def test_refine_explicit_base_treatment(client: TestClient, done_session: dict) -> None:
    sid = done_session["id"]
    [v1] = done_session["treatments"]
    _refine(client, sid, "sáng hơn")
    body = _refine(client, sid, "ấm hơn", treatment_id=v1["id"]).json()
    newest = max(body["treatments"], key=lambda t: t["version"])
    assert newest["parent_id"] == v1["id"]


def test_unknown_feedback_keeps_actions(client: TestClient, done_session: dict) -> None:
    sid = done_session["id"]
    [v1] = done_session["treatments"]
    body = _refine(client, sid, "xyz không rõ nghĩa").json()
    newest = max(body["treatments"], key=lambda t: t["version"])
    assert newest["refine_source"] == "none"
    assert newest["adjustments"] == []
    assert [a["operation"] for a in newest["actions"]] == [a["operation"] for a in v1["actions"]]


def test_refine_validation_and_state(
    client: TestClient, app_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _create(client)
    sid = created["id"]
    assert _refine(client, sid, "da hơi vàng").status_code == 409  # chưa có kết quả
    _answer(client, sid, style="natural")
    assert _refine(client, sid, "   ").status_code == 422
    assert _refine(client, sid, "x" * 2001).status_code == 422
    missing = _refine(client, sid, "sáng hơn", treatment_id="0" * 32)
    assert missing.status_code == 404
    activate_missing = client.post(
        f"/api/v1/workspace/sessions/{sid}/treatments/{'0' * 32}/activate"
    )
    assert activate_missing.status_code == 404


def test_other_user_cannot_refine_or_activate(client: TestClient, done_session: dict) -> None:
    sid = done_session["id"]
    [v1] = done_session["treatments"]
    bob = TestClient(app)
    _login(bob, "bob")
    assert _refine(bob, sid, "sáng hơn").status_code == 404
    assert (
        bob.post(f"/api/v1/workspace/sessions/{sid}/treatments/{v1['id']}/activate").status_code
        == 404
    )


def test_refine_failure_keeps_active_version(
    client: TestClient, done_session: dict, app_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = done_session["id"]

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("secret refine failure")

    monkeypatch.setattr(workspace, "refine", broken)
    response = _refine(client, sid, "sáng hơn")
    assert response.status_code == 500 and "secret" not in response.text
    detail = client.get(f"/api/v1/workspace/sessions/{sid}").json()
    assert detail["status"] == "done"
    assert detail["active_treatment_id"] == done_session["active_treatment_id"]
    assert len(detail["treatments"]) == 1
    with app_db.sessionmaker() as db:
        assert [r.kind for r in db.scalars(select(Run).where(Run.status == "failed"))] == ["refine"]


def test_history_survives_reopen(client: TestClient, done_session: dict, app_db: Database) -> None:
    sid = done_session["id"]
    refined = _refine(client, sid, "da hơi vàng").json()
    init_db(app_db.settings)  # engine mới trên cùng thư mục dữ liệu
    try:
        reopened = TestClient(app)
        reopened.cookies.update(client.cookies)
        again = reopened.get(f"/api/v1/workspace/sessions/{sid}").json()
        assert again["treatments"] == refined["treatments"]
        assert again["messages"] == refined["messages"]
    finally:
        close_db()
