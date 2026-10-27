"""
Test đánh giá kết quả và xuất JSON phiên (S6): upsert đánh giá, kiểm quyền, nội dung file xuất
(đủ dữ liệu để phân tích, không lộ thread_id / đường dẫn đĩa / lỗi nội bộ). Phiên được tạo bằng
pipeline rule-based thật trên ảnh nhỏ (offline).
"""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import workspace
from src.api.db import Database
from src.api.main import app
from tests.unit.test_api_flow import _answer, _create, _login


@pytest.fixture
def client(app_db: Database) -> TestClient:
    test_client = TestClient(app)
    _login(test_client, "alice")
    return test_client


@pytest.fixture
def done_session(client: TestClient) -> dict:
    created = _create(client, variants=1)
    body = _answer(client, created["id"], style="natural").json()
    assert body["status"] == "done"
    return body


def _rate(client: TestClient, sid: str, **body: Any):
    return client.put(f"/api/v1/workspace/sessions/{sid}/ratings", json=body)


def test_rating_is_upserted_per_result(client: TestClient, done_session: dict) -> None:
    sid = done_session["id"]
    run_id = done_session["process"]["run_id"]
    treatment_id = done_session["active_treatment_id"]

    first = _rate(client, sid, run_id=run_id, treatment_id=treatment_id, rating="good", note=" ok ")
    assert first.status_code == 200, first.text
    assert first.json()["note"] == "ok"
    second = _rate(client, sid, run_id=run_id, treatment_id=treatment_id, rating="bad", note="tối")
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["rating"] == "bad"

    # Đánh giá cấp run (không gắn phiên bản) là một bản ghi riêng, cũng được cập nhật tại chỗ
    run_level = _rate(client, sid, run_id=run_id, rating="good")
    again = _rate(client, sid, run_id=run_id, rating="bad")
    assert run_level.json()["id"] == again.json()["id"] != first.json()["id"]

    ratings = client.get(f"/api/v1/workspace/sessions/{sid}").json()["ratings"]
    assert len(ratings) == 2
    assert {r["rating"] for r in ratings} == {"bad"}


def test_rating_validation_and_ownership(client: TestClient, done_session: dict) -> None:
    sid = done_session["id"]
    run_id = done_session["process"]["run_id"]
    assert _rate(client, sid, run_id=run_id, rating="excellent").status_code == 422
    assert _rate(client, sid, run_id=run_id, rating="good", note="x" * 2001).status_code == 422
    assert _rate(client, sid, run_id="0" * 32, rating="good").status_code == 404
    assert (
        _rate(client, sid, run_id=run_id, treatment_id="0" * 32, rating="good").status_code == 404
    )
    bob = TestClient(app)
    _login(bob, "bob")
    assert _rate(bob, sid, run_id=run_id, rating="good").status_code == 404


def test_export_contains_analysis_data_without_internals(
    client: TestClient, done_session: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = done_session["id"]
    run_id = done_session["process"]["run_id"]
    _rate(
        client,
        sid,
        run_id=run_id,
        treatment_id=done_session["active_treatment_id"],
        rating="good",
        note="ổn",
    )
    # Thêm một run lỗi để chắc lỗi nội bộ thô không xuất ra
    monkeypatch.setattr(
        workspace, "refine", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret boom"))
    )
    client.post(f"/api/v1/workspace/sessions/{sid}/refine", json={"feedback": "sáng hơn"})

    response = client.get(f"/api/v1/workspace/sessions/{sid}/export")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == (f'attachment; filename="session-{sid}.json"')
    data = response.json()
    assert data["schema"] == "iid.session-export/1"
    assert data["app"]["gemini_key_configured"] is False
    assert data["session"]["id"] == sid
    assert [r["rating"] for r in data["ratings"]] == ["good"]
    process = next(r for r in data["runs"] if r["kind"] == "process")
    assert process["result"]["sources"]["plan"] == ["rule_based"]
    assert process["result"]["history"][0]["plan"]["actions"] is not None
    assert {"analyze", "process", "refine"} <= {r["kind"] for r in data["runs"]}
    assert [m["kind"] for m in data["messages"]][:4] == [
        "diagnosis",
        "questions",
        "answers",
        "result",
    ]
    assert data["treatments"] and data["artifacts"]
    assert data["variants"] == []  # một kết quả mỗi lần chạy (không còn chọn phong cách)

    text = json.dumps(data)
    assert "thread_id" not in text
    assert "rel_path" not in text and "users/" not in text and "original.png" not in text
    assert "secret boom" not in text
    assert all("error" not in run for run in data["runs"])


def test_export_requires_owner(client: TestClient, done_session: dict) -> None:
    bob = TestClient(app)
    _login(bob, "bob")
    assert bob.get(f"/api/v1/workspace/sessions/{done_session['id']}/export").status_code == 404
