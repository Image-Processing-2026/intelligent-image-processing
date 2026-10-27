"""
Test thông tin hệ thống và bộ nhớ ca bệnh theo người dùng (S7): chế độ chạy theo cấu hình
không lộ khóa API, bộ nhớ chỉ trả ca của người dùng hiện tại. Offline, dữ liệu trong tmp_path.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.agent import memory as case_memory
from src.agent.memory import CaseMemory, CaseRecord
from src.api.db import Database
from src.api.main import app
from src.api.models import User
from tests.unit.test_api_flow import _login


def _user_id(app_db: Database, username: str) -> str:
    with app_db.sessionmaker() as db:
        return db.scalar(select(User.id).where(User.username == username))


def test_system_requires_login_and_reports_rule_based_mode(app_db: Database) -> None:
    client = TestClient(app)
    assert client.get("/api/v1/workspace/system").status_code == 401
    _login(client, "alice")
    body = client.get("/api/v1/workspace/system").json()
    assert body["backend"] == "ok"
    assert body["gemini_key_configured"] is False
    assert body["memory_enabled"] is False  # conftest tắt bộ nhớ
    assert body["pipeline_concurrency"] == 1
    assert body["max_upload_mb"] == 20


def test_system_never_leaks_the_api_key(app_db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-secret-key-123")
    client = TestClient(app)
    _login(client, "alice")
    response = client.get("/api/v1/workspace/system")
    assert response.json()["gemini_key_configured"] is True
    assert "fake-secret-key-123" not in response.text


def test_memory_is_scoped_to_the_signed_in_user(
    app_db: Database, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = CaseMemory(tmp_path / "cases.sqlite")
    monkeypatch.setattr(case_memory, "_MEMORY", store)
    alice = TestClient(app)
    _login(alice, "alice")
    bob = TestClient(app)
    _login(bob, "bob")
    alice_id = _user_id(app_db, "alice")
    bob_id = _user_id(app_db, "bob")
    for index, owner in enumerate([alice_id, alice_id, bob_id]):
        store._save(
            CaseRecord(
                id=f"case{index}",
                user_id=owner,
                created_at=f"2026-10-22T10:0{index}:00+00:00",
                feedback=["cooler@face:1"],
            )
        )

    mine = alice.get("/api/v1/workspace/memory").json()
    assert mine == {
        "enabled": True,
        "cases": 2,
        "feedback": {"cooler": 2},
    }
    theirs = bob.get("/api/v1/workspace/memory").json()
    assert theirs["cases"] == 1
    # Không có cách xem bộ nhớ người khác qua tham số client
    spoofed = alice.get("/api/v1/workspace/memory", params={"user_id": bob_id}).json()
    assert spoofed["cases"] == 2


def test_memory_disabled(app_db: Database) -> None:
    client = TestClient(app)
    _login(client, "alice")
    assert client.get("/api/v1/workspace/memory").json()["enabled"] is False
