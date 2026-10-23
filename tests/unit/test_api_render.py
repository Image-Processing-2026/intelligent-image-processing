"""
Test render phiên bản ở độ phân giải gốc (S4): phác đồ phiên bản → treatment mới, artifact cùng
kích thước ảnh gốc, ghi lựa chọn vào bộ nhớ ca bệnh, quyền sở hữu, lỗi render. Phiên được tạo
bằng pipeline rule-based thật trên ảnh nhỏ (offline).
"""

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from src.agent import memory as case_memory
from src.agent.memory import CaseMemory
from src.api import workspace
from src.api.db import Database
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
    created = _create(client, variants=2)
    body = _answer(client, created["id"], style="natural").json()
    assert body["status"] == "done"
    return body


def test_render_creates_variant_treatment_at_full_resolution(
    client: TestClient, done_session: dict
) -> None:
    sid = done_session["id"]
    [pipeline] = done_session["treatments"]
    variant = done_session["variants"][-1]

    response = client.post(f"/api/v1/workspace/sessions/{sid}/variants/{variant['id']}/render")
    assert response.status_code == 200, response.text
    body = response.json()

    treatments = {t["id"]: t for t in body["treatments"]}
    active = treatments[body["active_treatment_id"]]
    assert active["source"] == "variant" and active["version"] == 2
    assert active["parent_id"] == pipeline["id"]
    assert active["variant_id"] == variant["id"]
    assert [a["operation"] for a in active["actions"]] == [
        a["operation"] for a in variant["actions"]
    ]  # giữ nguyên thứ tự phác đồ

    artifacts = {a["id"]: a for a in body["artifacts"]}
    rendered = artifacts[active["artifact_id"]]
    assert (rendered["width"], rendered["height"]) == (64, 48)  # độ phân giải gốc
    image = Image.open(BytesIO(client.get(rendered["download_url"]).content))
    assert image.size == (64, 48)

    chosen = [v for v in body["variants"] if v["chosen"]]
    assert [v["id"] for v in chosen] == [variant["id"]]
    assert body["messages"][-1]["kind"] == "render_result"
    assert body["messages"][-1]["content"]["label"] == variant["label"]

    case = case_memory.get_case_memory().get(done_session["process"]["case_id"])
    assert case is not None and case.chosen_variant == variant["style_id"]


def test_render_twice_switches_chosen_variant(client: TestClient, done_session: dict) -> None:
    sid = done_session["id"]
    first, second = done_session["variants"][0], done_session["variants"][1]
    client.post(f"/api/v1/workspace/sessions/{sid}/variants/{first['id']}/render")
    body = client.post(f"/api/v1/workspace/sessions/{sid}/variants/{second['id']}/render").json()
    assert [v["id"] for v in body["variants"] if v["chosen"]] == [second["id"]]
    assert [t["version"] for t in body["treatments"]] == [1, 2, 3]


def test_render_requires_owner_and_done_session(
    client: TestClient, done_session: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = done_session["id"]
    variant_id = done_session["variants"][0]["id"]
    bob = TestClient(app)
    _login(bob, "bob")
    assert (
        bob.post(f"/api/v1/workspace/sessions/{sid}/variants/{variant_id}/render").status_code
        == 404
    )
    missing = client.post(f"/api/v1/workspace/sessions/{sid}/variants/{'0' * 32}/render")
    assert missing.status_code == 404


def test_render_failure_keeps_session_done(
    client: TestClient, done_session: dict, app_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = done_session["id"]

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("secret render failure")

    monkeypatch.setattr(workspace, "execute_plan", broken)
    response = client.post(
        f"/api/v1/workspace/sessions/{sid}/variants/{done_session['variants'][0]['id']}/render"
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "pipeline_failed"
    assert "secret" not in response.text
    detail = client.get(f"/api/v1/workspace/sessions/{sid}").json()
    assert detail["status"] == "done"
    assert detail["active_treatment_id"] == done_session["active_treatment_id"]
    with app_db.sessionmaker() as db:
        assert [r.kind for r in db.scalars(select(Run).where(Run.status == "failed"))] == ["render"]
