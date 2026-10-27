"""
Test luồng hỏi ý → chạy pipeline (S3): trả lời câu hỏi, nhánh ảnh chuẩn (P2), thread hết hạn và
hỏi ý lại (P7), lỗi pipeline, khóa phiên đang chạy, đọc lại kết quả từ CSDL. Các luồng chính chạy
pipeline rule-based thật trên ảnh nhỏ (không API key, không mạng).
"""

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from src.agent import memory as case_memory
from src.agent.memory import CaseMemory
from src.agent.session import SessionNotFoundError, answer_session, end_session
from src.api import workspace
from src.api.db import Database, close_db, init_db
from src.api.main import app
from src.api.models import ProcessingSession, Run, User

PASSWORD = "correct horse"


def _png(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _dark_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(10, 50, (48, 64, 3)).astype(np.uint8)


def _login(client: TestClient, username: str) -> None:
    client.cookies.clear()
    response = client.post(
        "/api/v1/auth/register", json={"username": username, "password": PASSWORD}
    )
    assert response.status_code == 201


def _create(client: TestClient, ground_truth: Any = None) -> dict:
    files = {"file": ("dark.png", _png(_dark_image()), "image/png")}
    if ground_truth is not None:
        files["ground_truth"] = ("gt.png", _png(ground_truth), "image/png")
    response = client.post(
        "/api/v1/workspace/sessions",
        files=files,
        data={"max_iterations": "1"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "awaiting_answers"
    return body


def _answer(client: TestClient, sid: str, **answers: str):
    return client.post(
        f"/api/v1/workspace/sessions/{sid}/answers",
        json={"answers": answers, "notes": "giữ tự nhiên"},
    )


def _thread_id(app_db: Database, sid: str) -> str:
    with app_db.sessionmaker() as db:
        return db.get(ProcessingSession, sid).thread_id


@pytest.fixture
def client(app_db: Database) -> TestClient:
    test_client = TestClient(app)
    _login(test_client, "alice")
    return test_client


def _forbid(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Bất kỳ lời gọi pipeline nào trong danh sách đều làm test thất bại."""

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("pipeline must not be called")

    for name in names:
        monkeypatch.setattr(workspace, name, boom)


# ---------------- luồng chính ----------------
def test_answer_runs_pipeline_and_persists_everything(
    client: TestClient,
    app_db: Database,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = CaseMemory(tmp_path / "cases.sqlite")
    monkeypatch.setattr(case_memory, "_MEMORY", store)
    created = _create(client)
    sid = created["id"]

    response = _answer(client, sid, style="natural")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "done"
    assert [m["kind"] for m in body["messages"]] == ["diagnosis", "questions", "answers", "result"]
    answers_message = body["messages"][2]["content"]
    assert answers_message["answers"]["style"] == "natural"
    assert answers_message["notes"] == "giữ tự nhiên"

    process = body["process"]
    assert process["final_decision"] in {"SHIP", "STOP_BEST_EFFORT", "RE_PROCESS"}
    assert process["total_iterations"] == 1
    assert set(process["sources"]["plan"]) <= {"rule_based"}
    assert process["is_synthetic"] is False
    assert len(process["history"]) == 1
    assert "base64" not in response.text

    artifacts = {a["id"]: a for a in body["artifacts"]}
    final = artifacts[process["artifacts"]["final"]]
    assert (final["width"], final["height"]) == (64, 48)
    assert client.get(final["url"]).headers["content-type"] == "image/png"
    for artifact_id in process["artifacts"]["intermediate"]:
        assert client.get(artifacts[artifact_id]["url"]).status_code == 200

    [treatment] = body["treatments"]
    assert treatment["source"] == "pipeline" and treatment["version"] == 1
    assert body["active_treatment_id"] == treatment["id"]
    assert treatment["artifact_id"] == final["id"]

    # Bộ nhớ ca bệnh tách theo người dùng (D8)
    with app_db.sessionmaker() as db:
        user_id = db.scalar(select(User.id).where(User.username == "alice"))
    assert process["case_id"] and [c.id for c in store.cases(user_id)] == [process["case_id"]]
    # Thread đã xong được dọn khỏi checkpointer (end_session)
    with pytest.raises(SessionNotFoundError):
        answer_session(_thread_id(app_db, sid), {}, "")

    # Mở lại từ CSDL (engine mới) không chạy lại pipeline
    _forbid(monkeypatch, "start_session", "answer_session", "run_pipeline")
    init_db(app_db.settings)
    try:
        reopened = TestClient(app)
        reopened.cookies.update(client.cookies)
        again = reopened.get(f"/api/v1/workspace/sessions/{sid}").json()
        assert again["messages"] == body["messages"]
        assert again["variants"] == body["variants"]
        assert again["process"] == body["process"]
        # Đã có kết quả: không trả lời lại, không hỏi ý lại (P4)
        assert _answer(reopened, sid, style="vivid").status_code == 409
        assert reopened.post(f"/api/v1/workspace/sessions/{sid}/clarify").status_code == 409
    finally:
        close_db()


def test_ground_truth_branch_uses_run_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _create(client, ground_truth=_dark_image() + 60)
    assert created["options"]["has_ground_truth"] is True
    _forbid(monkeypatch, "answer_session")

    body = _answer(client, created["id"], style="balanced").json()
    assert body["status"] == "done"
    process = body["process"]
    assert process["is_synthetic"] is True
    assert {"psnr", "ssim"} <= set(process["final_evaluation"])
    assert process["intent"]["style"] is None


# ---------------- thread hết hạn và hỏi ý lại ----------------
def test_expired_thread_then_clarify_again(client: TestClient, app_db: Database) -> None:
    sid = _create(client)["id"]
    end_session(_thread_id(app_db, sid))  # mô phỏng restart server

    expired = _answer(client, sid, style="natural")
    assert expired.status_code == 409
    assert expired.json()["error"]["code"] == "thread_expired"
    detail = client.get(f"/api/v1/workspace/sessions/{sid}").json()
    assert detail["status"] == "awaiting_answers"
    assert detail["messages"][-1]["content"]["code"] == "thread_expired"

    clarified = client.post(f"/api/v1/workspace/sessions/{sid}/clarify")
    assert clarified.status_code == 200
    kinds = [m["kind"] for m in clarified.json()["messages"]]
    assert kinds == ["diagnosis", "questions", "notice", "diagnosis", "questions"]

    done = _answer(client, sid, style="natural")
    assert done.status_code == 200 and done.json()["status"] == "done"


# ---------------- lỗi và khóa ----------------
def test_pipeline_failure_is_reported_without_details(
    client: TestClient, app_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = _create(client)["id"]

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("secret stack detail C:/data/app")

    monkeypatch.setattr(workspace, "answer_session", broken)
    response = _answer(client, sid, style="natural")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "pipeline_failed"
    assert "secret" not in response.text

    detail = client.get(f"/api/v1/workspace/sessions/{sid}").json()
    assert detail["status"] == "failed"  # không kẹt ở 'running'
    assert detail["error"]["code"] == "pipeline_failed"
    with app_db.sessionmaker() as db:
        failed = db.scalars(select(Run).where(Run.status == "failed")).all()
    assert [run.kind for run in failed] == ["process"]

    # Từ trạng thái lỗi có thể hỏi ý lại
    clarified = client.post(f"/api/v1/workspace/sessions/{sid}/clarify")
    assert clarified.status_code == 200 and clarified.json()["status"] == "awaiting_answers"


def test_running_session_rejects_second_request(
    client: TestClient, app_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = _create(client)["id"]
    with app_db.sessionmaker() as db:
        db.get(ProcessingSession, sid).status = "running"
        db.commit()
    _forbid(monkeypatch, "answer_session", "start_session")
    busy = _answer(client, sid, style="natural")
    assert busy.status_code == 409 and busy.json()["error"]["code"] == "session_busy"
    assert client.post(f"/api/v1/workspace/sessions/{sid}/clarify").status_code == 409


def test_other_user_cannot_answer_or_clarify(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = _create(client)["id"]
    _forbid(monkeypatch, "answer_session", "start_session", "run_pipeline")
    bob = TestClient(app)
    _login(bob, "bob")
    assert _answer(bob, sid, style="natural").status_code == 404
    assert bob.post(f"/api/v1/workspace/sessions/{sid}/clarify").status_code == 404


def test_answers_payload_is_bounded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = _create(client)["id"]
    _forbid(monkeypatch, "answer_session")
    too_long = client.post(
        f"/api/v1/workspace/sessions/{sid}/answers", json={"answers": {}, "notes": "x" * 1001}
    )
    assert too_long.status_code == 422
    detail = client.get(f"/api/v1/workspace/sessions/{sid}").json()
    assert detail["status"] == "awaiting_answers"
