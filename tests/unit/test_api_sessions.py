"""
Test phiên xử lý (S2): tạo từ upload/ảnh mẫu, kiểm tra upload không tin cậy, quyền sở hữu,
đổi tên, xóa (kể cả file trên đĩa), artifact có kiểm quyền. start_session được giả lập trừ một
test chạy pipeline rule-based thật. Chạy offline, dữ liệu trong tmp_path.
"""

import dataclasses
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from src.agent.intent import Choice, Question
from src.agent.session import SessionResult
from src.agent.state import Defect, DiagnosisReport
from src.api import workspace
from src.api.db import Database, close_db, init_db
from src.api.main import app
from src.api.models import ProcessingSession, Run, User

PASSWORD = "correct horse"


# ---------------- tiện ích ----------------
def _png(width: int = 48, height: int = 32, mode: str = "RGB", color: Any = (40, 60, 80)) -> bytes:
    buffer = BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _login(client: TestClient, username: str) -> None:
    client.cookies.clear()
    response = client.post(
        "/api/v1/auth/register", json={"username": username, "password": PASSWORD}
    )
    assert response.status_code == 201


def _create(client: TestClient, data: bytes, name: str = "photo.png", **form: Any):
    return client.post(
        "/api/v1/workspace/sessions",
        files={"file": (name, data, "application/octet-stream")},
        data={key: str(value) for key, value in form.items()},
    )


class FakeAgent:
    """Giả lập start_session/end_session để test nhanh và tất định."""

    def __init__(self) -> None:
        self.started: List[Dict[str, Any]] = []
        self.ended: List[str] = []
        self.fail = False

    def start(self, image: np.ndarray, **kwargs: Any) -> SessionResult:
        if self.fail:
            raise RuntimeError("secret internal failure /data/app/users")
        self.started.append({"shape": image.shape, **kwargs})
        diagnosis = DiagnosisReport(
            scene_type="indoor",
            summary="Ảnh tối",
            defects=[Defect(type="underexposed", severity=2, origin="rule")],
            source="rule_based",
        )
        return SessionResult(
            session_id=f"thread-{len(self.started)}",
            status="needs_input",
            questions=[
                Question(
                    id="mood",
                    text="Ảnh đang khá tối. Giữ hay làm sáng?",
                    choices=[
                        Choice(value="keep_dark", label="Giữ", keep="low_key"),
                        Choice(value="brighten", label="Làm sáng", fix="underexposed"),
                    ],
                    default="brighten",
                )
            ],
            diagnosis=diagnosis,
        )

    def end(self, session_id: str) -> None:
        self.ended.append(session_id)


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> FakeAgent:
    fake = FakeAgent()
    monkeypatch.setattr(workspace, "start_session", fake.start)
    monkeypatch.setattr(workspace, "end_session", fake.end)
    return fake


@pytest.fixture
def client(app_db: Database) -> TestClient:
    test_client = TestClient(app)
    _login(test_client, "alice")
    return test_client


def _user_id(app_db: Database, username: str) -> str:
    with app_db.sessionmaker() as db:
        return db.scalar(select(User.id).where(User.username == username))


# ---------------- tạo phiên ----------------
def test_create_from_upload_runs_diagnosis(
    client: TestClient, agent: FakeAgent, app_db: Database
) -> None:
    response = _create(client, _png(), "My photo.png")
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_answers"
    assert body["title"] == "My photo"
    assert body["options"]["max_iterations"] == 3
    assert [m["kind"] for m in body["messages"]] == ["diagnosis", "questions"]
    assert body["diagnosis"]["source"] == "rule_based"
    assert [q["id"] for q in body["questions"]] == ["mood"]
    assert "thread" not in response.text  # thread_id không bao giờ ra client
    assert agent.started[0]["user_id"] == _user_id(app_db, "alice")
    assert agent.started[0]["shape"] == (32, 48, 3)

    original = client.get(body["original"]["url"])
    assert original.status_code == 200 and original.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(original.content)).size == (48, 32)
    thumb = client.get(body["thumbnail_url"])
    assert thumb.headers["content-type"] == "image/jpeg"

    listed = client.get("/api/v1/workspace/sessions").json()
    assert [item["id"] for item in listed] == [body["id"]]


def test_create_uses_user_default_settings(client: TestClient, agent: FakeAgent) -> None:
    client.patch("/api/v1/auth/me/settings", json={"max_iterations": 2})
    body = _create(client, _png()).json()
    assert body["options"]["max_iterations"] == 2
    assert agent.started[0]["max_iterations"] == 2
    explicit = _create(client, _png(), max_iterations=5).json()
    assert explicit["options"]["max_iterations"] == 5


def test_create_rejects_out_of_range_options(client: TestClient, agent: FakeAgent) -> None:
    response = _create(client, _png(), max_iterations=6)
    assert response.status_code == 422
    assert "max_iterations" in response.json()["error"]["fields"]
    assert agent.started == []


def test_create_requires_exactly_one_source(client: TestClient, agent: FakeAgent) -> None:
    neither = client.post("/api/v1/workspace/sessions", data={})
    assert neither.status_code == 422
    both = client.post(
        "/api/v1/workspace/sessions",
        files={"file": ("a.png", _png(), "image/png")},
        data={"sample": "underexposed"},
    )
    assert both.status_code == 422


def test_create_from_sample(
    client: TestClient, agent: FakeAgent, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    Image.new("RGB", (40, 30), (10, 10, 10)).save(samples / "coffee_underexposed.jpg")
    monkeypatch.setattr(workspace, "SAMPLES_DIR", samples)

    listed = client.get("/api/v1/workspace/samples").json()
    assert [item["id"] for item in listed] == ["underexposed"]  # chỉ mẫu có file thật
    assert client.get(listed[0]["thumbnail_url"]).headers["content-type"] == "image/jpeg"
    assert client.get("/api/v1/workspace/samples/noisy/thumbnail").status_code == 404

    body = client.post("/api/v1/workspace/sessions", data={"sample": "underexposed"}).json()
    assert body["title"] == "Ảnh mẫu: Thiếu sáng"
    assert body["options"]["sample"] == "underexposed"
    bad = client.post("/api/v1/workspace/sessions", data={"sample": "../../etc/passwd"})
    assert bad.status_code == 422


# ---------------- upload không tin cậy ----------------
def test_upload_rejects_non_images_and_gif(client: TestClient, agent: FakeAgent) -> None:
    text = _create(client, b"hello, not an image", "photo.png")
    assert text.status_code == 415
    assert text.json()["error"]["code"] == "unsupported_format"
    gif = BytesIO()
    Image.new("RGB", (20, 20)).save(gif, format="GIF")
    assert _create(client, gif.getvalue(), "anim.gif").status_code == 415
    assert agent.started == []


def test_upload_rejects_corrupt_image(client: TestClient, agent: FakeAgent) -> None:
    data = _png(64, 64)
    response = _create(client, data[: len(data) // 2])
    assert response.status_code in (415, 422)
    assert agent.started == []


def test_upload_rejects_too_large_file(app_db: Database, agent: FakeAgent) -> None:
    init_db(dataclasses.replace(app_db.settings, max_upload_mb=1))
    try:
        test_client = TestClient(app)
        _login(test_client, "carol")
        response = _create(test_client, b"\x89PNG" + b"0" * (1024 * 1024 + 10))
        assert response.status_code == 413
        assert response.json()["error"]["message"] == "Tệp vượt quá 1 MB."
    finally:
        close_db()


def test_upload_rejects_too_small_and_too_many_pixels(app_db: Database, agent: FakeAgent) -> None:
    init_db(dataclasses.replace(app_db.settings, max_image_pixels=1_000_000))
    try:
        test_client = TestClient(app)
        _login(test_client, "dave")
        tiny = _create(test_client, _png(8, 8))
        assert tiny.status_code == 422 and tiny.json()["error"]["code"] == "image_too_small"
        huge = _create(test_client, _png(1200, 1000))
        assert huge.status_code == 422 and huge.json()["error"]["code"] == "invalid_image"
    finally:
        close_db()


def test_exif_orientation_is_applied(client: TestClient, agent: FakeAgent) -> None:
    image = Image.new("RGB", (40, 20), (200, 10, 10))
    exif = image.getexif()
    exif[0x0112] = 6  # xoay 90° theo chiều kim đồng hồ khi hiển thị
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    body = _create(client, buffer.getvalue(), "rotated.jpg").json()
    assert (body["original"]["width"], body["original"]["height"]) == (20, 40)
    assert agent.started[0]["shape"] == (40, 20, 3)


@pytest.mark.parametrize("mode,color", [("L", 90), ("RGBA", (10, 20, 30, 0)), ("P", 3)])
def test_non_rgb_uploads_are_stored_as_rgb(
    client: TestClient, agent: FakeAgent, mode: str, color: Any
) -> None:
    body = _create(client, _png(mode=mode, color=color)).json()
    stored = Image.open(BytesIO(client.get(body["original"]["url"]).content))
    assert stored.mode == "RGB"
    assert agent.started[0]["shape"] == (32, 48, 3)


def test_ground_truth_must_match_size(client: TestClient, agent: FakeAgent) -> None:
    mismatch = client.post(
        "/api/v1/workspace/sessions",
        files={
            "file": ("a.png", _png(48, 32), "image/png"),
            "ground_truth": ("gt.png", _png(32, 32), "image/png"),
        },
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["error"]["code"] == "ground_truth_mismatch"
    ok = client.post(
        "/api/v1/workspace/sessions",
        files={
            "file": ("a.png", _png(48, 32), "image/png"),
            "ground_truth": ("gt.png", _png(48, 32), "image/png"),
        },
    ).json()
    assert ok["options"]["has_ground_truth"] is True
    assert ok["ground_truth"]["width"] == 48


# ---------------- lỗi chẩn đoán ----------------
def test_diagnosis_failure_keeps_session_failed_without_leaking(
    client: TestClient, agent: FakeAgent, app_db: Database
) -> None:
    agent.fail = True
    response = _create(client, _png())
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "pipeline_failed"
    assert "secret" not in response.text and "/data/app" not in response.text
    assert body["messages"][-1]["kind"] == "notice"
    with app_db.sessionmaker() as db:
        run = db.scalar(select(Run))
        assert run.status == "failed" and "secret internal failure" in run.error


# ---------------- quyền sở hữu ----------------
def test_other_user_cannot_access_session(client: TestClient, agent: FakeAgent) -> None:
    body = _create(client, _png()).json()
    sid = body["id"]
    original_url = body["original"]["url"]

    bob = TestClient(app)
    _login(bob, "bob")
    assert bob.get("/api/v1/workspace/sessions").json() == []
    assert bob.get(f"/api/v1/workspace/sessions/{sid}").status_code == 404
    assert bob.patch(f"/api/v1/workspace/sessions/{sid}", json={"title": "x"}).status_code == 404
    assert bob.delete(f"/api/v1/workspace/sessions/{sid}").status_code == 404
    assert bob.get(original_url).status_code == 404
    assert bob.get(f"{original_url}?download=1").status_code == 404
    # Phiên của alice vẫn còn nguyên
    assert client.get(f"/api/v1/workspace/sessions/{sid}").status_code == 200


def test_endpoints_require_login(app_db: Database) -> None:
    anonymous = TestClient(app)
    assert anonymous.get("/api/v1/workspace/sessions").status_code == 401
    assert anonymous.get("/api/v1/workspace/samples").status_code == 401
    assert anonymous.post("/api/v1/workspace/sessions", data={}).status_code == 401


def test_unknown_ids_are_404(client: TestClient) -> None:
    assert client.get("/api/v1/workspace/sessions/nope").status_code == 404
    assert client.get("/api/v1/workspace/sessions/" + "0" * 32).status_code == 404


# ---------------- đổi tên, tải về, xóa ----------------
def test_rename_session(client: TestClient, agent: FakeAgent) -> None:
    sid = _create(client, _png()).json()["id"]
    renamed = client.patch(f"/api/v1/workspace/sessions/{sid}", json={"title": "  Cà phê\n tối  "})
    assert renamed.status_code == 200 and renamed.json()["title"] == "Cà phê tối"
    empty = client.patch(f"/api/v1/workspace/sessions/{sid}", json={"title": "   "})
    assert empty.status_code == 422
    too_long = client.patch(f"/api/v1/workspace/sessions/{sid}", json={"title": "x" * 121})
    assert too_long.status_code == 422


def test_download_sets_attachment(client: TestClient, agent: FakeAgent) -> None:
    body = _create(client, _png(), "pic.png").json()
    response = client.get(body["original"]["download_url"])
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")
    assert "pic-original.png" in response.headers["content-disposition"]


def test_artifacts_are_revalidated_not_cached_blindly(client: TestClient, agent: FakeAgent) -> None:
    # Trình duyệt dùng chung: ảnh đã xóa hoặc của người khác không được lấy lại từ cache
    body = _create(client, _png(), "pic.png").json()
    response = client.get(body["original"]["url"])
    assert response.status_code == 200
    cache = response.headers["cache-control"]
    assert "no-cache" in cache and "private" in cache
    assert "immutable" not in cache and "max-age" not in cache


def test_delete_removes_records_and_files(
    client: TestClient, agent: FakeAgent, app_db: Database
) -> None:
    body = _create(client, _png()).json()
    sid = body["id"]
    folder = app_db.settings.users_dir / _user_id(app_db, "alice") / "sessions" / sid
    assert (folder / "original.png").is_file()

    assert client.delete(f"/api/v1/workspace/sessions/{sid}").status_code == 204
    assert not folder.exists()
    assert agent.ended == ["thread-1"]
    assert client.get(f"/api/v1/workspace/sessions/{sid}").status_code == 404
    assert client.get(body["original"]["url"]).status_code == 404
    with app_db.sessionmaker() as db:
        assert db.scalars(select(Run)).all() == []


def test_delete_running_session_is_rejected(
    client: TestClient, agent: FakeAgent, app_db: Database
) -> None:
    sid = _create(client, _png()).json()["id"]
    with app_db.sessionmaker() as db:
        db.get(ProcessingSession, sid).status = "running"
        db.commit()
    response = client.delete(f"/api/v1/workspace/sessions/{sid}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_busy"


# ---------------- pipeline rule-based thật ----------------
def test_create_with_real_rule_based_diagnosis(client: TestClient) -> None:
    dark = np.full((48, 64, 3), 25, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(dark).save(buffer, format="PNG")
    response = _create(client, buffer.getvalue(), max_iterations=1)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_answers"
    assert body["diagnosis"]["source"] == "rule_based"
    assert body["questions"][-1]["id"] == "approach"
    assert any(defect["type"] == "underexposed" for defect in body["diagnosis"]["defects"])
