"""
Test khung web app (S0): cấu hình, CSDL SQLite, envelope lỗi của router mới và phục vụ bản
build tĩnh. Chạy offline, mọi dữ liệu trong tmp_path.
"""

import sqlite3
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import select

from src.api.config import AppSettings, settings_from_env
from src.api.db import Database, close_db, init_db
from src.api.errors import ApiError, ApiRoute
from src.api.models import Message, ProcessingSession, User
from src.api.web_static import mount_web


# ---------------- cấu hình ----------------
def test_settings_defaults_and_clamping() -> None:
    settings = settings_from_env({})
    assert settings.data_dir == Path("data/app")
    assert settings.pipeline_concurrency == 1
    assert settings.cors_origins == ()
    assert settings.max_upload_bytes == 20 * 1024 * 1024

    custom = settings_from_env(
        {
            "PIPELINE_CONCURRENCY": "99",
            "MAX_UPLOAD_MB": "abc",
            "COOKIE_SECURE": "true",
            "ALLOW_REGISTRATION": "0",
            "CORS_ORIGINS": " http://a.test , ,http://b.test",
        }
    )
    assert custom.pipeline_concurrency == 4
    assert custom.max_upload_mb == 20
    assert custom.cookie_secure is True
    assert custom.allow_registration is False
    assert custom.cors_origins == ("http://a.test", "http://b.test")


# ---------------- CSDL ----------------
def test_init_db_creates_tables_with_wal(app_db: Database) -> None:
    path = app_db.settings.db_path
    assert path.is_file()
    connection = sqlite3.connect(str(path))
    try:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    finally:
        connection.close()
    assert mode == "wal"
    expected = {
        "users",
        "auth_tokens",
        "processing_sessions",
        "messages",
        "runs",
        "artifacts",
        "variants",
        "treatments",
        "ratings",
    }
    assert expected <= tables


def test_interrupted_sessions_marked_failed_on_restart(app_settings: AppSettings) -> None:
    database = init_db(app_settings)
    with database.sessionmaker() as db:
        user = User(username="alice", password_hash="x")
        db.add(user)
        db.flush()
        db.add(ProcessingSession(user_id=user.id, title="t", status="running"))
        db.add(ProcessingSession(user_id=user.id, title="ok", status="done"))
        db.commit()

    database = init_db(app_settings)  # mô phỏng khởi động lại server
    try:
        with database.sessionmaker() as db:
            statuses = {
                s.title: (s.status, s.error_code) for s in db.scalars(select(ProcessingSession))
            }
            notices = db.scalars(select(Message).where(Message.kind == "notice")).all()
        assert statuses["t"] == ("failed", "server_restarted")
        assert statuses["ok"] == ("done", None)
        assert len(notices) == 1
    finally:
        close_db()


def test_user_ids_are_uuid_hex(app_db: Database) -> None:
    with app_db.sessionmaker() as db:
        user = User(username="bob", password_hash="x")
        db.add(user)
        db.commit()
        assert len(user.id) == 32
        int(user.id, 16)


# ---------------- envelope lỗi ----------------
class _Body(BaseModel):
    count: int


def _error_app() -> FastAPI:
    router = APIRouter(prefix="/api/v1/test", route_class=ApiRoute)

    @router.get("/api-error")
    def api_error() -> None:
        raise ApiError(409, "session_busy")

    @router.get("/crash")
    def crash() -> None:
        raise RuntimeError("secret path C:/data/app/users/abc and key AIza-123")

    @router.post("/validate")
    def validate(body: _Body) -> dict:
        return {"count": body.count}

    app = FastAPI()
    app.include_router(router)
    return app


def test_api_error_envelope() -> None:
    client = TestClient(_error_app())
    response = client.get("/api/v1/test/api-error")
    assert response.status_code == 409
    assert response.json() == {
        "error": {"code": "session_busy", "message": "Phiên này đang được xử lý. Vui lòng đợi."}
    }


def test_unexpected_error_hides_details() -> None:
    client = TestClient(_error_app(), raise_server_exceptions=False)
    response = client.get("/api/v1/test/crash")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal"
    assert "secret" not in response.text and "AIza" not in response.text


def test_validation_error_envelope() -> None:
    client = TestClient(_error_app())
    response = client.post("/api/v1/test/validate", json={"count": "not-a-number"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation"
    assert "count" in error["fields"]
    assert "not-a-number" not in response.text


# ---------------- route cũ ----------------
def test_legacy_health_contract_unchanged(app_db: Database) -> None:
    from src.api.main import app

    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "intelligent-image-processing"}


# ---------------- bản build tĩnh ----------------
def _web_app(dist: Path) -> TestClient:
    app = FastAPI()
    router = APIRouter(prefix="/api/v1")

    @router.get("/ping")
    def ping() -> dict:
        return {"pong": True}

    app.include_router(router)
    assert mount_web(app, dist) is True
    return TestClient(app)


def test_mount_web_serves_spa_and_assets(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    client = _web_app(dist)

    assert "id=root" in client.get("/").text
    deep = client.get("/s/abc123")
    assert deep.status_code == 200 and "id=root" in deep.text
    asset = client.get("/assets/app.js")
    assert asset.text == "console.log(1)"
    assert "immutable" in asset.headers["cache-control"]
    assert client.get("/api/v1/ping").json() == {"pong": True}

    missing_api = client.get("/api/v1/nope")
    assert missing_api.status_code == 404
    assert missing_api.json() == {"detail": "Not Found"}

    traversal = client.get("/..%2Fsecret.txt")
    assert "top secret" not in traversal.text


def test_mount_web_without_build(tmp_path: Path) -> None:
    app = FastAPI()
    assert mount_web(app, tmp_path / "missing") is False
    assert TestClient(app).get("/").status_code == 404


def test_close_db_is_idempotent() -> None:
    close_db()
    close_db()
