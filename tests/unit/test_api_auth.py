"""
Test tài khoản (S1): đăng ký, đăng nhập, đăng xuất, cookie, hạn token, cài đặt.
Chạy offline với CSDL trong tmp_path.
"""

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.api.auth import COOKIE_NAME, hash_token
from src.api.db import Database, close_db, init_db
from src.api.main import app
from src.api.models import AuthToken, User

PASSWORD = "correct horse"


@pytest.fixture
def client(app_db: Database) -> TestClient:
    return TestClient(app)


def _register(client: TestClient, username: str = "alice", password: str = PASSWORD):
    return client.post("/api/v1/auth/register", json={"username": username, "password": password})


def test_register_sets_httponly_cookie_and_logs_in(client: TestClient, app_db: Database) -> None:
    response = _register(client, "Alice")
    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "alice"
    assert body["settings"] == {"max_iterations": 3, "num_variants": 3}
    assert "password" not in response.text and "hash" not in response.text

    cookie = response.headers["set-cookie"]
    assert f"{COOKIE_NAME}=" in cookie
    assert "HttpOnly" in cookie and "samesite=lax" in cookie.lower()

    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200 and me.json()["id"] == body["id"]

    token = client.cookies.get(COOKIE_NAME)
    with app_db.sessionmaker() as db:
        user = db.scalar(select(User))
        stored = db.scalars(select(AuthToken)).all()
    assert user.password_hash.startswith("$argon2") and PASSWORD not in user.password_hash
    assert [record.token_hash for record in stored] == [hash_token(token)]
    assert token not in {record.token_hash for record in stored}


def test_duplicate_username_is_case_insensitive(client: TestClient) -> None:
    assert _register(client, "alice").status_code == 201
    client.cookies.clear()
    duplicate = _register(client, "ALICE")
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "username_taken"


def test_invalid_username_and_password_rules(client: TestClient) -> None:
    response = _register(client, "a b", "short")
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation"
    assert set(error["fields"]) == {"username", "password"}
    assert "short" not in response.text


def test_login_wrong_password_and_unknown_user_same_error(client: TestClient) -> None:
    _register(client)
    client.cookies.clear()
    wrong = client.post("/api/v1/auth/login", json={"username": "alice", "password": "nope-nope"})
    unknown = client.post("/api/v1/auth/login", json={"username": "ghost", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert wrong.json()["error"]["code"] == "invalid_credentials"
    assert client.get("/api/v1/auth/me").status_code == 401


def test_login_then_logout_revokes_token(client: TestClient, app_db: Database) -> None:
    _register(client)
    client.cookies.clear()
    login = client.post("/api/v1/auth/login", json={"username": "ALICE", "password": PASSWORD})
    assert login.status_code == 200
    token = client.cookies.get(COOKIE_NAME)
    assert client.get("/api/v1/auth/me").status_code == 200

    logout = client.post("/api/v1/auth/logout")
    assert logout.status_code == 204
    assert (
        f'{COOKIE_NAME}=""' in logout.headers["set-cookie"]
        or "Max-Age=0" in logout.headers["set-cookie"]
    )
    with app_db.sessionmaker() as db:
        assert db.get(AuthToken, hash_token(token)) is None

    # Token cũ bị đánh cắp cũng không dùng lại được
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/api/v1/auth/me").status_code == 401
    # Đăng xuất khi chưa đăng nhập vẫn 204
    client.cookies.clear()
    assert client.post("/api/v1/auth/logout").status_code == 204


def test_expired_token_rejected_and_removed(client: TestClient, app_db: Database) -> None:
    _register(client)
    token = client.cookies.get(COOKIE_NAME)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    with app_db.sessionmaker() as db:
        db.get(AuthToken, hash_token(token)).expires_at = past
        db.commit()
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"
    with app_db.sessionmaker() as db:
        assert db.get(AuthToken, hash_token(token)) is None


def test_garbage_cookie_is_unauthenticated(client: TestClient) -> None:
    client.cookies.set(COOKIE_NAME, "x" * 5000)
    assert client.get("/api/v1/auth/me").status_code == 401


def test_update_settings_within_existing_bounds(client: TestClient) -> None:
    _register(client)
    updated = client.patch("/api/v1/auth/me/settings", json={"num_variants": 1})
    assert updated.status_code == 200
    assert updated.json()["settings"] == {"max_iterations": 3, "num_variants": 1}
    too_many = client.patch("/api/v1/auth/me/settings", json={"max_iterations": 6})
    assert too_many.status_code == 422
    assert client.get("/api/v1/auth/me").json()["settings"]["max_iterations"] == 3


def test_settings_requires_login(client: TestClient) -> None:
    response = client.patch("/api/v1/auth/me/settings", json={"num_variants": 1})
    assert response.status_code == 401


def test_registration_can_be_closed(app_db: Database) -> None:
    init_db(dataclasses.replace(app_db.settings, allow_registration=False))
    try:
        response = _register(TestClient(app))
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "registration_closed"
    finally:
        close_db()
