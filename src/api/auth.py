"""
Tài khoản người dùng: đăng ký, đăng nhập, đăng xuất, cài đặt (router /api/v1/auth).

Mật khẩu băm bằng argon2. Token đăng nhập là chuỗi ngẫu nhiên trong cookie httpOnly
SameSite=Lax; CSDL chỉ lưu SHA-256 của token kèm hạn dùng. Server luôn lấy người dùng từ
cookie, không bao giờ tin user_id do client gửi (D7).
"""

import hashlib
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import AppSettings
from .db import get_db, get_settings
from .errors import ApiError, ApiRoute
from .models import AuthToken, User, utc_now
from .workspace_schemas import (
    DEFAULT_USER_SETTINGS,
    Credentials,
    UserOut,
    UserSettings,
    UserSettingsUpdate,
)

logger = logging.getLogger(__name__)

COOKIE_NAME = "iid_token"
USERNAME_PATTERN = re.compile(r"^[a-z0-9_.-]{3,32}$")
PASSWORD_LENGTH = (8, 128)
USERNAME_RULE = "Tên đăng nhập gồm 3–32 ký tự: chữ cái không dấu, số, dấu _ . -"
PASSWORD_RULE = "Mật khẩu cần 8–128 ký tự."

_HASHER = PasswordHasher()
# Hash giả để đăng nhập với tên không tồn tại vẫn tốn thời gian như khi sai mật khẩu
_DUMMY_HASH = _HASHER.hash(secrets.token_urlsafe(16))

router = APIRouter(prefix="/api/v1/auth", route_class=ApiRoute, tags=["auth"])


# ---------------- tiện ích ----------------
def hash_token(token: str) -> str:
    """SHA-256 hex của token đăng nhập (chỉ giá trị này được lưu vào CSDL)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_time(value: str) -> datetime:
    """Đọc thời điểm ISO-8601; thiếu múi giờ thì coi là UTC."""
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def normalize_username(raw: str) -> str:
    """Tên đăng nhập không phân biệt hoa thường: lưu và so sánh bằng chữ thường."""
    return raw.strip().lower()


def _validate_credentials(username: str, password: str) -> None:
    """Kiểm tra định dạng tên đăng nhập và mật khẩu; lỗi → 422 kèm câu tiếng Việt theo trường."""
    fields: Dict[str, str] = {}
    if not USERNAME_PATTERN.match(username):
        fields["username"] = USERNAME_RULE
    if not PASSWORD_LENGTH[0] <= len(password) <= PASSWORD_LENGTH[1]:
        fields["password"] = PASSWORD_RULE
    if fields:
        raise ApiError(422, "validation", fields=fields)


def user_settings(user: User) -> UserSettings:
    """Cài đặt của người dùng, điền giá trị mặc định cho trường còn thiếu."""
    return UserSettings(**{**DEFAULT_USER_SETTINGS, **(user.settings or {})})


def user_out(user: User) -> UserOut:
    """Chuyển bản ghi User thành response (không có hash mật khẩu)."""
    return UserOut(
        id=user.id,
        username=user.username,
        settings=user_settings(user),
        created_at=user.created_at,
    )


def _issue_token(db: Session, user: User, response: Response, settings: AppSettings) -> None:
    """Tạo token mới, lưu hash vào CSDL và đặt cookie httpOnly."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    ttl = timedelta(days=settings.auth_token_ttl_days)
    # Dọn token đã hết hạn của người dùng này
    db.execute(
        delete(AuthToken).where(
            AuthToken.user_id == user.id, AuthToken.expires_at < now.isoformat()
        )
    )
    db.add(
        AuthToken(
            token_hash=hash_token(token),
            user_id=user.id,
            created_at=now.isoformat(),
            expires_at=(now + ttl).isoformat(),
        )
    )
    db.commit()
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=int(ttl.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def _clear_cookie(response: Response, settings: AppSettings) -> None:
    """Xóa cookie đăng nhập ở trình duyệt."""
    response.delete_cookie(
        COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=settings.cookie_secure
    )


# ---------------- dependency ----------------
def optional_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Người dùng của cookie hiện tại, hoặc None nếu không có / hết hạn / không hợp lệ."""
    token = request.cookies.get(COOKIE_NAME)
    if not token or len(token) > 256:
        return None
    record = db.get(AuthToken, hash_token(token))
    if record is None:
        return None
    if _parse_time(record.expires_at) <= datetime.now(timezone.utc):
        db.delete(record)
        db.commit()
        return None
    return db.get(User, record.user_id)


def current_user(user: Optional[User] = Depends(optional_user)) -> User:
    """Dependency bắt buộc đăng nhập cho mọi endpoint mới; không có → 401."""
    if user is None:
        raise ApiError(401, "unauthenticated")
    return user


# ---------------- endpoint ----------------
@router.post("/register", response_model=UserOut, status_code=201)
def register(
    body: Credentials,
    response: Response,
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> UserOut:
    """Tạo tài khoản rồi đăng nhập luôn (đặt cookie)."""
    if not settings.allow_registration:
        raise ApiError(403, "registration_closed")
    username = normalize_username(body.username)
    _validate_credentials(username, body.password)
    if db.scalar(select(User.id).where(User.username == username)) is not None:
        raise ApiError(409, "username_taken")
    user = User(
        username=username,
        password_hash=_HASHER.hash(body.password),
        settings=dict(DEFAULT_USER_SETTINGS),
        created_at=utc_now(),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Hai request đăng ký cùng tên chạy song song
        db.rollback()
        raise ApiError(409, "username_taken")
    logger.info("Registered user %s", user.id)
    _issue_token(db, user, response, settings)
    return user_out(user)


@router.post("/login", response_model=UserOut)
def login(
    body: Credentials,
    response: Response,
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> UserOut:
    """Đăng nhập; sai tên hoặc sai mật khẩu đều trả cùng một lỗi 401."""
    username = normalize_username(body.username)
    user = db.scalar(select(User).where(User.username == username))
    try:
        _HASHER.verify(user.password_hash if user else _DUMMY_HASH, body.password)
    except (VerificationError, InvalidHashError):
        raise ApiError(401, "invalid_credentials")
    if user is None:
        raise ApiError(401, "invalid_credentials")
    if _HASHER.check_needs_rehash(user.password_hash):
        user.password_hash = _HASHER.hash(body.password)
    _issue_token(db, user, response, settings)
    return user_out(user)


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> Response:
    """Đăng xuất: xóa token trong CSDL và cookie; luôn trả 204."""
    token = request.cookies.get(COOKIE_NAME)
    if token and len(token) <= 256:
        db.execute(delete(AuthToken).where(AuthToken.token_hash == hash_token(token)))
        db.commit()
    response = Response(status_code=204)
    _clear_cookie(response, settings)
    return response


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    """Người dùng đang đăng nhập."""
    return user_out(user)


@router.patch("/me/settings", response_model=UserOut)
def update_settings(
    body: UserSettingsUpdate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> UserOut:
    """Cập nhật số vòng / số phiên bản mặc định."""
    merged = user_settings(user).model_dump()
    merged.update(body.model_dump(exclude_none=True))
    user.settings = merged
    db.commit()
    return user_out(user)
