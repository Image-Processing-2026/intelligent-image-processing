"""
Cấu hình của web app (tài khoản, phiên xử lý, lưu trữ ảnh) đọc từ biến môi trường.
Route cũ (/diagnose, /process, …) không dùng cấu hình này.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    """Đọc số nguyên từ env, kẹp vào [low, high]; giá trị hỏng → mặc định."""
    try:
        value = int(env.get(name, "").strip() or default)
    except ValueError:
        value = default
    return min(max(value, low), high)


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Đọc cờ bật/tắt từ env ('1', 'true', 'yes', 'on' là bật)."""
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppSettings:
    """Cấu hình đã chuẩn hóa của web app."""

    data_dir: Path = Path("data/app")
    pipeline_concurrency: int = 1
    max_upload_mb: int = 20
    max_image_pixels: int = 40_000_000
    min_image_side: int = 16
    auth_token_ttl_days: int = 30
    cookie_secure: bool = False
    allow_registration: bool = True
    cors_origins: tuple[str, ...] = ()
    web_dist_dir: Path = Path("web/dist")

    @property
    def db_path(self) -> Path:
        """File SQLite của web app."""
        return self.data_dir / "app.sqlite"

    @property
    def users_dir(self) -> Path:
        """Thư mục gốc chứa ảnh theo người dùng."""
        return self.data_dir / "users"

    @property
    def max_upload_bytes(self) -> int:
        """Dung lượng upload tối đa tính bằng byte."""
        return self.max_upload_mb * 1024 * 1024


def settings_from_env(env: Optional[Mapping[str, str]] = None) -> AppSettings:
    """Tạo AppSettings từ biến môi trường (mặc định os.environ)."""
    env = os.environ if env is None else env
    origins = tuple(
        origin.strip() for origin in env.get("CORS_ORIGINS", "").split(",") if origin.strip()
    )
    return AppSettings(
        data_dir=Path(env.get("APP_DATA_DIR", "").strip() or "data/app"),
        pipeline_concurrency=_int(env, "PIPELINE_CONCURRENCY", 1, 1, 4),
        max_upload_mb=_int(env, "MAX_UPLOAD_MB", 20, 1, 200),
        max_image_pixels=_int(env, "MAX_IMAGE_PIXELS", 40_000_000, 1_000_000, 200_000_000),
        min_image_side=_int(env, "MIN_IMAGE_SIDE", 16, 1, 1024),
        auth_token_ttl_days=_int(env, "AUTH_TOKEN_TTL_DAYS", 30, 1, 365),
        cookie_secure=_bool(env, "COOKIE_SECURE", False),
        allow_registration=_bool(env, "ALLOW_REGISTRATION", True),
        cors_origins=origins,
        web_dist_dir=Path(env.get("WEB_DIST_DIR", "").strip() or "web/dist"),
    )
