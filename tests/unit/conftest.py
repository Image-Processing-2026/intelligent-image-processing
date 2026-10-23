"""
Fixtures cho test của web app (src/api): CSDL và thư mục dữ liệu trong tmp_path.
Không autouse: chỉ test API mới dùng.
"""

from pathlib import Path
from typing import Iterator

import pytest

from src.api.config import AppSettings
from src.api.db import Database, close_db, init_db


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    """Cấu hình web app trỏ vào tmp_path (không đụng data/app của repo)."""
    return AppSettings(data_dir=tmp_path / "app", web_dist_dir=tmp_path / "dist")


@pytest.fixture
def app_db(app_settings: AppSettings) -> Iterator[Database]:
    """CSDL web app mới cho mỗi test; đóng kết nối sau test để Windows xóa được file."""
    database = init_db(app_settings)
    yield database
    close_db()
