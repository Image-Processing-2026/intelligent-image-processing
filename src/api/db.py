"""
Kết nối SQLite (WAL) của web app. Bảng được tạo bằng create_all khi khởi tạo (chưa dùng
Alembic). CSDL được mở lười ở lần dùng đầu tiên, giống bộ nhớ ca bệnh.
"""

import logging
import threading
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import AppSettings, settings_from_env
from .models import Base, Message, ProcessingSession

logger = logging.getLogger(__name__)


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    """Bật WAL, khóa ngoại và thời gian chờ khóa cho mỗi kết nối SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


class Database:
    """Engine + sessionmaker của web app cùng cấu hình đã dùng để tạo nó."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.users_dir.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(
            f"sqlite:///{settings.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        event.listen(self.engine, "connect", _sqlite_pragmas)
        self.sessionmaker = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self._add_missing_columns()
        self._recover_interrupted()

    def _add_missing_columns(self) -> None:
        """
        create_all không sửa bảng đã có: thêm cột mới vào CSDL cũ bằng ALTER TABLE (chưa dùng
        Alembic). Mỗi phần tử là (bảng, cột, kiểu SQL kèm giá trị mặc định).
        """
        additions = (
            ("processing_sessions", "draft_edits", "JSON DEFAULT '[]'"),
            ("processing_sessions", "conversation", "JSON DEFAULT '{}'"),
        )
        inspector = inspect(self.engine)
        with self.engine.begin() as connection:
            for table, column, ddl in additions:
                existing = {col["name"] for col in inspector.get_columns(table)}
                if column not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                    logger.info("Added column %s.%s to the existing database.", table, column)

    def _recover_interrupted(self) -> None:
        """
        Phiên còn 'running' lúc khởi động là phiên bị ngắt giữa chừng (server dừng khi đang xử
        lý, P8): chuyển sang 'failed' và ghi thông báo vào hội thoại.
        """
        with self.sessionmaker() as db:
            stuck = db.scalars(
                select(ProcessingSession).where(ProcessingSession.status == "running")
            ).all()
            for item in stuck:
                item.status = (
                    ("done" if item.active_treatment_id else "awaiting_answers")
                    if item.conversation
                    else "failed"
                )
                item.error_code = "server_restarted"
                db.add(
                    Message(
                        session_id=item.id,
                        role="system",
                        kind="notice",
                        content={
                            "code": "server_restarted",
                            "text": "Máy chủ đã khởi động lại khi đang xử lý.",
                        },
                    )
                )
            if stuck:
                logger.warning("Marked %d interrupted processing sessions as failed.", len(stuck))
            db.commit()

    def dispose(self) -> None:
        """Đóng mọi kết nối (cần trên Windows trước khi xóa file DB)."""
        self.engine.dispose()


_DATABASE: Optional[Database] = None
_LOCK = threading.Lock()


def init_db(settings: Optional[AppSettings] = None) -> Database:
    """Khởi tạo (hoặc khởi tạo lại) CSDL với cấu hình cho trước; mặc định đọc từ env."""
    global _DATABASE
    with _LOCK:
        if _DATABASE is not None:
            _DATABASE.dispose()
        _DATABASE = Database(settings or settings_from_env())
        return _DATABASE


def get_database() -> Database:
    """CSDL đang dùng; mở lười từ env ở lần gọi đầu tiên."""
    global _DATABASE
    with _LOCK:
        if _DATABASE is None:
            _DATABASE = Database(settings_from_env())
        return _DATABASE


def close_db() -> None:
    """Đóng và bỏ CSDL đang dùng (dùng trong test)."""
    global _DATABASE
    with _LOCK:
        if _DATABASE is not None:
            _DATABASE.dispose()
        _DATABASE = None


def get_settings() -> AppSettings:
    """Dependency FastAPI: cấu hình của CSDL đang dùng."""
    return get_database().settings


def get_db() -> Iterator[Session]:
    """Dependency FastAPI: một Session SQLAlchemy cho mỗi request."""
    with get_database().sessionmaker() as db:
        yield db
