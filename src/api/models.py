"""
Mô hình dữ liệu SQLAlchemy 2.0 của web app (docs/frontend/spec.md §2).
Ảnh nằm trên đĩa; bảng chỉ giữ đường dẫn tương đối và metadata.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SESSION_STATUSES: tuple[str, ...] = ("uploaded", "awaiting_answers", "running", "done", "failed")
MESSAGE_KINDS: tuple[str, ...] = (
    "diagnosis",
    "questions",
    "answers",
    "analysis_note",
    "analysis_update",
    "result",
    "feedback",
    "refine_result",
    "render_result",
    "notice",
)
RUN_KINDS: tuple[str, ...] = ("analyze", "process", "render", "refine")
ARTIFACT_KINDS: tuple[str, ...] = (
    "original",
    "ground_truth",
    "thumbnail",
    "final",
    "variant_preview",
    "intermediate",
    "region_overlay",
    "executed_overlay",
    "render",
    "refine",
)


def new_id() -> str:
    """Id ngẫu nhiên dạng UUID hex (không tái dùng khi xóa DB, xem P1)."""
    return uuid.uuid4().hex


def utc_now() -> str:
    """Thời điểm hiện tại, ISO-8601 UTC."""
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    """Lớp nền của mọi bảng."""


class User(Base):
    """Tài khoản người dùng; str(id) là user_id của CaseMemory."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class AuthToken(Base):
    """Token đăng nhập; chỉ lưu SHA-256 của token, token thô nằm trong cookie."""

    __tablename__ = "auth_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)
    expires_at: Mapped[str] = mapped_column(String(40))


class ProcessingSession(Base):
    """Phiên xử lý: đơn vị người dùng nhìn thấy (một ảnh, một hội thoại)."""

    __tablename__ = "processing_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="uploaded")
    options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Thread LangGraph (src/agent/session.py); không bao giờ gửi ra client
    thread_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    case_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    questions: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)
    # Các DiagnosisEdit (dict) người dùng đã bổ sung khi chat "bổ sung phân tích"; NULL ở phiên cũ
    draft_edits: Mapped[Optional[list[Any]]] = mapped_column(JSON, default=list, nullable=True)
    # Không đặt ForeignKey để tránh vòng phụ thuộc bảng với treatments
    active_treatment_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now, onupdate=utc_now)


class Message(Base):
    """Một lượt hội thoại của phiên xử lý."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("processing_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(20))
    content: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Run(Base):
    """Một lần gọi backend xử lý (analyze/process/render/refine) và kết quả của nó."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("processing_sessions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Lỗi nội bộ để chẩn đoán; không trả nguyên văn lên UI
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Artifact(Base):
    """Một file ảnh của phiên xử lý trên đĩa."""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("processing_sessions.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(20))
    # Tương đối với thư mục phiên, do server sinh
    rel_path: Mapped[str] = mapped_column(String(255))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Variant(Base):
    """Một phiên bản phong cách của lần chạy pipeline."""

    __tablename__ = "variants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("processing_sessions.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    style_id: Mapped[str] = mapped_column(String(20))
    label: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text, default="")
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_improved: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    critic_note: Mapped[str] = mapped_column(Text, default="")
    actions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    preview_artifact_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True
    )
    recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    chosen: Mapped[bool] = mapped_column(Boolean, default=False)


class Treatment(Base):
    """Một phiên bản phác đồ (pipeline, render phiên bản, hoặc tinh chỉnh theo góp ý)."""

    __tablename__ = "treatments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("processing_sessions.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("treatments.id", ondelete="SET NULL"), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16))
    variant_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("variants.id", ondelete="SET NULL"), nullable=True
    )
    actions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    feedback_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    adjustments: Mapped[list[Any]] = mapped_column(JSON, default=list)
    notes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    refine_source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    artifact_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Rating(Base):
    """Đánh giá của người dùng cho một kết quả (tốt/chưa tốt + ghi chú)."""

    __tablename__ = "ratings"
    __table_args__ = (UniqueConstraint("run_id", "treatment_id", name="uq_rating_target"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("processing_sessions.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    treatment_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("treatments.id", ondelete="CASCADE"), nullable=True
    )
    rating: Mapped[str] = mapped_column(String(8))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now, onupdate=utc_now)
