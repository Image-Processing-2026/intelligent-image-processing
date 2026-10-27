"""
Chạy pipeline tương tác (Phase 4): graph dừng ở node clarify để hỏi ý định người dùng, lưu
trạng thái bằng checkpointer theo thread_id, rồi chạy tiếp khi có câu trả lời.

Checkpointer là MemorySaver trong tiến trình: phiên mất khi khởi động lại server. Số phiên
giữ lại bị giới hạn (MAX_SESSIONS); phiên cũ nhất bị xóa khỏi checkpointer khi vượt giới hạn.
"""

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.analyzer_evaluator.metadata import ImageMetadata

from .graph import build_doctor_graph, initial_doctor_state
from .intent import Question
from .memory import DEFAULT_USER
from .state import DiagnosisEdit, DiagnosisReport, DoctorState

logger = logging.getLogger(__name__)

MAX_SESSIONS = 32

_CHECKPOINTER = MemorySaver()
_APP = build_doctor_graph(checkpointer=_CHECKPOINTER)
_SESSIONS: "OrderedDict[str, None]" = OrderedDict()
_LOCK = threading.Lock()


class SessionNotFoundError(KeyError):
    """Phiên không tồn tại (đã hết hạn hoặc chưa từng được tạo)."""


@dataclass
class SessionResult:
    """Kết quả một bước của phiên: cần người dùng trả lời, hoặc đã xử lý xong."""

    session_id: str
    status: str  # "needs_input" | "done"
    questions: List[Question] = field(default_factory=list)
    diagnosis: Optional[DiagnosisReport] = None
    state: Optional[DoctorState] = None


def _config(session_id: str) -> Dict[str, Any]:
    """Cấu hình LangGraph cho một phiên (thread_id = session_id)."""
    return {"configurable": {"thread_id": session_id}}


def _remember(session_id: str) -> None:
    """Ghi nhận phiên mới; xóa phiên cũ nhất khỏi checkpointer khi vượt MAX_SESSIONS."""
    with _LOCK:
        _SESSIONS[session_id] = None
        while len(_SESSIONS) > MAX_SESSIONS:
            expired, _ = _SESSIONS.popitem(last=False)
            _CHECKPOINTER.delete_thread(expired)
            logger.info("Session %s expired (more than %d sessions).", expired, MAX_SESSIONS)


def _result(session_id: str, output: Dict[str, Any]) -> SessionResult:
    """Đọc đầu ra của graph: còn interrupt → cần trả lời; không → đã xong."""
    interrupts = output.get("__interrupt__") or []
    if interrupts:
        payload = interrupts[0].value
        diagnosis = payload.get("diagnosis")
        return SessionResult(
            session_id=session_id,
            status="needs_input",
            questions=[Question.model_validate(q) for q in payload.get("questions") or []],
            diagnosis=DiagnosisReport.model_validate(diagnosis) if diagnosis else None,
        )
    return SessionResult(session_id=session_id, status="done", state=output)  # type: ignore[arg-type]


def start_session(
    image: np.ndarray,
    max_iterations: int = 3,
    session_id: Optional[str] = None,
    user_id: str = DEFAULT_USER,
    image_metadata: Optional[ImageMetadata] = None,
) -> SessionResult:
    """
    Bắt đầu một phiên tương tác: phân tích và chẩn đoán vòng 1 rồi dừng để hỏi ý định.
    image_metadata: metadata EXIF/JPEG của file gốc (read_metadata), đưa vào state cho Perceive.
    """
    session_id = session_id or uuid.uuid4().hex
    _remember(session_id)
    state = initial_doctor_state(
        image,
        max_iterations=max_iterations,
        interactive=True,
        user_id=user_id,
        image_metadata=image_metadata,
    )
    return _result(session_id, _APP.invoke(state, _config(session_id)))


def answer_session(
    session_id: str,
    answers: Optional[Dict[str, str]] = None,
    notes: str = "",
    edits: Optional[Sequence[Union[DiagnosisEdit, Dict[str, Any]]]] = None,
) -> SessionResult:
    """
    Trả lời câu hỏi của phiên và chạy tiếp pipeline đến khi xong. edits là các chỉnh sửa
    chẩn đoán đã parse ở tầng API (DiagnosisEdit hoặc dict); edit hỏng bị clarify bỏ qua.
    """
    with _LOCK:
        known = session_id in _SESSIONS
    if not known:
        raise SessionNotFoundError(session_id)
    snapshot = _APP.get_state(_config(session_id))
    if not snapshot.next:
        raise ValueError(f"session {session_id} is not waiting for input")
    payload_edits = [
        edit.model_dump(mode="json") if isinstance(edit, DiagnosisEdit) else edit
        for edit in edits or []
    ]
    command = Command(resume={"answers": answers or {}, "notes": notes, "edits": payload_edits})
    return _result(session_id, _APP.invoke(command, _config(session_id)))


def end_session(session_id: str) -> None:
    """Xóa phiên và trạng thái đã lưu của nó."""
    with _LOCK:
        _SESSIONS.pop(session_id, None)
    _CHECKPOINTER.delete_thread(session_id)
