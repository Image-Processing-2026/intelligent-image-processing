"""
Tầng dịch vụ của phiên xử lý: bọc các hàm có sẵn của src.agent (start_session, …) với giới
hạn số lần chạy pipeline đồng thời, ghi kết quả vào CSDL và dựng response từ CSDL.
Không đổi thuật toán, prompt hay ngưỡng của agent.
"""

import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.agent.region_overlay import diagnosis_region_overlay, executed_region_overlay
from src.agent.regions import canonical_region
from src.agent.session import SessionResult
from src.agent.state import DiagnosisEdit, DiagnosisReport
from src.agent.treatment import treatment_recipe
from src.agent.user_analysis import apply_edits, validate_edits
from src.agent.vlm_diagnostician import DEFAULT_GEMINI_MODEL
from src.region_engine import runtime_capabilities

from .config import AppSettings
from .errors import MESSAGES
from .models import (
    Artifact,
    Message,
    ProcessingSession,
    Rating,
    Run,
    Treatment,
    Variant,
    utc_now,
)
from .storage import save_png, session_dir
from .workspace_schemas import (
    ArtifactOut,
    MessageOut,
    SessionDetailOut,
    SessionError,
    SessionSummaryOut,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
API_BASE = "/api/v1/workspace"

# ---------------- giới hạn chạy đồng thời ----------------
_SLOTS: Dict[int, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()


def _slots(limit: int) -> threading.BoundedSemaphore:
    """Semaphore dùng chung cho mọi lời gọi pipeline với một giới hạn cho trước."""
    with _SLOTS_LOCK:
        if limit not in _SLOTS:
            _SLOTS[limit] = threading.BoundedSemaphore(limit)
        return _SLOTS[limit]


@contextmanager
def pipeline_slot(settings: AppSettings) -> Iterator[None]:
    """Chờ (xếp hàng) tới khi còn chỗ chạy pipeline; máy CPU không bị quá tải (D3, P3)."""
    slots = _slots(settings.pipeline_concurrency)
    slots.acquire()
    try:
        yield
    finally:
        slots.release()


# ---------------- theo dõi phiên đang chạy ----------------
# Phiên 'running' mà tiến trình này không còn chạy gì cho nó (luồng chết, bị hủy) là phiên mồ côi:
# không có ai ghi kết quả nên nó sẽ "chạy mãi". Server khởi động lại thì db._recover_interrupted lo.
_ACTIVE: Dict[str, int] = {}
_ACTIVE_LOCK = threading.Lock()
# Khoảng chờ giữa lúc phiên được đánh dấu 'running' và lúc pipeline thật sự đăng ký chạy
# (đọc ảnh từ đĩa); trong khoảng này phiên chưa bị coi là mồ côi.
STALE_GRACE_SECONDS = 120


@contextmanager
def active_session(session_id: Optional[str]) -> Iterator[None]:
    """Đăng ký phiên đang được tiến trình này xử lý (kể cả lúc xếp hàng chờ slot)."""
    if session_id is None:
        yield
        return
    with _ACTIVE_LOCK:
        _ACTIVE[session_id] = _ACTIVE.get(session_id, 0) + 1
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE[session_id] -= 1
            if _ACTIVE[session_id] <= 0:
                del _ACTIVE[session_id]


def is_active(session_id: str) -> bool:
    """Tiến trình này đang xử lý phiên không."""
    with _ACTIVE_LOCK:
        return session_id in _ACTIVE


def _age_seconds(timestamp: str) -> float:
    """Số giây từ một mốc ISO-8601 (utc_now) đến hiện tại; lỗi định dạng → rất lớn."""
    try:
        moment = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return float("inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds()


def recover_orphaned(db: Session, items: List[ProcessingSession]) -> bool:
    """
    Đưa các phiên 'running' mồ côi (không có luồng nào xử lý, quá STALE_GRACE_SECONDS) sang
    'failed' kèm thông báo để người dùng hỏi ý lại. Trả True nếu có phiên được sửa.
    """
    fixed = False
    for item in items:
        if (
            item.status == "running"
            and not is_active(item.id)
            and _age_seconds(item.updated_at) > STALE_GRACE_SECONDS
        ):
            item.status = (
                ("done" if item.active_treatment_id else "awaiting_answers")
                if item.conversation
                else "failed"
            )
            item.error_code = "interrupted"
            add_message(
                db,
                item,
                "system",
                "notice",
                {"code": "interrupted", "text": MESSAGES["interrupted"]},
            )
            fixed = True
    if fixed:
        db.commit()
    return fixed


def run_timed(
    settings: AppSettings,
    fn: Callable[..., T],
    *args: Any,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[T, int]:
    """
    Gọi một hàm agent trong slot pipeline; trả (kết quả, thời gian ms). session_id (nếu có)
    đánh dấu phiên đang được xử lý để không bị coi là mồ côi.
    """
    with active_session(session_id), pipeline_slot(settings):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, int((time.perf_counter() - started) * 1000)


# ---------------- đường dẫn và artifact ----------------
def session_path(settings: AppSettings, item: ProcessingSession) -> Path:
    """Thư mục trên đĩa của phiên."""
    return session_dir(settings, item.user_id, item.id)


def artifact_out(item: Artifact) -> ArtifactOut:
    """Artifact → response với URL có kiểm quyền."""
    url = f"{API_BASE}/sessions/{item.session_id}/artifacts/{item.id}"
    return ArtifactOut(
        id=item.id,
        kind=item.kind,
        url=url,
        download_url=f"{url}?download=1",
        width=item.width,
        height=item.height,
    )


def add_message(
    db: Session, item: ProcessingSession, role: str, kind: str, content: Dict[str, Any]
) -> Message:
    """Thêm một lượt hội thoại vào phiên."""
    message = Message(session_id=item.id, role=role, kind=kind, content=content)
    db.add(message)
    return message


# ---------------- ghi kết quả ----------------
def record_analyze(
    db: Session,
    item: ProcessingSession,
    result: SessionResult,
    duration_ms: int,
    original: Artifact,
) -> None:
    """Ghi kết quả start_session (chẩn đoán vòng 1 + câu hỏi): run 'analyze' và hai message."""
    diagnosis = result.diagnosis.model_dump(mode="json") if result.diagnosis else None
    questions = [question.model_dump(mode="json") for question in result.questions]
    run = Run(
        session_id=item.id,
        kind="analyze",
        status="done",
        result={"diagnosis": diagnosis, "questions": questions},
        duration_ms=duration_ms,
    )
    db.add(run)
    db.flush()
    add_message(
        db,
        item,
        "assistant",
        "diagnosis",
        {"run_id": run.id, "diagnosis": diagnosis, "original_artifact_id": original.id},
    )
    add_message(db, item, "assistant", "questions", {"run_id": run.id, "questions": questions})
    item.thread_id = result.session_id
    item.questions = questions
    item.status = "awaiting_answers" if result.status == "needs_input" else item.status
    item.error_code = None


def record_failure(
    db: Session,
    item: ProcessingSession,
    kind: str,
    exc: BaseException,
    status: str,
    code: str = "pipeline_failed",
    duration_ms: Optional[int] = None,
) -> None:
    """
    Ghi một lần chạy lỗi: run 'failed' giữ lỗi nội bộ (không trả lên UI), message notice tiếng
    Việt, và trạng thái phiên sau lỗi.
    """
    db.add(
        Run(
            session_id=item.id,
            kind=kind,
            status="failed",
            result={},
            error=f"{type(exc).__name__}: {exc}"[:4000],
            duration_ms=duration_ms,
        )
    )
    add_message(db, item, "system", "notice", {"code": code, "text": MESSAGES[code]})
    item.status = status
    item.error_code = code


# ---------------- kết quả pipeline ----------------
def plain(value: Any) -> Any:
    """
    Chuyển kết quả của agent (Pydantic model, numpy) thành JSON thuần để lưu CSDL và trả client:
    mảng numpy (ví dụ binary_mask) bị bỏ, số numpy → số Python, NaN/Inf → None.
    """
    if isinstance(value, BaseModel):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sources(history: List[Any]) -> Dict[str, List[str]]:
    """Nguồn THẬT của chẩn đoán và phác đồ theo từng vòng (không suy từ API key)."""
    return {
        "diagnosis": [item.diagnosis.source if item.diagnosis else "" for item in history],
        "plan": [item.plan.source if item.plan else "" for item in history],
    }


def next_version(db: Session, item: ProcessingSession) -> int:
    """Số phiên bản phác đồ tiếp theo trong phiên."""
    versions = db.scalars(select(Treatment.version).where(Treatment.session_id == item.id)).all()
    return max(versions, default=0) + 1


def save_run_image(
    db: Session,
    settings: AppSettings,
    item: ProcessingSession,
    run: Run,
    kind: str,
    image: np.ndarray,
    name: str,
) -> Artifact:
    """Ghi một ảnh của run vào runs/<run_id>/<name> và thêm bản ghi artifact."""
    saved = save_png(
        np.asarray(image, dtype=np.uint8), session_path(settings, item), f"runs/{run.id}/{name}"
    )
    artifact = Artifact(
        session_id=item.id,
        run_id=run.id,
        kind=kind,
        rel_path=saved.rel_path,
        width=saved.width,
        height=saved.height,
        bytes=saved.bytes,
    )
    db.add(artifact)
    db.flush()
    return artifact


def _executed_overlay_artifact(
    add_image: Callable[[str, np.ndarray, str], Artifact],
    state: Dict[str, Any],
    history: List[Any],
) -> Optional[Artifact]:
    """
    Overlay đúng các mask đã thực thi (tra theo mask_hash trong cache của tiến trình). Mask đã
    bị đẩy khỏi cache (LRU) thì bỏ qua, không báo lỗi; lỗi dựng overlay chỉ ghi log.
    """
    executed = [action for entry in history for action in getattr(entry, "executed", [])]
    original = state.get("original_image")
    if original is None or not executed:
        return None
    try:
        overlay, drawn = executed_region_overlay(original, executed)
        if not drawn:
            return None
        return add_image("executed_overlay", overlay, "executed_overlay.png")
    except Exception:
        logger.warning("Could not build executed-region overlay", exc_info=True)
        return None


def record_process(
    db: Session,
    settings: AppSettings,
    item: ProcessingSession,
    state: Dict[str, Any],
    duration_ms: int,
    case_id: Optional[str],
) -> Run:
    """
    Ghi trạng thái cuối của pipeline: ảnh kết quả, ảnh trung gian, ảnh preview các phiên bản lên
    đĩa; run 'process' (JSON không base64), các phiên bản, phác đồ v1 và message kết quả.
    """
    history = list(state.get("history") or [])
    run = Run(session_id=item.id, kind="process", status="done", result={}, duration_ms=duration_ms)
    db.add(run)
    db.flush()

    def add_image(kind: str, image: np.ndarray, name: str) -> Artifact:
        return save_run_image(db, settings, item, run, kind, image, name)

    final = add_image("final", state["current_image"], "final.png")
    executed_overlay = _executed_overlay_artifact(add_image, state, history)
    intermediates = [
        add_image("intermediate", image, f"iter_{index + 1}.png").id
        for index, image in enumerate(state.get("intermediate_images") or [])
    ]

    run.result = {
        "total_iterations": int(state.get("iteration", 1)) - 1,
        "final_decision": str(state.get("decision") or ""),
        "final_evaluation": plain(state.get("evaluation_result") or {}),
        # Kết cục trung thực (R-01) và kết quả review (R-08) của đúng ảnh được trả
        "outcome": plain(state.get("outcome")),
        "review": plain(state.get("review")),
        "history": plain(history),
        "treatment": plain(treatment_recipe(history)),
        "intent": plain(state.get("intent")),
        "rolled_back": bool(state.get("rolled_back")),
        "is_synthetic": bool(state.get("is_synthetic")),
        "case_id": case_id,
        "artifacts": {
            "final": final.id,
            "intermediate": intermediates,
            "executed_overlay": executed_overlay.id if executed_overlay else None,
        },
        "sources": _sources(history),
    }
    treatment = Treatment(
        session_id=item.id,
        version=next_version(db, item),
        source="pipeline",
        actions=run.result["treatment"],
        artifact_id=final.id,
        run_id=run.id,
    )
    db.add(treatment)
    db.flush()
    add_message(
        db,
        item,
        "assistant",
        "result",
        {
            "run_id": run.id,
            "final_artifact_id": final.id,
            "final_decision": run.result["final_decision"],
            "outcome_status": (run.result["outcome"] or {}).get("status"),
            "total_iterations": run.result["total_iterations"],
        },
    )
    item.status = "done"
    item.case_id = case_id
    item.active_treatment_id = treatment.id
    item.error_code = None
    return run


def answers_content(
    questions: List[Dict[str, Any]],
    answers: Dict[str, str],
    notes: str,
    edits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Nội dung message 'answers': giá trị thô + nhãn dễ đọc theo câu hỏi đã lưu (+ edit nháp)."""
    labelled = []
    for question in questions:
        value = answers.get(question.get("id", ""))
        choice = next((c for c in question.get("choices", []) if c.get("value") == value), None)
        if choice is not None:
            labelled.append({"question": question.get("text", ""), "answer": choice.get("label")})
    content: Dict[str, Any] = {"answers": answers, "labels": labelled, "notes": notes}
    if edits:
        content["edits"] = edits
    return content


# ---------------- chat bổ sung phân tích ----------------
REGION_OVERLAY_NAME = "region_overlay.png"


def original_diagnosis(db: Session, item: ProcessingSession) -> Optional[DiagnosisReport]:
    """Chẩn đoán vòng 1 đã lưu của lần analyze mới nhất (chưa áp edit của người dùng)."""
    analyze = _latest_run(db, item, "analyze")
    raw = analyze.result.get("diagnosis") if analyze else None
    if not raw:
        return None
    try:
        return DiagnosisReport.model_validate(raw)
    except ValueError:
        logger.warning("Stored diagnosis of session %s is not valid.", item.id, exc_info=True)
        return None


def stored_edits(item: ProcessingSession) -> List[DiagnosisEdit]:
    """Edit nháp của phiên (edit hỏng bị bỏ và ghi log)."""
    return validate_edits(item.draft_edits or [])


def draft_diagnosis(db: Session, item: ProcessingSession) -> Optional[DiagnosisReport]:
    """Chẩn đoán đã lưu + các edit đã lưu của người dùng (bản nháp trước khi xử lý)."""
    base = original_diagnosis(db, item)
    return apply_edits(base, stored_edits(item)) if base is not None else None


def _overlay_region_names(diagnosis: DiagnosisReport) -> List[str]:
    """Tên vùng (chuẩn hóa, trừ 'full') mà overlay của chẩn đoán này sẽ vẽ."""
    names = list(diagnosis.subjects) + [defect.region for defect in diagnosis.defects]
    return sorted({canonical_region(name) for name in names} - {"full"})


def overlay_artifact(db: Session, item: ProcessingSession) -> Optional[Artifact]:
    """Artifact overlay vùng của phiên (không thuộc run nào)."""
    return db.scalar(
        select(Artifact).where(
            Artifact.session_id == item.id,
            Artifact.kind == "region_overlay",
            Artifact.run_id.is_(None),
        )
    )


def overlay_is_stale(item: ProcessingSession, diagnosis: DiagnosisReport) -> bool:
    """Tập vùng của chẩn đoán đã khác với tập vùng của overlay đang lưu."""
    stored = (item.options or {}).get("overlay_regions")
    return stored != _overlay_region_names(diagnosis)


def refresh_region_overlay(
    db: Session,
    settings: AppSettings,
    item: ProcessingSession,
    image: np.ndarray,
    diagnosis: DiagnosisReport,
) -> Optional[Artifact]:
    """
    Dựng lại overlay vùng của chẩn đoán trên ảnh gốc và lưu cạnh original.png (thay bản cũ).
    Mask lấy từ cùng cache với Perceive/executor. Lỗi dựng overlay chỉ ghi log và trả None,
    không chặn phiên. Người gọi commit.
    """
    try:
        with pipeline_slot(settings):
            overlay, masks = diagnosis_region_overlay(image, diagnosis)
        saved = save_png(overlay, session_path(settings, item), REGION_OVERLAY_NAME)
    except Exception:
        logger.warning("Could not build region overlay for session %s", item.id, exc_info=True)
        return None
    artifact = overlay_artifact(db, item)
    if artifact is None:
        artifact = Artifact(session_id=item.id, kind="region_overlay", rel_path=saved.rel_path)
        db.add(artifact)
    artifact.rel_path = saved.rel_path
    artifact.width, artifact.height, artifact.bytes = saved.width, saved.height, saved.bytes
    options = dict(item.options or {})
    # Tên vùng đã yêu cầu (kể cả vùng không định vị được) để biết khi nào cần dựng lại
    options["overlay_regions"] = _overlay_region_names(diagnosis)
    options["regions"] = [
        {
            "name": name,
            "source": str(getattr(subject.source, "value", subject.source)),
            "confidence": round(float(subject.confidence), 3),
        }
        for name, subject in sorted(masks.items())
    ]
    item.options = options
    db.flush()
    return artifact


# ---------------- dựng response ----------------
def _latest_run(db: Session, item: ProcessingSession, kind: str) -> Optional[Run]:
    """Run 'done' mới nhất thuộc loại kind."""
    return db.scalars(
        select(Run)
        .where(Run.session_id == item.id, Run.kind == kind, Run.status == "done")
        .order_by(Run.created_at.desc())
        .limit(1)
    ).first()


def summaries(db: Session, items: List[ProcessingSession]) -> List[SessionSummaryOut]:
    """Danh sách cho sidebar; gom truy vấn thumbnail và quyết định cuối thành hai câu."""
    ids = [item.id for item in items]
    thumbs: Dict[str, Artifact] = {}
    decisions: Dict[str, str] = {}
    outcomes: Dict[str, Optional[str]] = {}
    if ids:
        for artifact in db.scalars(
            select(Artifact).where(Artifact.session_id.in_(ids), Artifact.kind == "thumbnail")
        ):
            thumbs[artifact.session_id] = artifact
        for run in db.scalars(
            select(Run)
            .where(Run.session_id.in_(ids), Run.kind == "process", Run.status == "done")
            .order_by(Run.created_at)
        ):
            decisions[run.session_id] = str(run.result.get("final_decision") or "") or None
            outcomes[run.session_id] = (run.result.get("outcome") or {}).get("status")
    return [
        SessionSummaryOut(
            id=item.id,
            title=item.title,
            status=item.status,
            created_at=item.created_at,
            updated_at=item.updated_at,
            thumbnail_url=artifact_out(thumbs[item.id]).url if item.id in thumbs else None,
            final_decision=decisions.get(item.id),
            outcome_status=outcomes.get(item.id),
        )
        for item in items
    ]


def _error(item: ProcessingSession) -> Optional[SessionError]:
    """Lỗi gần nhất của phiên ở trạng thái failed."""
    if item.status != "failed" or not item.error_code:
        return None
    message = MESSAGES.get(item.error_code, MESSAGES["pipeline_failed"])
    return SessionError(code=item.error_code, message=message)


def session_detail(db: Session, item: ProcessingSession) -> SessionDetailOut:
    """Toàn bộ phiên đọc từ CSDL."""
    summary = summaries(db, [item])[0]
    artifacts = db.scalars(select(Artifact).where(Artifact.session_id == item.id)).all()
    by_kind = {artifact.kind: artifact for artifact in artifacts if artifact.run_id is None}
    messages = db.scalars(
        select(Message).where(Message.session_id == item.id).order_by(Message.id)
    ).all()
    analyze = _latest_run(db, item, "analyze")
    process = _latest_run(db, item, "process")
    variants = db.scalars(
        select(Variant).where(Variant.session_id == item.id).order_by(Variant.rank)
    ).all()
    treatments = db.scalars(
        select(Treatment).where(Treatment.session_id == item.id).order_by(Treatment.version)
    ).all()
    ratings = db.scalars(select(Rating).where(Rating.session_id == item.id)).all()
    draft = draft_diagnosis(db, item) if item.status == "awaiting_answers" else None
    if item.status == "awaiting_answers" and (item.conversation or {}).get("diagnosis"):
        draft = DiagnosisReport.model_validate(item.conversation["diagnosis"])
    overlay = by_kind.get("region_overlay")
    return SessionDetailOut(
        **summary.model_dump(),
        conversation=item.conversation or {},
        options=item.options or {},
        original=artifact_out(by_kind["original"]) if "original" in by_kind else None,
        ground_truth=artifact_out(by_kind["ground_truth"]) if "ground_truth" in by_kind else None,
        messages=[
            MessageOut(
                id=message.id,
                role=message.role,
                kind=message.kind,
                content=message.content or {},
                created_at=message.created_at,
            )
            for message in messages
        ],
        diagnosis=(analyze.result.get("diagnosis") if analyze else None),
        questions=item.questions if item.status == "awaiting_answers" else None,
        process=({"run_id": process.id, **process.result} if process else None),
        variants=[variant_dict(variant) for variant in variants],
        treatments=[treatment_dict(treatment) for treatment in treatments],
        active_treatment_id=item.active_treatment_id,
        ratings=[rating_dict(rating) for rating in ratings],
        error=_error(item),
        artifacts=[artifact_out(artifact) for artifact in artifacts],
        draft_edits=list(item.draft_edits or []),
        draft_diagnosis=draft.model_dump(mode="json") if draft is not None else None,
        region_overlay_url=artifact_out(overlay).url if overlay is not None else None,
        regions=list((item.options or {}).get("regions", [])) if overlay is not None else [],
    )


def variant_dict(item: Variant) -> Dict[str, Any]:
    """Variant cũ (phiên trước khi bỏ phong cách, chỉ đọc) → JSON cho client."""
    return {
        "id": item.id,
        "run_id": item.run_id,
        "style_id": item.style_id,
        "label": item.label,
        "description": item.description,
        "rank": item.rank,
        "quality_score": item.quality_score,
        "quality_improved": item.quality_improved,
        "critic_note": item.critic_note,
        "actions": item.actions,
        "preview_artifact_id": item.preview_artifact_id,
        "recommended": item.recommended,
        "chosen": item.chosen,
    }


def treatment_dict(item: Treatment) -> Dict[str, Any]:
    """Treatment → JSON cho client."""
    return {
        "id": item.id,
        "parent_id": item.parent_id,
        "version": item.version,
        "source": item.source,
        "variant_id": item.variant_id,
        "actions": item.actions,
        "feedback_text": item.feedback_text,
        "adjustments": item.adjustments,
        "notes": item.notes,
        "refine_source": item.refine_source,
        "quality_score": item.quality_score,
        "artifact_id": item.artifact_id,
        "run_id": item.run_id,
        "created_at": item.created_at,
    }


def rating_dict(item: Rating) -> Dict[str, Any]:
    """Rating → JSON cho client."""
    return {
        "id": item.id,
        "run_id": item.run_id,
        "treatment_id": item.treatment_id,
        "rating": item.rating,
        "note": item.note,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


# ---------------- xuất JSON ----------------
EXPORT_SCHEMA = "iid.session-export/1"


def gemini_mode() -> Dict[str, Any]:
    """Chế độ chạy theo cấu hình (có khóa Gemini hay không); KHÔNG chứng minh Gemini đã chạy."""
    return {
        "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "gemini_model": os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
    }


def export_payload(db: Session, item: ProcessingSession) -> Dict[str, Any]:
    """
    Toàn bộ phiên để phân tích về sau (spec §6): hội thoại, các run (chẩn đoán, phác đồ, lịch
    sử, chỉ số, nguồn), phiên bản, phác đồ, đánh giá và metadata ảnh. Không có ảnh, đường dẫn
    đĩa, thread_id hay lỗi nội bộ thô.
    """
    by_session = lambda model: select(model).where(model.session_id == item.id)  # noqa: E731
    messages = db.scalars(by_session(Message).order_by(Message.id)).all()
    runs = db.scalars(by_session(Run).order_by(Run.created_at)).all()
    variants = db.scalars(by_session(Variant).order_by(Variant.rank)).all()
    treatments = db.scalars(by_session(Treatment).order_by(Treatment.version)).all()
    ratings = db.scalars(by_session(Rating).order_by(Rating.created_at)).all()
    artifacts = db.scalars(by_session(Artifact).order_by(Artifact.created_at)).all()
    return {
        "schema": EXPORT_SCHEMA,
        "exported_at": utc_now(),
        "app": gemini_mode(),
        "region_runtime": runtime_capabilities(),
        "session": {
            "id": item.id,
            "title": item.title,
            "status": item.status,
            "options": item.options or {},
            "conversation": item.conversation or {},
            "draft_edits": item.draft_edits or [],
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "case_id": item.case_id,
            "active_treatment_id": item.active_treatment_id,
        },
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "kind": m.kind,
                "content": m.content or {},
                "created_at": m.created_at,
            }
            for m in messages
        ],
        "runs": [
            {
                "id": r.id,
                "kind": r.kind,
                "status": r.status,
                "duration_ms": r.duration_ms,
                "result": r.result or {},
                "created_at": r.created_at,
            }
            for r in runs
        ],
        "variants": [variant_dict(v) for v in variants],
        "treatments": [treatment_dict(t) for t in treatments],
        "ratings": [rating_dict(r) for r in ratings],
        "artifacts": [
            {
                "id": a.id,
                "kind": a.kind,
                "width": a.width,
                "height": a.height,
                "bytes": a.bytes,
                "run_id": a.run_id,
            }
            for a in artifacts
        ],
    }
