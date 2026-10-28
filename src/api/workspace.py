"""
Router /api/v1/workspace: phiên xử lý của người dùng đang đăng nhập (tạo, xem, đổi tên, xóa),
ảnh qua endpoint có kiểm quyền và ảnh mẫu. Mọi endpoint yêu cầu đăng nhập; phiên của người
khác trả 404 như phiên không tồn tại (D7).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.agent.graph import run_pipeline
from src.agent.intent import Question, intent_from_answers
from src.agent.memory import get_case_memory, remember_run
from src.agent.refine import refine
from src.agent.session import (
    SessionNotFoundError,
    answer_session,
    end_session,
    start_session,
)
from src.agent.state import RegionOperation
from src.agent.user_analysis import apply_edits, parse_edits
from src.analyzer_evaluator.metadata import read_metadata
from src.region_engine import runtime_capabilities

from .auth import current_user, user_settings
from .config import AppSettings
from .db import get_db, get_settings
from .errors import MESSAGES, ApiError, ApiRoute
from .models import (
    Artifact,
    Message,
    ProcessingSession,
    Rating,
    Run,
    Treatment,
    User,
    new_id,
    utc_now,
)
from .schemas import MemoryStats
from .storage import (
    clean_title,
    decode_image,
    delete_tree,
    load_rgb,
    read_limited,
    resolve_inside,
    save_png,
    save_thumbnail,
    thumbnail_bytes,
    title_from_filename,
)
from .workspace_schemas import (
    MAX_DRAFT_EDITS,
    MAX_ITERATIONS_RANGE,
    AnalysisNoteIn,
    AnswersIn,
    RatingIn,
    RatingOut,
    RefineIn,
    RenameRequest,
    SampleOut,
    SessionDetailOut,
    SessionSummaryOut,
    SystemOut,
)
from .workspace_service import (
    API_BASE,
    add_message,
    answers_content,
    export_payload,
    gemini_mode,
    next_version,
    original_diagnosis,
    overlay_is_stale,
    plain,
    rating_dict,
    record_analyze,
    record_failure,
    record_process,
    refresh_region_overlay,
    run_timed,
    save_run_image,
    session_detail,
    session_path,
    stored_edits,
    summaries,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix=API_BASE, route_class=ApiRoute, tags=["workspace"])

# Ảnh mẫu: allowlist cứng, client chỉ gửi id (không bao giờ gửi đường dẫn)
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "real"
SAMPLES: Dict[str, Tuple[str, str]] = {
    "underexposed": ("coffee_underexposed.jpg", "Thiếu sáng"),
    "noisy": ("astro_noisy.jpg", "Nhiễu"),
    "warm_cast": ("cat_warm_cast.jpg", "Ám màu"),
}


# ---------------- tiện ích ----------------
def owned_session(db: Session, user: User, session_id: str) -> ProcessingSession:
    """Phiên của người dùng hiện tại; không có hoặc của người khác → 404."""
    item = db.get(ProcessingSession, session_id) if len(session_id) == 32 else None
    if item is None or item.user_id != user.id:
        raise ApiError(404, "not_found")
    return item


def _sample_path(sample_id: str) -> Optional[Path]:
    """Đường dẫn ảnh mẫu nếu id hợp lệ và file tồn tại."""
    entry = SAMPLES.get(sample_id)
    if entry is None:
        return None
    path = SAMPLES_DIR / entry[0]
    return path if path.is_file() else None


def _check_range(value: int, bounds: Tuple[int, int], field: str) -> int:
    """Kiểm tra tùy chọn số nằm trong giới hạn hiện có của pipeline."""
    if not bounds[0] <= value <= bounds[1]:
        raise ApiError(
            422,
            "validation",
            fields={field: f"Giá trị phải từ {bounds[0]} đến {bounds[1]}."},
        )
    return value


def _add_artifact(
    db: Session, item: ProcessingSession, kind: str, saved, run_id: Optional[str] = None
) -> Artifact:
    """Thêm bản ghi artifact cho một file vừa ghi."""
    artifact = Artifact(
        session_id=item.id,
        run_id=run_id,
        kind=kind,
        rel_path=saved.rel_path,
        width=saved.width,
        height=saved.height,
        bytes=saved.bytes,
    )
    db.add(artifact)
    return artifact


# ---------------- phiên xử lý ----------------
@router.get("/sessions", response_model=List[SessionSummaryOut])
def list_sessions(
    user: User = Depends(current_user), db: Session = Depends(get_db)
) -> List[SessionSummaryOut]:
    """Các phiên của người dùng, cập nhật gần nhất trước."""
    items = db.scalars(
        select(ProcessingSession)
        .where(ProcessingSession.user_id == user.id)
        .order_by(ProcessingSession.updated_at.desc())
    ).all()
    return summaries(db, list(items))


@router.post("/sessions", response_model=SessionDetailOut, status_code=201)
def create_session(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Form(None),
    ground_truth: Optional[UploadFile] = File(None),
    max_iterations: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> SessionDetailOut:
    """
    Tạo phiên từ ảnh upload hoặc ảnh mẫu, rồi chạy chẩn đoán vòng 1 và sinh câu hỏi ý định
    (start_session). Lỗi chẩn đoán không làm mất phiên: phiên ở trạng thái 'failed' kèm lỗi
    tiếng Việt để người dùng thử lại.
    """
    has_file = file is not None and bool(file.filename)
    if has_file == bool(sample):
        raise ApiError(422, "validation", fields={"file": "Chọn một ảnh hoặc một ảnh mẫu."})
    defaults = user_settings(user)
    iterations = _check_range(
        max_iterations or defaults.max_iterations, MAX_ITERATIONS_RANGE, "max_iterations"
    )

    if has_file:
        content = read_limited(file.file, settings)
        default_title = title_from_filename(file.filename)
    else:
        path = _sample_path(sample or "")
        if path is None:
            raise ApiError(422, "validation", fields={"sample": "Ảnh mẫu không tồn tại."})
        with path.open("rb") as stream:
            content = read_limited(stream, settings)
        default_title = f"Ảnh mẫu: {SAMPLES[sample][1]}"
    image = decode_image(content, settings)
    # EXIF/bảng lượng tử JPEG đọc từ bytes gốc (không bao giờ raise), cho prior của Perceive
    metadata = read_metadata(content)

    truth: Optional[np.ndarray] = None
    if ground_truth is not None and ground_truth.filename:
        truth = decode_image(read_limited(ground_truth.file, settings), settings)
        if truth.shape != image.shape:
            raise ApiError(422, "ground_truth_mismatch")

    item = ProcessingSession(
        id=new_id(),
        user_id=user.id,
        title=clean_title(title) or default_title or f"Phiên {utc_now()[:16].replace('T', ' ')}",
        status="running",
        options={
            "max_iterations": iterations,
            "num_variants": 1,  # Deprecated: luôn một kết quả mỗi lần chạy; giữ cho client cũ
            "has_ground_truth": truth is not None,
            "sample": sample if not has_file else None,
        },
    )
    base = session_path(settings, item)
    try:
        db.add(item)
        db.flush()
        original = _add_artifact(db, item, "original", save_png(image, base, "original.png"))
        _add_artifact(db, item, "thumbnail", save_thumbnail(image, base, "thumb.jpg"))
        if truth is not None:
            _add_artifact(db, item, "ground_truth", save_png(truth, base, "ground_truth.png"))
        db.commit()
    except Exception:
        db.rollback()
        delete_tree(settings, base)
        raise

    try:
        result, duration = run_timed(
            settings,
            start_session,
            image,
            max_iterations=iterations,
            user_id=str(user.id),
            image_metadata=metadata,
        )
        record_analyze(db, item, result, duration, original)
        if result.diagnosis is not None:
            refresh_region_overlay(db, settings, item, image, result.diagnosis)
    except Exception as exc:
        logger.exception("Diagnosis failed for processing session %s", item.id)
        db.rollback()
        record_failure(db, item, "analyze", exc, status="failed")
    db.commit()
    return session_detail(db, item)


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(
    session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> SessionDetailOut:
    """Toàn bộ phiên đọc từ CSDL (không chạy pipeline)."""
    return session_detail(db, owned_session(db, user, session_id))


@router.patch("/sessions/{session_id}", response_model=SessionSummaryOut)
def rename_session(
    session_id: str,
    body: RenameRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SessionSummaryOut:
    """Đổi tên phiên."""
    item = owned_session(db, user, session_id)
    title = clean_title(body.title)
    if not title or len(body.title.strip()) > 120:
        raise ApiError(422, "validation", fields={"title": "Tên phiên cần 1–120 ký tự."})
    item.title = title
    db.commit()
    return summaries(db, [item])[0]


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> Response:
    """Xóa phiên: bản ghi (cascade), thư mục ảnh trên đĩa và thread LangGraph nếu còn."""
    item = owned_session(db, user, session_id)
    if item.status == "running":
        raise ApiError(409, "session_busy")
    base = session_path(settings, item)
    thread_id = item.thread_id
    db.delete(item)
    db.commit()
    if thread_id:
        end_session(thread_id)
    try:
        delete_tree(settings, base)
    except OSError:
        # Bản ghi đã xóa; file sót lại chỉ chiếm chỗ, không lộ dữ liệu qua API
        logger.exception("Could not delete files of processing session %s", session_id)
    return Response(status_code=204)


@router.get("/sessions/{session_id}/artifacts/{artifact_id}")
def get_artifact(
    session_id: str,
    artifact_id: str,
    download: bool = False,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> FileResponse:
    """Một ảnh của phiên (kiểm quyền); download=1 → tải về với tên file đã làm sạch."""
    item = owned_session(db, user, session_id)
    artifact = db.get(Artifact, artifact_id) if len(artifact_id) == 32 else None
    if artifact is None or artifact.session_id != item.id:
        raise ApiError(404, "not_found")
    path = resolve_inside(session_path(settings, item), artifact.rel_path)
    if path is None or not path.is_file():
        raise ApiError(404, "not_found")
    media_type = "image/jpeg" if path.suffix == ".jpg" else "image/png"
    # no-cache: trình duyệt luôn hỏi lại server (kiểm quyền, có ETag) trước khi dùng bản cache,
    # để ảnh đã xóa hoặc của người dùng khác trên cùng trình duyệt không hiện lại từ cache
    headers = {"Cache-Control": "private, no-cache"}
    if download:
        name = "".join(c for c in item.title if c not in '\\/:*?"<>|').strip() or "anh"
        return FileResponse(
            path,
            media_type=media_type,
            headers=headers,
            filename=f"{name}-{artifact.kind}{path.suffix}",
        )
    return FileResponse(path, media_type=media_type, headers=headers)


# ---------------- hỏi ý và chạy pipeline ----------------
def _claim(db: Session, item: ProcessingSession, allowed: Tuple[str, ...]) -> str:
    """
    Chuyển phiên sang 'running' một cách nguyên tử (chống hai request cùng chạy một phiên).
    Trả trạng thái trước đó; phiên đang chạy → 409 session_busy, trạng thái khác → 409.
    """
    previous = item.status
    claimed = db.execute(
        update(ProcessingSession)
        .where(ProcessingSession.id == item.id, ProcessingSession.status.in_(allowed))
        .values(status="running", updated_at=utc_now())
    ).rowcount
    db.commit()
    db.refresh(item)
    if not claimed:
        raise ApiError(409, "session_busy" if item.status == "running" else "invalid_state")
    return previous


def _has_finished_process(db: Session, item: ProcessingSession) -> bool:
    """Phiên đã có một lần chạy pipeline hoàn tất chưa (P4: tối đa một lần)."""
    found = db.scalar(
        select(Run.id).where(Run.session_id == item.id, Run.kind == "process", Run.status == "done")
    )
    return found is not None


def _session_artifact(db: Session, item: ProcessingSession, kind: str) -> Optional[Artifact]:
    """Artifact gốc của phiên (original / ground_truth), không thuộc run nào."""
    return db.scalar(
        select(Artifact).where(
            Artifact.session_id == item.id, Artifact.kind == kind, Artifact.run_id.is_(None)
        )
    )


def _load_session_image(
    settings: AppSettings, item: ProcessingSession, artifact: Optional[Artifact]
) -> np.ndarray:
    """Đọc lại ảnh đã chuẩn hóa của phiên từ đĩa."""
    path = resolve_inside(session_path(settings, item), artifact.rel_path) if artifact else None
    if path is None or not path.is_file():
        raise FileNotFoundError("Session image is missing on disk")
    return load_rgb(path)


def _unparsed_notes(db: Session, item: ProcessingSession) -> List[str]:
    """Tin nhắn bổ sung phân tích không ra edit nào, của lần chẩn đoán hiện tại."""
    messages = db.scalars(
        select(Message).where(Message.session_id == item.id).order_by(Message.id)
    ).all()
    last_diagnosis = max((m.id for m in messages if m.kind == "diagnosis"), default=0)
    return [
        str(m.content.get("text", "")).strip()
        for m in messages
        if m.id > last_diagnosis
        and m.kind == "analysis_update"
        and m.content.get("unparsed")
        and str(m.content.get("text", "")).strip()
    ]


def _sync_overlay(db: Session, settings: AppSettings, item: ProcessingSession, draft: Any) -> None:
    """Dựng lại overlay khi tập vùng của bản nháp đổi (edit thêm/bỏ vùng); lỗi chỉ ghi log."""
    if draft is None or not overlay_is_stale(item, draft):
        return
    try:
        original = _session_artifact(db, item, "original")
        refresh_region_overlay(
            db, settings, item, _load_session_image(settings, item, original), draft
        )
    except Exception:
        logger.warning("Could not refresh region overlay of session %s", item.id, exc_info=True)


@router.post("/sessions/{session_id}/analysis-notes", response_model=SessionDetailOut)
def add_analysis_note(
    session_id: str,
    body: AnalysisNoteIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> SessionDetailOut:
    """
    Chat bổ sung phân tích: parse một tin nhắn thành DiagnosisEdit (luật trước, Gemini sau) và
    cộng dồn vào bản nháp của phiên; chưa chạy pipeline. Chỉ hợp lệ khi phiên chờ trả lời.
    Tin nhắn không ra edit nào được lưu với unparsed=True và nối vào notes khi bắt đầu xử lý.
    """
    item = owned_session(db, user, session_id)
    if item.status != "awaiting_answers":
        raise ApiError(409, "invalid_state")
    text = body.message.strip()
    if not text:
        raise ApiError(422, "validation", fields={"message": "Hãy nhập nội dung bổ sung."})
    base = original_diagnosis(db, item)
    if base is None:
        raise ApiError(409, "invalid_state")
    saved = stored_edits(item)
    if len(saved) >= MAX_DRAFT_EDITS:
        raise ApiError(
            422,
            "validation",
            fields={"message": f"Tối đa {MAX_DRAFT_EDITS} chỉnh sửa. Hãy xóa bớt trước."},
        )
    edits, source = parse_edits(text, apply_edits(base, saved))
    if len(saved) + len(edits) > MAX_DRAFT_EDITS:
        raise ApiError(
            422,
            "validation",
            fields={"message": f"Tối đa {MAX_DRAFT_EDITS} chỉnh sửa. Hãy xóa bớt trước."},
        )
    dumped = [edit.model_dump(mode="json") for edit in edits]
    item.draft_edits = [*(item.draft_edits or []), *dumped]
    draft = apply_edits(base, [*saved, *edits])
    add_message(db, item, "user", "analysis_note", {"text": text})
    add_message(
        db,
        item,
        "assistant",
        "analysis_update",
        {
            "text": text,
            "edits": dumped,
            "source": source,
            "draft_diagnosis": draft.model_dump(mode="json"),
            "unparsed": not edits,
        },
    )
    _sync_overlay(db, settings, item, draft)
    db.commit()
    return session_detail(db, item)


@router.delete("/sessions/{session_id}/analysis-edits/{index}", response_model=SessionDetailOut)
def delete_analysis_edit(
    session_id: str,
    index: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> SessionDetailOut:
    """Bỏ một chỉnh sửa khỏi bản nháp (theo vị trí, từ 0) và trả phiên với bản nháp mới."""
    item = owned_session(db, user, session_id)
    if item.status != "awaiting_answers":
        raise ApiError(409, "invalid_state")
    edits = list(item.draft_edits or [])
    if not 0 <= index < len(edits):
        raise ApiError(404, "not_found")
    del edits[index]
    item.draft_edits = edits
    base = original_diagnosis(db, item)
    if base is not None:
        _sync_overlay(db, settings, item, apply_edits(base, stored_edits(item)))
    db.commit()
    return session_detail(db, item)


@router.post("/sessions/{session_id}/answers", response_model=SessionDetailOut)
def answer_questions(
    session_id: str,
    body: AnswersIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> SessionDetailOut:
    """
    Trả lời câu hỏi ý định rồi chạy pipeline đến khi xong (đồng bộ). Không có ảnh chuẩn: chạy
    tiếp thread LangGraph (answer_session). Có ảnh chuẩn (P2): suy ý định từ câu hỏi đã lưu rồi
    gọi run_pipeline với ground truth.
    """
    item = owned_session(db, user, session_id)
    if not item.thread_id or not item.questions:
        raise ApiError(409, "invalid_state")
    _claim(db, item, ("awaiting_answers",))
    answers = {key[:64]: value[:64] for key, value in body.answers.items()}
    # Tin nhắn chat không parse được vẫn đến Plan prompt qua notes (T5)
    notes = "; ".join(part for part in (body.notes.strip(), *_unparsed_notes(db, item)) if part)
    edits = [dict(edit) for edit in (item.draft_edits or [])]
    thread_id = item.thread_id
    questions = list(item.questions)
    started = time.perf_counter()
    try:
        if item.options.get("has_ground_truth"):
            intent = intent_from_answers(
                [Question.model_validate(q) for q in questions],
                answers,
                notes,
                edits=stored_edits(item),
            )
            original = _load_session_image(settings, item, _session_artifact(db, item, "original"))
            truth = _load_session_image(settings, item, _session_artifact(db, item, "ground_truth"))
            state, duration = run_timed(
                settings,
                run_pipeline,
                image=original,
                ground_truth=truth,
                is_synthetic=True,
                max_iterations=int(item.options.get("max_iterations", 3)),
                intent=intent,
                user_id=str(user.id),
            )
        else:
            result, duration = run_timed(
                settings, answer_session, thread_id, answers, notes, edits=edits
            )
            if result.status != "done" or result.state is None:
                raise RuntimeError(f"Thread {thread_id} returned status {result.status}")
            state = result.state
        add_message(db, item, "user", "answers", answers_content(questions, answers, notes, edits))
        case_id = remember_run(state, str(user.id))
        record_process(db, settings, item, state, duration, case_id)
        db.commit()
    except SessionNotFoundError:
        # Thread mất do restart hoặc vượt MAX_SESSIONS (P7): giữ phiên chờ trả lời, hỏi ý lại
        db.rollback()
        item.status = "awaiting_answers"
        add_message(
            db,
            item,
            "system",
            "notice",
            {"code": "thread_expired", "text": MESSAGES["thread_expired"]},
        )
        db.commit()
        raise ApiError(409, "thread_expired")
    except Exception as exc:
        logger.exception("Pipeline failed for processing session %s", item.id)
        db.rollback()
        elapsed = int((time.perf_counter() - started) * 1000)
        record_failure(db, item, "process", exc, status="failed", duration_ms=elapsed)
        db.commit()
        end_session(thread_id)
        raise ApiError(500, "pipeline_failed")
    end_session(thread_id)
    return session_detail(db, item)


@router.post("/sessions/{session_id}/clarify", response_model=SessionDetailOut)
def clarify_again(
    session_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> SessionDetailOut:
    """
    Hỏi ý lại: chạy lại chẩn đoán vòng 1 trên ảnh đã lưu và sinh câu hỏi mới. Dùng khi thread
    hết hạn hoặc lần chạy trước lỗi; không hợp lệ khi phiên đã có kết quả.
    """
    item = owned_session(db, user, session_id)
    if _has_finished_process(db, item):
        raise ApiError(409, "invalid_state")
    _claim(db, item, ("awaiting_answers", "failed", "uploaded"))
    old_thread = item.thread_id
    try:
        original = _session_artifact(db, item, "original")
        result, duration = run_timed(
            settings,
            start_session,
            _load_session_image(settings, item, original),
            max_iterations=int(item.options.get("max_iterations", 3)),
            user_id=str(user.id),
        )
        record_analyze(db, item, result, duration, original)
        # Chẩn đoán mới → bỏ các edit của lần phân tích trước
        item.draft_edits = []
        if result.diagnosis is not None:
            refresh_region_overlay(
                db, settings, item, _load_session_image(settings, item, original), result.diagnosis
            )
        db.commit()
    except Exception as exc:
        logger.exception("Clarify failed for processing session %s", item.id)
        db.rollback()
        record_failure(db, item, "analyze", exc, status="failed")
        db.commit()
        raise ApiError(500, "pipeline_failed")
    if old_thread and old_thread != item.thread_id:
        end_session(old_thread)
    return session_detail(db, item)


# ---------------- lỗi render/refine ----------------
def _record_failed_run(db: Session, item: ProcessingSession, kind: str, exc: Exception) -> None:
    """Render/refine lỗi: chỉ thêm run 'failed', phiên vẫn 'done' (spec §3.3)."""
    db.rollback()
    db.add(
        Run(
            session_id=item.id,
            kind=kind,
            status="failed",
            result={},
            error=f"{type(exc).__name__}: {exc}"[:4000],
        )
    )
    db.commit()


# ---------------- tinh chỉnh theo góp ý ----------------
def _owned_treatment(
    db: Session, item: ProcessingSession, treatment_id: Optional[str]
) -> Treatment:
    """Phiên bản phác đồ của phiên (mặc định: bản đang dùng); không thuộc phiên → 404."""
    target = treatment_id or item.active_treatment_id
    treatment = db.get(Treatment, target) if target and len(target) == 32 else None
    if treatment is None or treatment.session_id != item.id:
        raise ApiError(404, "not_found")
    return treatment


@router.post("/sessions/{session_id}/refine", response_model=SessionDetailOut)
def refine_treatment(
    session_id: str,
    body: RefineIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> SessionDetailOut:
    """
    Chỉnh kết quả theo góp ý tiếng Việt ("da hơi vàng"): refine() phân tích góp ý thành điều
    chỉnh, áp vào phác đồ (mặc định bản đang dùng) rồi render lại từ ảnh gốc. Mỗi góp ý tạo một
    phiên bản phác đồ mới; ghi góp ý vào bộ nhớ ca bệnh.
    """
    item = owned_session(db, user, session_id)
    # Gộp khoảng trắng/xuống dòng; nội dung giữ nguyên để rule tiếng Việt của refine() đọc
    feedback = " ".join(body.feedback.split())[:500]
    if not feedback:
        raise ApiError(422, "validation", fields={"feedback": "Nhập góp ý (1–500 ký tự)."})
    if item.status != "done":
        raise ApiError(409, "session_busy" if item.status == "running" else "invalid_state")
    base = _owned_treatment(db, item, body.treatment_id)
    try:
        original = _load_session_image(settings, item, _session_artifact(db, item, "original"))
        actions = [RegionOperation.model_validate(action) for action in base.actions]
        result, duration = run_timed(settings, refine, original, actions, feedback)
        adjustments = plain(result.adjustments)
        run = Run(
            session_id=item.id,
            kind="refine",
            status="done",
            result={
                "feedback": feedback,
                "base_treatment_id": base.id,
                "adjustments": adjustments,
                "notes": list(result.notes),
                "source": result.source,
                "quality_score": plain(result.quality_score),
            },
            duration_ms=duration,
        )
        db.add(run)
        db.flush()
        version = next_version(db, item)
        artifact = save_run_image(
            db, settings, item, run, "refine", result.image, f"refine_v{version}.png"
        )
        treatment = Treatment(
            session_id=item.id,
            parent_id=base.id,
            version=version,
            source="refine",
            variant_id=base.variant_id,
            actions=plain(result.actions),
            feedback_text=feedback,
            adjustments=adjustments,
            notes=list(result.notes),
            refine_source=result.source,
            quality_score=plain(result.quality_score),
            artifact_id=artifact.id,
            run_id=run.id,
        )
        db.add(treatment)
        db.flush()
        item.active_treatment_id = treatment.id
        add_message(db, item, "user", "feedback", {"text": feedback, "base_version": base.version})
        add_message(
            db,
            item,
            "assistant",
            "refine_result",
            {
                "run_id": run.id,
                "treatment_id": treatment.id,
                "version": version,
                "base_version": base.version,
                "artifact_id": artifact.id,
                "adjustments": adjustments,
                "notes": list(result.notes),
                "source": result.source,
                "quality_score": plain(result.quality_score),
            },
        )
        db.commit()
    except ApiError:
        raise
    except Exception as exc:
        logger.exception("Refine failed for processing session %s", item.id)
        _record_failed_run(db, item, "refine", exc)
        raise ApiError(500, "pipeline_failed")
    memory = get_case_memory()
    if memory is not None and item.case_id and result.adjustments:
        try:
            memory.record_feedback(item.case_id, feedback, result.adjustments)
        except Exception:
            logger.warning("Could not record feedback for case %s", item.case_id)
    return session_detail(db, item)


@router.post(
    "/sessions/{session_id}/treatments/{treatment_id}/activate", response_model=SessionDetailOut
)
def activate_treatment(
    session_id: str,
    treatment_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SessionDetailOut:
    """Quay về (hoặc chuyển sang) một phiên bản phác đồ đã có; không chạy lại pipeline."""
    item = owned_session(db, user, session_id)
    treatment = _owned_treatment(db, item, treatment_id)
    if item.status != "done":
        raise ApiError(409, "session_busy" if item.status == "running" else "invalid_state")
    item.active_treatment_id = treatment.id
    db.commit()
    return session_detail(db, item)


# ---------------- đánh giá và xuất JSON ----------------
@router.put("/sessions/{session_id}/ratings", response_model=RatingOut)
def put_rating(
    session_id: str,
    body: RatingIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> RatingOut:
    """
    Đánh giá một kết quả (tốt/chưa tốt + ghi chú) để phục vụ đánh giá chất lượng về sau.
    Mỗi (run, phiên bản phác đồ) có tối đa một đánh giá; gửi lại là cập nhật.
    """
    item = owned_session(db, user, session_id)
    run = db.get(Run, body.run_id) if len(body.run_id) == 32 else None
    if run is None or run.session_id != item.id:
        raise ApiError(404, "not_found")
    if body.treatment_id is not None:
        treatment = db.get(Treatment, body.treatment_id) if len(body.treatment_id) == 32 else None
        if treatment is None or treatment.session_id != item.id:
            raise ApiError(404, "not_found")
    note = body.note.strip()
    # Tìm thủ công: UNIQUE của SQLite coi các treatment_id NULL là khác nhau
    rating = db.scalar(
        select(Rating).where(
            Rating.run_id == run.id,
            Rating.treatment_id.is_(None)
            if body.treatment_id is None
            else Rating.treatment_id == body.treatment_id,
        )
    )
    if rating is None:
        rating = Rating(
            session_id=item.id,
            run_id=run.id,
            treatment_id=body.treatment_id,
            rating=body.rating,
            note=note,
        )
        db.add(rating)
    else:
        rating.rating = body.rating
        rating.note = note
        rating.updated_at = utc_now()
    db.commit()
    return RatingOut(**rating_dict(rating))


@router.get("/sessions/{session_id}/export")
def export_session(
    session_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """
    Xuất toàn bộ phiên dạng JSON (spec §6) để phân tích về sau: không có ảnh, đường dẫn đĩa,
    thread_id hay lỗi nội bộ thô.
    """
    item = owned_session(db, user, session_id)
    payload = export_payload(db, item)
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": f'attachment; filename="session-{item.id}.json"',
            "Cache-Control": "no-store",
        },
    )


# ---------------- hệ thống và bộ nhớ ----------------
@router.get("/system", response_model=SystemOut)
def system_info(
    user: User = Depends(current_user),
    settings: AppSettings = Depends(get_settings),
) -> SystemOut:
    """
    Trạng thái máy chủ và chế độ chạy theo cấu hình. Có khóa Gemini KHÔNG chứng minh Gemini đã
    chạy: nguồn thật nằm ở diagnosis.source / plan.source của từng kết quả. Không gọi model.
    """
    return SystemOut(
        backend="ok",
        **gemini_mode(),
        memory_enabled=get_case_memory() is not None,
        pipeline_concurrency=settings.pipeline_concurrency,
        registration_open=settings.allow_registration,
        max_upload_mb=settings.max_upload_mb,
        region_runtime=runtime_capabilities(),
    )


@router.get("/memory", response_model=MemoryStats)
def my_memory(user: User = Depends(current_user)) -> MemoryStats:
    """Thống kê bộ nhớ ca bệnh của CHÍNH người dùng (user_id lấy từ cookie, không từ client)."""
    memory = get_case_memory()
    if memory is None:
        return MemoryStats(enabled=False)
    return MemoryStats(enabled=True, **memory.stats(str(user.id)))


# ---------------- ảnh mẫu ----------------
@router.get("/samples", response_model=List[SampleOut])
def list_samples(user: User = Depends(current_user)) -> List[SampleOut]:
    """Ảnh mẫu thực sự có trên máy chủ (data/real/ không được track trong git)."""
    return [
        SampleOut(id=key, label=label, thumbnail_url=f"{API_BASE}/samples/{key}/thumbnail")
        for key, (_, label) in SAMPLES.items()
        if _sample_path(key) is not None
    ]


@router.get("/samples/{sample_id}/thumbnail")
def sample_thumbnail(sample_id: str, user: User = Depends(current_user)) -> Response:
    """Thumbnail JPEG của một ảnh mẫu."""
    path = _sample_path(sample_id)
    if path is None:
        raise ApiError(404, "not_found")
    return Response(
        content=thumbnail_bytes(path),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )
