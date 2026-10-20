"""
Các định tuyến API RESTful (API Routes).
Cung cấp các endpoint: /diagnose, /process, /render, /refine, /sessions, /health.
"""

import base64
import json
from io import BytesIO
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import ValidationError

from src.agent.executor import execute_plan
from src.agent.graph import plan_treatment, run_pipeline
from src.agent.perception import perceive
from src.agent.planner import sanitize_actions
from src.agent.refine import refine
from src.agent.session import (
    SessionNotFoundError,
    SessionResult,
    answer_session,
    end_session,
    start_session,
)
from src.agent.state import RegionOperation, TreatmentPlan
from src.agent.variants import treatment_recipe
from src.analyzer_evaluator.analyzer import analyze_image

from .schemas import (
    AnswerRequest,
    DiagnoseResponse,
    ProcessResponse,
    RefineResponse,
    RenderResponse,
    SessionResponse,
    VariantOut,
)

router = APIRouter(prefix="/api/v1")


def _read_image_file(file_bytes: bytes) -> np.ndarray:
    """Chuyển đổi bytes file ảnh tải lên thành np.ndarray RGB."""
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def _encode_image_to_base64(img_array: np.ndarray) -> str:
    """Chuyển đổi np.ndarray RGB thành chuỗi Base64 PNG."""
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@router.get("/health")
def health_check():
    """Kiểm tra trạng thái hoạt động của backend service."""
    return {"status": "ok", "service": "intelligent-image-processing"}


def _diagnose(img: np.ndarray) -> DiagnoseResponse:
    """Chỉ số kỹ thuật → chẩn đoán (Perceive) → kế hoạch điều trị, không xử lý ảnh."""
    metrics = analyze_image(img).model_dump()
    diagnosis = perceive(img, metrics, iteration=1)
    plan = plan_treatment(img, metrics, diagnosis, iteration=1)
    return DiagnoseResponse(technical_metrics=metrics, diagnosis=diagnosis, treatment_plan=plan)


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose_image_endpoint(file: UploadFile = File(...)):
    """Phân tích chỉ số kỹ thuật, chẩn đoán (Perceive) và lập kế hoạch điều trị, không xử lý ảnh."""
    try:
        content = await file.read()
        img = _read_image_file(content)
        # Hai lời gọi Gemini + phát hiện khuôn mặt là blocking: chạy ngoài event loop
        return await run_in_threadpool(_diagnose, img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=ProcessResponse)
async def process_image_endpoint(
    file: UploadFile = File(...),
    ground_truth: UploadFile = File(None),
    max_iterations: int = Form(3),
    num_variants: int = Form(1),
):
    """Thực thi toàn bộ chu trình xử lý ảnh khép kín với LangGraph."""
    try:
        content = await file.read()
        img = _read_image_file(content)

        gt_img = None
        is_synthetic = False
        if ground_truth is not None:
            gt_content = await ground_truth.read()
            gt_img = _read_image_file(gt_content)
            is_synthetic = True

        # Chạy quy trình LangGraph (blocking: nhiều lời gọi Gemini và xử lý CPU) ngoài event loop
        result_state = await run_in_threadpool(
            run_pipeline,
            image=img,
            ground_truth=gt_img,
            is_synthetic=is_synthetic,
            max_iterations=max_iterations,
            num_variants=num_variants,
        )

        return _process_response(result_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _process_response(result_state: Dict[str, Any]) -> ProcessResponse:
    """Đóng gói trạng thái cuối của pipeline thành ProcessResponse (ảnh dạng Base64 PNG)."""
    history = result_state.get("history", [])
    return ProcessResponse(
        processed_image_base64=_encode_image_to_base64(result_state["current_image"]),
        total_iterations=result_state["iteration"] - 1,
        final_decision=result_state["decision"],
        final_evaluation=result_state.get("evaluation_result", {}),
        history=[item.model_dump() if hasattr(item, "model_dump") else item for item in history],
        intermediate_images_base64=[
            _encode_image_to_base64(img) for img in result_state.get("intermediate_images", [])
        ],
        variants=[
            VariantOut(
                **variant.model_dump(exclude={"image"}),
                preview_base64=_encode_image_to_base64(variant.image),
            )
            for variant in result_state.get("variants") or []
        ],
        recommended_variant=result_state.get("recommended_variant"),
        variant_ranking_source=result_state.get("variant_ranking_source"),
        treatment=treatment_recipe(history),
        intent=result_state.get("intent"),
    )


def _parse_actions(raw: str) -> list:
    """Parse danh sách thao tác (JSON) của một phiên bản; lỗi định dạng → HTTP 422."""
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("actions must be a JSON list")
        return [RegionOperation.model_validate(item) for item in items]
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid actions: {exc}") from exc


@router.post("/render", response_model=RenderResponse)
async def render_variant_endpoint(file: UploadFile = File(...), actions: str = Form(...)):
    """
    Render một phiên bản đã chọn ở độ phân giải gốc: áp đúng danh sách thao tác của phiên bản
    (trường `actions` trong /process) lên ảnh gốc. Thao tác được lọc và kẹp như planner nhưng
    GIỮ NGUYÊN thứ tự, để kết quả khớp với ảnh preview người dùng đã chọn.
    """
    parsed = _parse_actions(actions)
    try:
        img = _read_image_file(await file.read())
        applied = sanitize_actions(parsed)
        plan = TreatmentPlan(reasoning="render", actions=applied)
        rendered = await run_in_threadpool(execute_plan, img, plan)
        return RenderResponse(
            image_base64=_encode_image_to_base64(rendered),
            width=int(rendered.shape[1]),
            height=int(rendered.shape[0]),
            applied_actions=applied,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine", response_model=RefineResponse)
async def refine_endpoint(
    file: UploadFile = File(...), actions: str = Form(...), feedback: str = Form(...)
):
    """
    Chỉnh kết quả theo góp ý ("da hơi vàng, trời gắt quá"): phân tích góp ý thành điều chỉnh,
    áp vào phác đồ `actions` (treatment hoặc actions của một phiên bản) rồi render lại từ
    ảnh gốc ở độ phân giải gốc. Gửi lại `actions` trả về để góp ý tiếp.
    """
    parsed = _parse_actions(actions)
    try:
        img = _read_image_file(await file.read())
        result = await run_in_threadpool(refine, img, parsed, feedback)
        return RefineResponse(
            image_base64=_encode_image_to_base64(result.image),
            width=int(result.image.shape[1]),
            height=int(result.image.shape[0]),
            actions=result.actions,
            adjustments=result.adjustments,
            notes=result.notes,
            source=result.source,
            quality_score=result.quality_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _session_response(result: SessionResult) -> SessionResponse:
    """Đóng gói một bước của phiên tương tác."""
    return SessionResponse(
        session_id=result.session_id,
        status=result.status,
        questions=result.questions,
        diagnosis=result.diagnosis,
        result=_process_response(result.state) if result.state is not None else None,
    )


@router.post("/sessions", response_model=SessionResponse)
async def start_session_endpoint(
    file: UploadFile = File(...),
    max_iterations: int = Form(3),
    num_variants: int = Form(3),
):
    """
    Bắt đầu phiên tương tác: phân tích, chẩn đoán vòng 1 rồi dừng để hỏi ý định
    (status='needs_input', kèm câu hỏi). Trả lời qua POST /sessions/{id}/answers.
    """
    try:
        img = _read_image_file(await file.read())
        result = await run_in_threadpool(
            start_session, img, max_iterations=max_iterations, num_variants=num_variants
        )
        return _session_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/answers", response_model=SessionResponse)
async def answer_session_endpoint(session_id: str, request: AnswerRequest):
    """Trả lời câu hỏi của phiên; pipeline chạy tiếp và trả kết quả (status='done')."""
    try:
        result = await run_in_threadpool(answer_session, session_id, request.answers, request.notes)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _session_response(result)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session_endpoint(session_id: str) -> None:
    """Xóa phiên và trạng thái đã lưu."""
    end_session(session_id)
