"""
Các định tuyến API RESTful (API Routes).
Cung cấp các endpoint: /diagnose, /process, /render, /health.
"""

import base64
import json
from io import BytesIO

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import ValidationError

from src.agent.executor import execute_plan
from src.agent.graph import plan_treatment, run_pipeline
from src.agent.perception import perceive
from src.agent.planner import sanitize_actions
from src.agent.state import RegionOperation, TreatmentPlan
from src.analyzer_evaluator.analyzer import analyze_image

from .schemas import DiagnoseResponse, ProcessResponse, RenderResponse, VariantOut

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

        history_serialized = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in result_state.get("history", [])
        ]

        # Encode ảnh trung gian
        intermediate_b64 = [
            _encode_image_to_base64(img) for img in result_state.get("intermediate_images", [])
        ]

        return ProcessResponse(
            processed_image_base64=_encode_image_to_base64(result_state["current_image"]),
            total_iterations=result_state["iteration"] - 1,
            final_decision=result_state["decision"],
            final_evaluation=result_state.get("evaluation_result", {}),
            history=history_serialized,
            intermediate_images_base64=intermediate_b64,
            variants=[
                VariantOut(
                    **variant.model_dump(exclude={"image"}),
                    preview_base64=_encode_image_to_base64(variant.image),
                )
                for variant in result_state.get("variants") or []
            ],
            recommended_variant=result_state.get("recommended_variant"),
            variant_ranking_source=result_state.get("variant_ranking_source"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
