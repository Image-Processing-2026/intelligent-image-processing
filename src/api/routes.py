"""
Các định tuyến API RESTful (API Routes).
Cung cấp các endpoint: /diagnose, /process, /health.
"""

import base64
from io import BytesIO
import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np
from PIL import Image
from src.agent.graph import run_pipeline
from src.agent.planner import validate_and_sort_plan
from src.agent.vlm_diagnostician import diagnose_and_plan
from src.analyzer_evaluator.analyzer import analyze_image
from .schemas import DiagnoseResponse, ProcessResponse

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


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose_image_endpoint(file: UploadFile = File(...)):
    """Chỉ phân tích chỉ số kỹ thuật và chẩn đoán kế hoạch điều trị."""
    try:
        content = await file.read()
        img = _read_image_file(content)
        metrics = analyze_image(img).model_dump()
        plan = diagnose_and_plan(img, metrics, iteration=1)
        validated_plan = validate_and_sort_plan(plan)
        return DiagnoseResponse(
            technical_metrics=metrics,
            treatment_plan=validated_plan
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=ProcessResponse)
async def process_image_endpoint(
    file: UploadFile = File(...),
    ground_truth: UploadFile = File(None),
    max_iterations: int = Form(3)
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

        # Chạy quy trình LangGraph
        result_state = run_pipeline(
            image=img,
            ground_truth=gt_img,
            is_synthetic=is_synthetic,
            max_iterations=max_iterations
        )

        history_serialized = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in result_state.get("history", [])
        ]

        return ProcessResponse(
            processed_image_base64=_encode_image_to_base64(result_state["current_image"]),
            total_iterations=result_state["iteration"] - 1,
            final_decision=result_state["decision"],
            final_evaluation=result_state.get("evaluation_result", {}),
            history=history_serialized
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
