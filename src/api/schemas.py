"""
Mô hình dữ liệu Request & Response cho REST API (API Schemas).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.agent.state import TreatmentPlan


class DiagnoseRequest(BaseModel):
    """Yêu cầu chỉ phân tích và chẩn đoán không qua xử lý."""

    image_base64: str = Field(..., description="Ảnh đầu vào dạng base64")


class DiagnoseResponse(BaseModel):
    """Kết quả chẩn đoán và đề xuất điều trị."""

    technical_metrics: Dict[str, Any]
    treatment_plan: TreatmentPlan


class ProcessResponse(BaseModel):
    """Kết quả xử lý ảnh hoàn chỉnh sau chu trình khép kín."""

    processed_image_base64: str
    total_iterations: int
    final_decision: str
    final_evaluation: Dict[str, Any]
    history: List[Dict[str, Any]]
    intermediate_images_base64: List[str] = Field(
        default_factory=list,
        description="Danh sách ảnh trung gian sau mỗi vòng lặp (Base64 PNG thumbnails)",
    )
