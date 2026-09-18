"""
Trạng thái làm việc của LangGraph (Doctor State Schema).
Lưu trữ toàn bộ dữ liệu ảnh, chỉ số kỹ thuật, lịch sử xử lý và kế hoạch điều trị.
"""

from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from pydantic import BaseModel, Field


class RegionOperation(BaseModel):
    """Chi tiết từng thao tác xử lý trên một vùng cụ thể."""

    region_id: str = Field(..., description="Tên vùng, ví dụ: 'sky', 'face', 'full'")
    target_prompt: str = Field(..., description="Từ khóa nhận diện vùng hoặc mô tả không gian")
    region_type: str = Field(
        default="full",
        description="Loại vùng: 'semantic' (GroundingDINO+SAM), 'face' (MediaPipe), 'spatial' (quadrant), 'full' (toàn ảnh)",
    )
    detected_issue: str = Field(
        ..., description="Vấn đề kỹ thuật: underexposed, noise, low_contrast, etc."
    )
    operation: str = Field(
        ...,
        description="Tên thao tác trong toolbox: denoise, gamma_correct, clahe, sharpen, color_correct",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Các siêu tham số cụ thể")
    order: int = Field(default=0, description="Thứ tự ưu tiên xử lý")


class TreatmentPlan(BaseModel):
    """Kế hoạch xử lý tổng thể do VLM đưa ra cho một vòng lặp."""

    iteration: int = 1
    reasoning: str = Field(..., description="Lý do chuyên môn từ mô hình VLM")
    actions: List[RegionOperation] = Field(
        default_factory=list, description="Danh sách các thao tác thực thi"
    )


class HistoryItem(BaseModel):
    """Bản ghi lịch sử sau mỗi vòng lặp."""

    iteration: int
    plan: TreatmentPlan
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    eval_score: Dict[str, Any]
    decision: str


class DoctorState(TypedDict):
    """Kiểu dữ liệu trạng thái được truyền qua lại giữa các Node trong LangGraph."""

    original_image: np.ndarray
    current_image: np.ndarray
    ground_truth_image: Optional[np.ndarray]
    is_synthetic: bool
    iteration: int
    max_iterations: int
    technical_metrics: Dict[str, Any]
    treatment_plan: Optional[TreatmentPlan]
    evaluation_result: Dict[str, Any]
    history: List[HistoryItem]
    decision: str  # "SHIP", "RE_PROCESS", "STOP_BEST_EFFORT"
    error_message: Optional[str]
