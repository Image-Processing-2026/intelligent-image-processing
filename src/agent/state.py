"""
Trạng thái làm việc của LangGraph (Doctor State Schema).
Lưu trữ toàn bộ dữ liệu ảnh, chỉ số kỹ thuật, lịch sử xử lý và kế hoạch điều trị.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
from pydantic import BaseModel, Field


class RegionOperation(BaseModel):
    """Chi tiết từng thao tác xử lý trên một vùng cụ thể."""

    region_id: str = Field(..., description="Tên vùng, ví dụ: 'sky', 'face', 'full'")
    target_prompt: str = Field(..., description="Từ khóa nhận diện vùng hoặc mô tả không gian")
    region_type: Literal["semantic", "face", "spatial", "full", "bbox", "binary_mask"] = Field(
        default="semantic",
        description="Bộ phân giải vùng được dùng bởi Module 2",
    )
    bbox: Optional[tuple[int, int, int, int]] = Field(
        default=None,
        description="Half-open pixel bbox (xmin, ymin, xmax, ymax) for bbox regions",
    )
    quadrant: Optional[str] = Field(
        default=None,
        description="top, bottom, left, right, or center for spatial regions",
    )
    binary_mask: Optional[Any] = Field(
        default=None,
        description="In-process bool/uint8 binary mask for binary_mask regions",
    )
    feather_radius: int = Field(default=15, ge=0)
    expand_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    merge_policy: Literal["max"] = Field(default="max")
    face_mode: Literal["bbox", "oval", "sam_refined"] = Field(
        default="bbox",
        description="Face region mode. bbox is backward-compatible; oval requires Face Landmarker.",
    )
    num_faces: int = Field(default=4, ge=1)
    instance_selection: Literal["all", "largest", "index"] = Field(default="all")
    instance_index: Optional[int] = Field(default=None, ge=0)
    box_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
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
