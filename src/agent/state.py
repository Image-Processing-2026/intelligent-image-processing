"""
Trạng thái làm việc của LangGraph (Doctor State Schema).
Lưu trữ toàn bộ dữ liệu ảnh, chỉ số kỹ thuật, lịch sử xử lý và kế hoạch điều trị.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
from pydantic import BaseModel, Field

PlanSource = Literal["vlm", "rule_based", "vlm_fallback"]

# Từ vựng chẩn đoán (giai đoạn Perceive). Chỉ gồm lỗi mà toolbox 5 công cụ xử lý được
# hoặc cần nhận diện để KHÔNG xử lý sai.
DEFECT_TYPES: tuple[str, ...] = (
    "underexposed",
    "overexposed",
    "backlit_subject",
    "low_contrast",
    "noise",
    "blur",
    "color_cast_warm",
    "color_cast_cool",
    "color_cast_green",
    "oversaturated",
    "undersaturated",
)
SCENE_TYPES: tuple[str, ...] = (
    "portrait",
    "group",
    "landscape",
    "cityscape",
    "night",
    "indoor",
    "food",
    "product",
    "document",
    "animal",
    "macro",
    "other",
)
# Đặc điểm thẩm mỹ có chủ ý; planner.apply_preserve_guard chặn thao tác phá vỡ chúng
PRESERVE_ASPECTS: tuple[str, ...] = (
    "warm_tone",
    "cool_tone",
    "low_key",
    "high_key",
    "silhouette",
    "film_grain",
    "soft_focus",
    "muted_colors",
    "vivid_colors",
)


class Defect(BaseModel):
    """Một lỗi kỹ thuật được chẩn đoán trên một vùng."""

    type: Literal[DEFECT_TYPES]  # type: ignore[valid-type]
    region: str = Field(default="full", description="'full' hoặc target_prompt của một vùng")
    severity: int = Field(default=1, ge=0, le=3, description="0 không đáng kể … 3 nặng")
    evidence: str = Field(default="", description="Bằng chứng ngắn gọn")
    # vlm: VLM chẩn đoán; rule: luật từ metric toàn cục; measured: suy ra từ số đo theo vùng
    origin: Literal["vlm", "rule", "measured"] = "vlm"


class PreserveItem(BaseModel):
    """Một đặc điểm thẩm mỹ có chủ ý cần giữ nguyên."""

    aspect: Literal[PRESERVE_ASPECTS]  # type: ignore[valid-type]
    region: str = Field(default="full", description="'full' hoặc target_prompt của một vùng")
    reason: str = ""


class RegionMetrics(BaseModel):
    """Số đo độ sáng của một vùng, tính có trọng số theo mặt nạ mềm của Module 2."""

    region: str
    backend: str = Field(default="unknown", description="Bộ phân giải vùng đã tạo mask")
    area_ratio: float = Field(..., description="Tỉ lệ diện tích vùng / ảnh (theo trọng số mask)")
    brightness_mean: float
    brightness_std: float
    brightness_level: Literal["underexposed", "normal", "overexposed"]
    highlight_clip_ratio: float
    shadow_clip_ratio: float
    brightness_vs_rest: Optional[float] = Field(
        default=None, description="Độ sáng vùng trừ độ sáng phần còn lại; None nếu vùng phủ gần hết"
    )


class DiagnosisReport(BaseModel):
    """Kết quả giai đoạn Perceive: hiểu cảnh, lỗi theo vùng và điều cần giữ."""

    iteration: int = 1
    scene_type: Literal[SCENE_TYPES] = "other"  # type: ignore[valid-type]
    lighting: str = ""
    subjects: List[str] = Field(default_factory=list, description="Vùng quan trọng cần đo riêng")
    defects: List[Defect] = Field(default_factory=list)
    preserve: List[PreserveItem] = Field(default_factory=list)
    summary: str = ""
    region_metrics: Dict[str, RegionMetrics] = Field(default_factory=dict)
    source: PlanSource = "vlm"

    @property
    def actionable_defects(self) -> List[Defect]:
        """Các lỗi có severity >= 1 (cần lập kế hoạch xử lý)."""
        return [defect for defect in self.defects if defect.severity >= 1]


class RegionOperation(BaseModel):
    """Chi tiết từng thao tác xử lý trên một vùng cụ thể."""

    region_id: str = Field(..., description="Tên vùng, ví dụ: 'sky', 'face', 'full'")
    target_prompt: str = Field(..., description="Từ khóa nhận diện vùng hoặc mô tả không gian")
    # None → executor suy ra loại vùng từ target_prompt (face/full/quadrant/semantic)
    region_type: Optional[Literal["semantic", "face", "spatial", "full", "bbox", "binary_mask"]] = (
        Field(
            default=None,
            description="Bộ phân giải vùng của Module 2; None thì suy ra từ target_prompt",
        )
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
    # Không đặt ge/le ở đây: một giá trị vượt biên từ VLM sẽ làm hỏng cả kế hoạch.
    # Biên được kẹp trong planner.REGION_FIELD_BOUNDS (giống PARAMETER_BOUNDS).
    feather_radius: int = Field(default=15)
    expand_ratio: float = Field(default=0.15)
    merge_policy: Literal["max"] = Field(default="max")
    face_mode: Literal["bbox", "oval", "sam_refined"] = Field(
        default="bbox",
        description="Face region mode. bbox is backward-compatible; oval requires Face Landmarker.",
    )
    num_faces: int = Field(default=4)
    instance_selection: Literal["all", "largest", "index"] = Field(default="all")
    instance_index: Optional[int] = Field(default=None)
    box_threshold: float = Field(default=0.35)
    text_threshold: float = Field(default=0.25)
    nms_iou_threshold: float = Field(default=0.8)
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
    # vlm: do Gemini lập; rule_based: không có API key; vlm_fallback: gọi Gemini lỗi → luật dự phòng
    knowledge: List[str] = Field(
        default_factory=list, description="Id playbook card/nguyên lý đã dùng để lập kế hoạch"
    )
    source: PlanSource = Field(default="vlm", description="Nguồn gốc kế hoạch điều trị")


class HistoryItem(BaseModel):
    """Bản ghi lịch sử sau mỗi vòng lặp."""

    iteration: int
    plan: TreatmentPlan
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    eval_score: Dict[str, Any]
    decision: str
    diagnosis: Optional[DiagnosisReport] = Field(
        default=None, description="Chẩn đoán giai đoạn Perceive của vòng này"
    )
    rolled_back: bool = Field(
        default=False, description="True nếu vòng này suy thoái và đã rollback về ảnh trước đó"
    )


class Variant(BaseModel):
    """Một phiên bản kết quả (phong cách) để người dùng chọn, render trên ảnh preview."""

    id: str = Field(..., description="Mã phong cách: balanced, natural, vivid")
    label: str
    description: str = ""
    actions: List[RegionOperation] = Field(
        default_factory=list, description="Phác đồ đã áp phong cách; dùng để render full-res"
    )
    # Ảnh preview (np.ndarray) chỉ dùng trong tiến trình; không đưa vào model_dump/JSON
    image: Optional[Any] = Field(default=None, exclude=True)
    quality_score: Optional[float] = Field(
        default=None, description="Điểm No-Reference của Module 1 trên ảnh preview"
    )
    quality_improved: Optional[bool] = Field(
        default=None, description="Module 1: phiên bản có cải thiện so với ảnh gốc không"
    )
    rank: Optional[int] = Field(default=None, description="Thứ hạng, 1 là tốt nhất")
    critic_note: str = Field(default="", description="Nhận xét của VLM critic (nếu có)")


class DoctorState(TypedDict):
    """Kiểu dữ liệu trạng thái được truyền qua lại giữa các Node trong LangGraph."""

    original_image: np.ndarray
    current_image: np.ndarray
    previous_image: Optional[np.ndarray] = None
    ground_truth_image: Optional[np.ndarray]
    is_synthetic: bool
    iteration: int
    max_iterations: int
    technical_metrics: Dict[str, Any]
    diagnosis: Optional[DiagnosisReport]
    treatment_plan: Optional[TreatmentPlan]
    evaluation_result: Dict[str, Any]
    history: List[HistoryItem]
    intermediate_images: List[np.ndarray]  # Ảnh thumbnail sau mỗi vòng lặp
    decision: str  # "SHIP", "RE_PROCESS", "STOP_BEST_EFFORT"
    rolled_back: bool  # True nếu kết quả cuối là ảnh đã rollback do suy thoái
    error_message: Optional[str]
    num_variants: int  # > 1 → sinh các phiên bản phong cách sau khi vòng lặp kết thúc
    # Các nhánh render song song (Send) cộng dồn kết quả vào đây (chưa xếp hạng)
    variant_candidates: Annotated[List[Variant], operator.add]
    variants: List[Variant]  # Đã bỏ trùng và xếp hạng, rank 1 trước
    recommended_variant: Optional[str]
    variant_ranking_source: Optional[str]  # "critic" (VLM) hoặc "score" (Module 1)
