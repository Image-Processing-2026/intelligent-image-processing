"""
Mô hình dữ liệu Request & Response cho REST API (API Schemas).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agent.intent import Question
from src.agent.refine import Adjustment
from src.agent.state import DiagnosisReport, IntentProfile, RegionOperation, TreatmentPlan


class DiagnoseRequest(BaseModel):
    """Yêu cầu chỉ phân tích và chẩn đoán không qua xử lý."""

    image_base64: str = Field(..., description="Ảnh đầu vào dạng base64")


class DiagnoseResponse(BaseModel):
    """Kết quả chẩn đoán và đề xuất điều trị."""

    technical_metrics: Dict[str, Any]
    diagnosis: Optional[DiagnosisReport] = None
    treatment_plan: TreatmentPlan


class VariantOut(BaseModel):
    """Một phiên bản kết quả (phong cách) để người dùng chọn."""

    id: str
    label: str
    description: str = ""
    rank: Optional[int] = None
    quality_score: Optional[float] = None
    quality_improved: Optional[bool] = None
    critic_note: str = ""
    actions: List[RegionOperation] = Field(
        default_factory=list, description="Gửi lại /render để xuất phiên bản này ở full-res"
    )
    preview_base64: str = Field(..., description="Ảnh preview (cạnh dài ≤ 1024) dạng PNG")


class RenderResponse(BaseModel):
    """Ảnh full-res của một phiên bản đã chọn."""

    image_base64: str
    width: int
    height: int
    applied_actions: List[RegionOperation]


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
    variants: List[VariantOut] = Field(
        default_factory=list, description="Các phiên bản đã xếp hạng (khi num_variants > 1)"
    )
    recommended_variant: Optional[str] = None
    variant_ranking_source: Optional[str] = Field(
        default=None, description="'critic' (VLM xếp hạng) hoặc 'score' (điểm Module 1)"
    )
    treatment: List[RegionOperation] = Field(
        default_factory=list,
        description="Phác đồ của ảnh kết quả (các vòng không rollback); gửi /refine để chỉnh",
    )
    intent: Optional[IntentProfile] = None
    recommended_reason: Optional[str] = Field(
        default=None, description="'intent', 'preference' (bộ nhớ), 'critic' hoặc 'score'"
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Id ca trong bộ nhớ; gửi kèm /render và /refine để ghi lựa chọn và góp ý",
    )


class SessionResponse(BaseModel):
    """Một bước của phiên tương tác: cần trả lời câu hỏi, hoặc đã có kết quả."""

    session_id: str
    status: str = Field(..., description="'needs_input' hoặc 'done'")
    questions: List[Question] = Field(default_factory=list)
    diagnosis: Optional[DiagnosisReport] = None
    result: Optional[ProcessResponse] = None


class AnswerRequest(BaseModel):
    """Câu trả lời cho các câu hỏi của phiên (id câu hỏi → value của lựa chọn) và ghi chú."""

    answers: Dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class RefineResponse(BaseModel):
    """Ảnh và phác đồ sau khi chỉnh theo góp ý."""

    image_base64: str
    width: int
    height: int
    actions: List[RegionOperation]
    adjustments: List[Adjustment]
    notes: List[str] = Field(default_factory=list, description="Điều chỉnh đã làm, tiếng Việt")
    source: str = Field(..., description="'rules', 'vlm' hoặc 'none' (không hiểu góp ý)")
    quality_score: Optional[float] = None


class MemoryStats(BaseModel):
    """Thống kê bộ nhớ ca bệnh."""

    enabled: bool
    cases: int = 0
    choices: Dict[str, int] = Field(default_factory=dict)
    feedback: Dict[str, int] = Field(default_factory=dict)
    preferred_style: Optional[str] = None
