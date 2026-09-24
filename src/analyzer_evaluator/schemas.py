"""
Mô hình dữ liệu Pydantic cho Module 1: Image Analyzer & Evaluator.
Định nghĩa cấu trúc TechnicalMetrics và EvaluationResult đồng bộ với docs/interfaces.md.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TechnicalMetrics(BaseModel):
    """
    Bản phân tích định lượng toàn diện về đặc tính vật lý và kỹ thuật của bức ảnh.
    Được sử dụng làm đầu vào cho VLM (Module 4) để chẩn đoán vấn đề.
    """

    model_config = ConfigDict(extra="allow")

    brightness_mean: float = Field(
        ..., ge=0.0, le=255.0, description="Độ sáng trung bình mức xám [0, 255]"
    )
    brightness_level: Literal["underexposed", "normal", "overexposed"] = Field(
        ..., description="Phân loại phơi sáng: underexposed (<70), normal, overexposed (>185)"
    )
    contrast_std: float = Field(
        ..., ge=0.0, description="Độ lệch chuẩn mức xám thể hiện độ tương phản động"
    )
    contrast_level: Literal["low", "normal", "high"] = Field(
        ..., description="Mức độ tương phản: low (<40), normal (40-80), high (>80)"
    )
    noise_variance: float = Field(..., ge=0.0, description="Ước lượng phương sai nhiễu tần số cao")
    noise_level: Literal["clean", "low", "medium", "severe"] = Field(
        ..., description="Mức độ nhiễu cảm nhận: clean (<3), low (<8), medium (<15), severe (>=15)"
    )
    sharpness_laplacian_var: float = Field(
        ..., ge=0.0, description="Phương sai toán tử vi phân bậc hai Laplacian"
    )
    blur_level: Literal["sharp", "mild_blur", "severe_blur"] = Field(
        ..., description="Mức độ mờ nét: severe_blur (<100), mild_blur (100-300), sharp (>=300)"
    )
    color_cast: Optional[str] = Field(
        default="none",
        description="Khuynh hướng ám màu chủ đạo: 'warm', 'cool', 'greenish', 'none'",
    )
    histogram_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Thống kê phân bố: min, max, skewness, clipping, RGB/HSV stats",
    )


class EvaluationResult(BaseModel):
    """
    Kết quả kiểm định chất lượng sau khi áp dụng thuật toán xử lý ảnh.
    Hỗ trợ cả Full-Reference (PSNR, SSIM, MSE trên synthetic)
    và No-Reference (BRISQUE, NIQE, Delta Metrics trên ảnh thực tế).
    """

    model_config = ConfigDict(extra="allow")

    iteration: int = Field(default=1, ge=1, le=10, description="Vòng lặp hiện tại (1..3)")
    is_reference_eval: bool = Field(
        default=False, description="True nếu là đánh giá có ảnh gốc ground-truth mẫu"
    )
    psnr: Optional[float] = Field(
        default=None,
        description="Peak Signal-to-Noise Ratio (dB) - chỉ áp dụng khi có ground truth",
    )
    ssim: Optional[float] = Field(
        default=None,
        description="Structural Similarity Index [0, 1] - chỉ áp dụng khi có ground truth",
    )
    mse: Optional[float] = Field(
        default=None, description="Mean Squared Error - chỉ áp dụng khi có ground truth"
    )
    brisque_score: Optional[float] = Field(
        default=None, description="Điểm BRISQUE [0, 100], càng thấp càng tự nhiên"
    )
    niqe_score: Optional[float] = Field(
        default=None, description="Điểm NIQE tự nhiên, càng thấp càng tốt"
    )
    estimated_quality_score: Optional[float] = Field(
        default=None,
        description="Điểm chất lượng cảm nhận tổng hợp theo Heuristic [0, 100], càng cao càng tốt",
    )
    technical_metrics: TechnicalMetrics = Field(
        ..., description="Chỉ số kỹ thuật của bức ảnh sau vòng xử lý hiện tại"
    )
    delta_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Chênh lệch chỉ số kỹ thuật so với vòng lặp trước đó",
    )
    visual_plausibility_passed: bool = Field(
        default=True, description="Đánh giá trực quan không phát hiện artifact nghiêm trọng"
    )
    quality_improved: bool = Field(
        default=True, description="Chỉ số tổng hợp có cải thiện so với ảnh đầu vào"
    )
    vlm_feedback: str = Field(default="", description="Nhận xét hoặc giải thích lý do đánh giá")
    decision: Literal["SHIP", "RE_PROCESS", "STOP_BEST_EFFORT"] = Field(
        default="SHIP", description="Quyết định rẽ nhánh chu trình"
    )
