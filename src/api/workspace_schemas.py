"""
Mô hình Request & Response (Pydantic v2) của API mới: /api/v1/auth/* và /api/v1/workspace/*.
Không dùng chung với schemas.py của route cũ.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Giới hạn giống Gradio (frontend/app.py); không đặt ngưỡng mới
MAX_ITERATIONS_RANGE = (1, 5)
NUM_VARIANTS_RANGE = (1, 3)
# Số chỉnh sửa chẩn đoán tối đa người dùng tích lũy trong một phiên (chat bổ sung phân tích)
MAX_DRAFT_EDITS = 20
DEFAULT_USER_SETTINGS = {"max_iterations": 3, "num_variants": 3}


class Credentials(BaseModel):
    """Tên đăng nhập + mật khẩu; kiểm tra định dạng chi tiết ở auth.py để báo lỗi tiếng Việt."""

    # max_length chỉ chặn payload quá lớn; quy tắc thật (3–32, 8–128) kiểm ở handler
    username: str = Field(..., max_length=256)
    password: str = Field(..., max_length=1024)


class UserSettings(BaseModel):
    """Cài đặt mặc định của người dùng cho phiên xử lý mới."""

    max_iterations: int = DEFAULT_USER_SETTINGS["max_iterations"]
    # Deprecated: không còn nhiều phiên bản; giữ để client cũ đọc/ghi cài đặt không lỗi
    num_variants: int = DEFAULT_USER_SETTINGS["num_variants"]


class UserSettingsUpdate(BaseModel):
    """Cập nhật một phần cài đặt; trường bỏ trống giữ nguyên."""

    max_iterations: Optional[int] = Field(
        default=None, ge=MAX_ITERATIONS_RANGE[0], le=MAX_ITERATIONS_RANGE[1]
    )
    num_variants: Optional[int] = Field(
        default=None, ge=NUM_VARIANTS_RANGE[0], le=NUM_VARIANTS_RANGE[1]
    )


class UserOut(BaseModel):
    """Thông tin người dùng trả về client (không có mật khẩu hay token)."""

    id: str
    username: str
    settings: UserSettings
    created_at: str


# ---------------- phiên xử lý ----------------
class ArtifactOut(BaseModel):
    """Một ảnh của phiên; url là endpoint có kiểm quyền (không có đường dẫn đĩa)."""

    id: str
    kind: str
    url: str
    download_url: str
    width: int
    height: int


class MessageOut(BaseModel):
    """Một lượt hội thoại; content là JSON theo kind (spec §5.4)."""

    id: int
    role: str
    kind: str
    content: Dict[str, Any]
    created_at: str


class SessionError(BaseModel):
    """Lỗi gần nhất của phiên, câu tiếng Việt để hiển thị."""

    code: str
    message: str


class SessionSummaryOut(BaseModel):
    """Một dòng trong sidebar."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    thumbnail_url: Optional[str] = None
    final_decision: Optional[str] = None
    # Kết cục trung thực của lần xử lý gần nhất (ACCEPTED, UNVERIFIED, …); None với phiên cũ
    outcome_status: Optional[str] = None


class SessionDetailOut(SessionSummaryOut):
    """Toàn bộ phiên, đọc từ CSDL (không chạy lại pipeline)."""

    options: Dict[str, Any]
    original: Optional[ArtifactOut] = None
    ground_truth: Optional[ArtifactOut] = None
    messages: List[MessageOut] = Field(default_factory=list)
    diagnosis: Optional[Dict[str, Any]] = None
    questions: Optional[List[Dict[str, Any]]] = None
    process: Optional[Dict[str, Any]] = None
    variants: List[Dict[str, Any]] = Field(default_factory=list)
    treatments: List[Dict[str, Any]] = Field(default_factory=list)
    active_treatment_id: Optional[str] = None
    ratings: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[SessionError] = None
    artifacts: List[ArtifactOut] = Field(default_factory=list)
    # Chat bổ sung phân tích: edit đã lưu, chẩn đoán nháp (chỉ khi chờ trả lời) và overlay vùng
    draft_edits: List[Dict[str, Any]] = Field(default_factory=list)
    draft_diagnosis: Optional[Dict[str, Any]] = None
    region_overlay_url: Optional[str] = None
    regions: List[Dict[str, Any]] = Field(default_factory=list)


class RenameRequest(BaseModel):
    """Đổi tên phiên."""

    title: str = Field(..., max_length=1000)


class SampleOut(BaseModel):
    """Một ảnh mẫu có trên máy chủ."""

    id: str
    label: str
    thumbnail_url: str


class AnswersIn(BaseModel):
    """Câu trả lời cho các câu hỏi ý định: id câu hỏi → value của lựa chọn, và ghi chú tự do."""

    answers: Dict[str, str] = Field(default_factory=dict, max_length=10)
    notes: str = Field(default="", max_length=1000)


class AnalysisNoteIn(BaseModel):
    """Một tin nhắn bổ sung/sửa phân tích bằng tiếng Việt, gửi trước khi bắt đầu xử lý."""

    message: str = Field(..., max_length=1000)


class RefineIn(BaseModel):
    """Góp ý tiếng Việt cho kết quả; treatment_id bỏ trống → áp vào phiên bản đang dùng."""

    feedback: str = Field(..., max_length=2000)
    treatment_id: Optional[str] = Field(default=None, max_length=64)


class RatingIn(BaseModel):
    """Đánh giá một kết quả: run (thường là run process) + phiên bản phác đồ đang xem."""

    run_id: str = Field(..., max_length=64)
    treatment_id: Optional[str] = Field(default=None, max_length=64)
    rating: Literal["good", "bad"]
    note: str = Field(default="", max_length=2000)


class RatingOut(BaseModel):
    """Đánh giá đã lưu."""

    id: str
    run_id: str
    treatment_id: Optional[str] = None
    rating: str
    note: str
    created_at: str
    updated_at: str


class SystemOut(BaseModel):
    """Trạng thái máy chủ và chế độ chạy (theo cấu hình, không phải bằng chứng đã gọi Gemini)."""

    backend: str
    gemini_key_configured: bool
    gemini_model: str
    memory_enabled: bool
    pipeline_concurrency: int
    registration_open: bool
    max_upload_mb: int
