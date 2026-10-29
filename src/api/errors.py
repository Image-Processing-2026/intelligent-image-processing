"""
Định dạng lỗi của các router mới: {"error": {"code", "message"}} với câu tiếng Việt.
Chỉ áp cho router dùng ApiRoute; route cũ giữ định dạng lỗi mặc định của FastAPI (P5).
Không bao giờ đưa traceback, chuỗi exception thô hay đường dẫn đĩa vào response.
"""

import logging
from typing import Any, Callable, Coroutine, Dict, Optional

from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)

MESSAGES: Dict[str, str] = {
    "unauthenticated": "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.",
    "invalid_credentials": "Tên đăng nhập hoặc mật khẩu không đúng.",
    "registration_closed": "Máy chủ đang tắt đăng ký tài khoản mới.",
    "not_found": "Không tìm thấy phiên xử lý hoặc tệp yêu cầu.",
    "username_taken": "Tên đăng nhập đã được dùng.",
    "session_busy": "Phiên này đang được xử lý. Vui lòng đợi.",
    "invalid_state": "Thao tác không phù hợp với trạng thái hiện tại của phiên.",
    "thread_expired": (
        "Phiên hỏi ý đã hết hạn (máy chủ khởi động lại hoặc quá nhiều phiên đang chờ). "
        "Hãy hỏi ý lại."
    ),
    "file_too_large": "Tệp vượt quá giới hạn dung lượng.",
    "unsupported_format": "Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP.",
    "invalid_image": "Không đọc được ảnh, hoặc ảnh quá lớn.",
    "image_too_small": "Ảnh quá nhỏ để xử lý.",
    "ground_truth_mismatch": "Ảnh chuẩn phải cùng kích thước với ảnh gốc.",
    "validation": "Dữ liệu gửi lên không hợp lệ.",
    "pipeline_failed": "Xử lý ảnh không thành công. Bạn có thể thử lại.",
    "internal": "Có lỗi phía máy chủ. Vui lòng thử lại.",
    "server_restarted": "Máy chủ đã khởi động lại khi đang xử lý. Hãy thử lại.",
    "interrupted": "Lần xử lý bị gián đoạn nên không có kết quả. Hãy thử lại.",
}

# HTTPException (ví dụ từ Starlette) không mang code riêng → suy từ mã HTTP
STATUS_CODES: Dict[int, str] = {
    401: "unauthenticated",
    404: "not_found",
    405: "invalid_state",
    409: "invalid_state",
    413: "file_too_large",
    415: "unsupported_format",
    422: "validation",
}


class ApiError(Exception):
    """Lỗi có chủ đích của API mới: mã HTTP, code ổn định và câu tiếng Việt cho UI."""

    def __init__(
        self,
        status: int,
        code: str,
        message: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message or MESSAGES.get(code, MESSAGES["internal"])
        self.fields = fields


def error_response(
    status: int,
    code: str,
    message: Optional[str] = None,
    fields: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """Response lỗi theo envelope chung."""
    body: Dict[str, Any] = {
        "code": code,
        "message": message or MESSAGES.get(code, MESSAGES["internal"]),
    }
    if fields:
        body["fields"] = fields
    return JSONResponse(status_code=status, content={"error": body}, headers=headers)


def _validation_fields(exc: RequestValidationError) -> Dict[str, str]:
    """Tên trường lỗi → câu ngắn tiếng Việt (không lặp lại giá trị người dùng gửi)."""
    fields: Dict[str, str] = {}
    for item in exc.errors():
        location = [str(part) for part in item.get("loc", ()) if part not in ("body", "query")]
        name = ".".join(location) or "request"
        fields.setdefault(name, "Giá trị không hợp lệ.")
    return fields


class ApiRoute(APIRoute):
    """APIRoute bọc mọi lỗi của handler (kể cả lỗi validate request) thành envelope."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def wrapped(request: Request) -> Response:
            try:
                return await handler(request)
            except ApiError as exc:
                return error_response(exc.status, exc.code, exc.message, exc.fields)
            except RequestValidationError as exc:
                return error_response(422, "validation", fields=_validation_fields(exc))
            except HTTPException as exc:
                code = STATUS_CODES.get(exc.status_code, "internal")
                return error_response(exc.status_code, code)
            except Exception:
                logger.exception("Unhandled error on %s %s", request.method, request.url.path)
                return error_response(500, "internal")

        return wrapped
