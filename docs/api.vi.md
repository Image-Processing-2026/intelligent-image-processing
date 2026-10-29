[English](api.md) | [Tiếng Việt](api.vi.md)

# Tích hợp API

Mốc source: `02dd6f9`. Chạy `uvicorn src.api.main:app --port 8000`. Schema tương tác ở http://localhost:8000/docs, OpenAPI ở http://localhost:8000/openapi.json. Xem schema cho đầy đủ trường/query export.

## Xác thực

`POST /api/v1/auth/register` và `/login` nhận JSON `username`, `password`; trả thông tin người dùng và cookie phiên HTTP-only. Giữ cookie giữa request. `/me` đọc người dùng; `/logout` kết thúc đăng nhập. Credential Gemini ở server.

PowerShell dùng `curl.exe` thay alias curl. Đăng ký qua UI trước. Tạo `scratch/login.json` chứa `username`, `password` cục bộ; không commit file này/cookie. Thay đường dẫn ảnh bằng file có thật:

```powershell
curl.exe -c scratch/cookies.txt -H 'Content-Type: application/json' --data-binary '@scratch/login.json' http://localhost:8000/api/v1/auth/login
curl.exe -b scratch/cookies.txt -F 'file=@C:/images/input.jpg' -F 'title=Demo' http://localhost:8000/api/v1/workspace/sessions
curl.exe -b scratch/cookies.txt http://localhost:8000/api/v1/workspace/sessions
```

Tạo phiên trả detail gồm `id`; list trả session summary. Xem OpenAPI cho ví dụ response đầy đủ, không giả định detail chỉ có ID.

## Vòng đời workspace

| Endpoint dưới `/api/v1/workspace` | Request/mục đích |
|---|---|
| `POST /sessions` | Multipart: đúng một file hoặc sample; tùy chọn title, user_prompt, ground_truth, max_iterations |
| `GET /sessions/{id}` | Đọc detail phiên sở hữu |
| `POST /sessions/{id}/chat` | JSON message và revision hiện tại; chuẩn bị phương án |
| `POST /sessions/{id}/execute` | JSON revision phương án và idempotency_key; render phương án hiển thị |
| `GET /sessions/{id}/export` | Export theo query OpenAPI |
| `DELETE /sessions/{id}` | Xóa phiên/artifact |

Lấy `conversation.revision` từ response mới nhất. Execute revision cũ trả 409. Lặp execute thành công cùng idempotency key không nhân đôi treatment. Xem [API test hội thoại](../tests/unit/test_api_conversation.py). OpenAPI còn mô tả hỏi ý, ghi chú, tinh chỉnh, rating.

## Lỗi, cấu hình và giới hạn

Route mới dùng `{"error":{"code":"…","message":"…"}}`, tùy chọn lỗi theo trường. Message hiện bằng tiếng Việt. Code thường có unauthenticated, validation, session_busy, thread_expired. Status gồm 401 xác thực, 404 thiếu tài nguyên, 409 sai trạng thái, 413 quá dung lượng, 415 sai định dạng, 422 sai trường. Route cũ có thể dùng lỗi FastAPI cũ.

Mặc định trong [config](../src/api/config.py): upload 20 MB (`MAX_UPLOAD_MB`), 40,000,000 pixel (`MAX_IMAGE_PIXELS`), cạnh ngắn 16 (`MIN_IMAGE_SIDE`), một pipeline đồng thời (`PIPELINE_CONCURRENCY`), token 30 ngày (`AUTH_TOKEN_TTL_DAYS`). Upload xoay EXIF. Ground truth khớp kích thước sau giải mã.

`APP_DATA_DIR` mặc định `data/app` lưu SQLite/ảnh. `CORS_ORIGINS` bật client khác origin rõ ràng; cùng origin không cần CORS. `COOKIE_SECURE` cho HTTPS; `ALLOW_REGISTRATION` kiểm soát đăng ký. Xem source cho parsing/kẹp miền.

Route cũ `/api/v1/diagnose`, `/process`, `/render`, `/refine`, `/sessions`, `/memory` tồn tại ngoài xác thực workspace. Rà trước khi public. Request đồng bộ; không cam kết hủy/tiến độ phần trăm trực tiếp. Ví dụ đã đối chiếu source; tình trạng chạy mạng ghi ở [kiểm chứng](documentation-verification.vi.md).
