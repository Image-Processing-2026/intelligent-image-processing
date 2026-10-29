[English](README.md) | [Tiếng Việt](README.vi.md)

# AI Image Doctor

Đồ án xử lý ảnh khép vòng cho môn Xử lý ảnh. Gemini và LangGraph chẩn đoán, điều phối; OpenCV, scikit-image và NumPy thực hiện biến đổi pixel bằng thuật toán cổ điển. Sinh ảnh bằng AI nằm ngoài phạm vi dự án.

## Chạy nhanh

Web app cần Python 3.10+, Node.js 20+ và npm. Chạy từ thư mục gốc repo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
npm --prefix web ci
npm --prefix web run build
uvicorn src.api.main:app --port 8000
```

Mở http://localhost:8000, đăng ký tài khoản cục bộ và tải ảnh lên. Sao chép `.env.example` thành `.env` để cấu hình Gemini; để trống `GEMINI_API_KEY` nếu dùng rule-based. Có key vẫn có thể fallback khi gọi thất bại. Xem nguồn thực tế của từng chẩn đoán/phác đồ.

Khi phát triển, chạy backend với `--reload` và `npm --prefix web run dev` ở terminal khác; mở http://localhost:5173. Gradio là demo cũ riêng: `python frontend/app.py`.

## Tài liệu

Bắt đầu ở [mục lục](docs/README.vi.md). Đọc [kiến trúc](docs/architecture.vi.md), [thuật toán](docs/algorithms.vi.md), [đánh giá](docs/evaluation.vi.md), [hướng dẫn sử dụng](docs/user-guide.vi.md) và [API](docs/api.vi.md).

Xem [cài đặt web](web/README.vi.md), [contract](docs/interfaces.vi.md), [đóng góp](CONTRIBUTING.md) và [quyết định](docs/decisions.md). Kế hoạch/báo cáo lịch sử có mục riêng trong mục lục.

## Dữ liệu và kiểm tra

`data/app/` chứa tài khoản, phiên và ảnh cục bộ. `data/memory/cases.sqlite` lưu bộ nhớ ca; `CASE_MEMORY_PATH=""` tắt bộ nhớ. Bản clone có thể thiếu ảnh mẫu; xem [chuẩn bị dữ liệu](data/README.md).

```powershell
ruff check src/ tests/
pytest tests/unit/ -v
npm --prefix web run typecheck
npm --prefix web test -- --run
```

Server còn route cũ ngoài ranh giới xác thực workspace. Dùng server cục bộ để demo; xem [giới hạn API](docs/api.vi.md) trước khi đưa ra mạng.
