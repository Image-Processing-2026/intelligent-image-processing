# AI Image Doctor

Xử lý ảnh khép vòng có tác tử, làm bài cho môn Xử lý ảnh. Gemini VLM và LangGraph chẩn đoán và điều phối; mọi
thao tác trên điểm ảnh đều dùng thuật toán cổ điển (OpenCV, scikit-image, NumPy), không dùng AI tạo sinh
(ADR-001).

Tài liệu: `CONTEXT.md` (mục tiêu, phân công), `docs/interfaces.md` (hợp đồng giữa các module),
`docs/decisions.md` (ADR), `CONTRIBUTING.md` (quy trình git), `docs/frontend/` (web app).

## Cài đặt

```powershell
pip install -r requirements-dev.txt
```

Khóa Gemini đặt trong `.env` (xem `.env.example`). Không có khóa thì pipeline chạy rule-based.

## Chạy

**Web app** (React + FastAPI, có tài khoản, lưu phiên): xem [web/README.md](web/README.md). Bản
production-local gồm build một lần rồi chạy một server:

```powershell
npm --prefix web ci
npm --prefix web run build
uvicorn src.api.main:app --port 8000
```

Mở http://localhost:8000. Khi phát triển giao diện thì chạy `uvicorn src.api.main:app --reload --port 8000`
và `npm --prefix web run dev` (http://localhost:5173).

**Gradio** (demo cũ, gọi pipeline trực tiếp): `python frontend/app.py`, mở http://localhost:7860. Xem
[frontend/README.md](frontend/README.md).

## Dữ liệu

- `data/app/`: tài khoản, phiên xử lý và ảnh của web app (gitignored).
- `data/memory/cases.sqlite`: bộ nhớ ca bệnh; tắt bằng `CASE_MEMORY_PATH=""`.
- `data/real/`: ảnh thật dùng làm mẫu; ảnh không nằm trong git (xem `data/README.md`).

## Kiểm tra

```powershell
ruff check src/ tests/
pytest tests/unit/ -v
npm --prefix web run typecheck
npm --prefix web test -- --run
```

CI chạy `ruff check` và `pytest tests/unit/` trên Python 3.10. Test chạy offline, không cần khóa.
