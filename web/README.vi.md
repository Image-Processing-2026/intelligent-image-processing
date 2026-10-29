[English](README.md) | [Tiếng Việt](README.vi.md)

# Ứng dụng web

React/Vite/TypeScript dùng workspace FastAPI có xác thực. Cần Node 20+. Chạy backend/Vite ở terminal riêng, mở http://localhost:5173. Proxy dùng IID_BACKEND_URL, mặc định http://127.0.0.1:8000. Chạy một server: build bằng `npm --prefix web run build`, chạy uvicorn rồi mở http://localhost:8000. WEB_DIST_DIR mặc định web/dist.

Chạy từ repo root:

```powershell
uvicorn src.api.main:app --reload --port 8000
npm --prefix web ci
npm --prefix web run dev
```

GEMINI_API_KEY chỉ ở server; trống thì rule-based. CASE_MEMORY_PATH rỗng tắt memory. Xem [quickstart](../README.vi.md), [cấu hình API](../docs/api.vi.md), [sử dụng](../docs/user-guide.vi.md).

[README chi tiết trước](README.reference.md) là tham khảo lịch sử tiếng Việt. Timing và mô tả UI nhiều phiên bản cũ thuộc snapshot đó.
