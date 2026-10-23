# Frontend decision log — Web app "AI Image Doctor"

Nhãn: `confirmed` (chủ dự án / tài liệu), `evidence-backed`, `assumed`, `proposed`, `unresolved`.

## Đã chốt bởi chủ dự án (prompt nhiệm vụ 2026-10-22) — confirmed

| # | Quyết định |
|---|---|
| D1 | Stack: React + Vite + TypeScript + Tailwind + shadcn/ui trong `web/`. Dev: Vite proxy `/api` → FastAPI (cùng origin, cookie). Production-local: `vite build`, FastAPI phục vụ `web/dist` (StaticFiles + fallback `index.html`), một lệnh uvicorn. |
| D2 | Gradio (`frontend/`) giữ nguyên; gỡ là PR riêng. |
| D3 | Không job nền / polling / SSE / hủy. Request đồng bộ chờ pipeline xong. Spinner không phần trăm + "Đang xử lý, thời gian phụ thuộc ảnh và số vòng". Không timeout ngắn ở fetch/proxy. Semaphore giới hạn số lần chạy pipeline đồng thời (mặc định 1). Khóa nút khi chạy, mở lại cả khi lỗi. |
| D4 | SQLite (WAL) + SQLAlchemy 2.0, `create_all` khi khởi động, chưa Alembic (thêm khi schema ổn định). DB tại `data/app/app.sqlite`. |
| D5 | Ảnh trên đĩa `data/app/users/<user_id>/sessions/<session_id>/…`; DB chỉ giữ đường dẫn + metadata. Ảnh trả qua GET artifact có kiểm quyền (FileResponse), không base64 trong API mới. `data/app/` vào `.gitignore`. |
| D6 | Giữ vô thời hạn; xóa thủ công (xóa phiên = xóa bản ghi + file). |
| D7 | Auth username + mật khẩu, argon2-cffi; token ngẫu nhiên trong cookie httpOnly SameSite=Lax, chỉ lưu SHA-256 của token trong `auth_tokens`, có hạn. Mọi endpoint mới yêu cầu đăng nhập; phiên của người khác → 404. Không tin `user_id` từ client. CORS không `*` kèm credentials. |
| D8 | `str(user.id)` làm `user_id` cho `run_pipeline` / `start_session` / `remember_run`. CaseMemory giữ nguyên. |
| D9 | Thuật ngữ: "Phiên xử lý" = `processing_sessions`; token đăng nhập = `auth_tokens`; phiên LangGraph = `thread`. |
| D10 | Thread đang chờ trả lời mất khi restart là chấp nhận được; UI hiện "Phiên hỏi ý đã hết hạn" và cho hỏi ý lại. Chỉ ghi kết quả hoàn tất vào DB. Không đổi MemorySaver. |
| D11 | Dependency mới được phép: python-multipart, sqlalchemy, argon2-cffi. numpy<2, không CUDA, Python CI 3.10. |

## Quyết định của agent trong spec — confirmed (chủ dự án duyệt 2026-10-22)

| # | Quyết định | Lý do |
|---|---|---|
| P1 | `users.id` là UUID hex (chuỗi 32 ký tự), không phải số tự tăng. | `str(user.id)` là khóa của CaseMemory (DB riêng). Id tự tăng sẽ được tái dùng nếu xóa `app.sqlite`, khiến người dùng mới thừa hưởng bộ nhớ ca của người cũ. |
| P2 | Chế độ nâng cao có ảnh chuẩn (ground truth): vẫn hỏi ý bằng `start_session`, nhưng khi trả lời thì backend tính `intent_from_answers(questions, answers, notes)` và gọi `run_pipeline(..., ground_truth, is_synthetic=True, intent=intent)` thay vì `answer_session`, rồi `end_session(thread)`. | `start_session` không nhận ground truth (`session.py:80-86`) và không được sửa agent. Hai hàm trên đều là API công khai; ý định suy ra tất định từ cùng câu hỏi. Hệ quả: chạy Perceive vòng 1 hai lần (thêm một lời gọi Gemini khi có key). |
| P3 | Endpoint mới là `def` đồng bộ (FastAPI tự chạy trong threadpool) thay vì `async` + `run_in_threadpool`; semaphore là `threading.BoundedSemaphore(PIPELINE_CONCURRENCY)` bọc mọi lời gọi pipeline (start/answer/run/render/refine). Request chờ semaphore (xếp hàng), không trả 429. | Tương đương về hành vi, ít code hơn; I/O DB và đĩa cũng không chặn event loop. |
| P4 | Một phiên xử lý có tối đa **một** lần chạy pipeline hoàn tất; muốn thử tùy chọn khác thì tạo phiên mới. Tinh chỉnh và render không giới hạn. | Giữ mô hình dữ liệu và UI đơn giản; "nhân bản phiên" đã ngoài phạm vi. |
| P5 | Bọc mọi lỗi của router mới thành `{"error": {"code", "message"}}` (message tiếng Việt) bằng `route_class` riêng cho các router mới, KHÔNG dùng exception handler toàn app. | Exception handler toàn app sẽ đổi định dạng lỗi 422/500 của route cũ (vi phạm "không đổi hợp đồng"). |
| P6 | Ảnh gốc được chuẩn hóa khi upload (EXIF transpose → RGB uint8) và lưu lại dạng PNG `original.png`; pipeline, render, refine đều đọc file này. | Bảo đảm render/refine dùng đúng ảnh pipeline đã thấy. Route cũ không xoay EXIF (`routes.py:96`) nên ảnh xoay có thể khác giữa Gradio và web mới — ghi là hạn chế đã biết, không sửa route cũ. |
| P7 | Thread hết hạn không chỉ do restart mà còn do `MAX_SESSIONS = 32` (`session.py:27,59-62`). Xử lý chung: answer trả 409 `thread_expired`, UI hiện thông báo + nút "Hỏi ý lại" (gọi lại `start_session` trên ảnh đã lưu). | Theo D10. |
| P8 | Trạng thái `running` được ghi trước khi chạy; khi khởi động server, mọi phiên `running` → `failed` (kèm message "Máy chủ đã khởi động lại khi đang xử lý"). | Request đồng bộ có thể chết giữa chừng khi restart. |
| P9 | Mặc định số vòng/số phiên bản lưu trong `users.settings` (JSON); theme sáng/tối lưu `localStorage` (tiện ích theo trình duyệt, mặc định theo hệ thống). | Theo yêu cầu H; theme không cần đồng bộ giữa máy. |
| P10 | Frontend dùng thêm: react-router, @tanstack/react-query, lucide-react, sonner (toast); test bằng vitest + @testing-library/react + jsdom. Slider so sánh và zoom tự viết (input range gốc, dễ dùng bàn phím) thay vì thêm thư viện. | Ít dependency; shadcn đã kéo Radix. |
| P11 | Đăng ký mở (env `ALLOW_REGISTRATION=1` mặc định). uvicorn mặc định bind `127.0.0.1`; README cảnh báo khi dùng `--host 0.0.0.0` (route cũ không có auth). | Chạy local, vài người dùng. |

## Ghi nhận khi triển khai S0 (2026-10-22) — evidence-backed

| # | Ghi nhận |
|---|---|
| E1 | npm cài bản mới nhất: React 19.3, Vite 8.3, TypeScript 7.0, Tailwind 4.3, vitest 4.1 (spec §1.2 đã cập nhật; prompt chỉ yêu cầu "React", không ghim 18). |
| E2 | npm 11.4.2 lỗi `Cannot read properties of null (reading 'edgesOut')` khi giải peer dependency tùy chọn của vitest 4/5 → `web/.npmrc` đặt `legacy-peer-deps=true`; `npm ci` sạch đã kiểm tra. |
| E3 | `npx shadcn add` timeout (registry phản hồi ~11 s/request từ máy này). Theo phương án dự phòng của plan: tải JSON registry `new-york-v4` bằng curl và ghi file component nguyên văn (chỉ sửa import `cn` → `@/lib/utils`; `sonner.tsx` dùng ThemeProvider của app thay cho next-themes). |
| E5 | S2: `POST /workspace/sessions` khi chẩn đoán (start_session) lỗi vẫn trả **201** với phiên `status=failed` + `error` tiếng Việt, thay vì 500 như spec §3.3. Lý do: phiên và ảnh đã được lưu, client cần id để mở phiên và thử lại; body không chứa chuỗi exception. Lỗi validate upload vẫn trả 413/415/422 và không tạo phiên. |
| E6 | Component shadcn tải từ registry thô có import `@/registry/new-york-v4/ui/*` và nhãn sr-only tiếng Anh "Close"; đã đổi thành `@/components/ui/*` và "Đóng". |
| E7 | S8: artifact từng gửi `private, max-age=31536000, immutable`, nên ảnh đã xóa hoặc của người dùng khác trên cùng trình duyệt vẫn lấy được từ cache. Đã đổi thành `private, no-cache` (luôn kiểm quyền lại); đổi lại là ảnh tải lại mỗi lần mở phiên. |
| E8 | S8: token `--success` theme sáng đổi thành `oklch(0.49 0.13 150)` (badge "Hoàn tất" ở mục sidebar đang chọn chỉ đạt 4,44:1). Thêm viền focus 2 px cho phần tử gốc không có kiểu focus riêng và cho chỉ báo backend. |
| E4 | Dev server dùng `.venv` (Python 3.12, SQLAlchemy 2.1.6); môi trường global có SQLAlchemy 2.0.54. Test S0 xanh trên cả hai. |

## Hoãn lại (deferred) — confirmed ngoài phạm vi

Job nền / polling / SSE / hủy tác vụ; overlay mask; đổi mật khẩu; xóa tài khoản; khóa tạm khi đăng nhập sai nhiều lần; archive / tìm kiếm / nhân bản phiên; xuất PDF; trang quản trị; đa ngôn ngữ; Alembic; deploy; SQLite checkpointer cho LangGraph; mọi thay đổi thuật toán / prompt / ngưỡng; gỡ Gradio; thêm auth cho route cũ (xem Q1).

## Câu hỏi đã trả lời — confirmed (chủ dự án chọn mặc định, 2026-10-22)

- Q1 — Route cũ (`/process`, `/sessions`, `/memory?user_id=`) vẫn mở không auth và nhận `user_id` từ client (`routes.py:140,296,338`); ai truy cập được cổng 8000 đều đọc được thống kê bộ nhớ của mọi người dùng. CORS hiện `allow_origins=["*"]` + `allow_credentials=True` (`main.py:31-37`). **Mặc định:** giữ nguyên route cũ (hợp đồng), nhưng đổi CORS sang danh sách origin từ env `CORS_ORIGINS` (mặc định rỗng = chỉ cùng origin); ghi hạn chế vào README.
- Q2 — Khi thread hết hạn, có cho backend tự khôi phục (dùng câu hỏi đã lưu + câu trả lời → `run_pipeline(intent=…)`, như P2) thay vì bắt hỏi ý lại không? **Mặc định:** không; làm đúng D10 (thông báo + "Hỏi ý lại").
- Q3 — Kiểm tra trình duyệt: thêm `@playwright/test` làm devDependency của `web/` (tải Chromium ~150 MB) để có script e2e chạy lại được, hay chỉ dùng browser pane có sẵn của agent? **Mặc định:** dùng browser pane có sẵn cho kiểm tra 1366×768 / 390×844, sáng/tối, console + network; ghi kịch bản thành checklist trong `docs/frontend/plan.md`, không thêm Playwright.
