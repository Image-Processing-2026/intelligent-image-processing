# Frontend context — Web app "AI Image Doctor"

Ngày: 2026-10-22. Trạng thái phase: **Phase 1 xong** (context do chủ dự án cung cấp trong prompt nhiệm vụ). Tiếp theo: duyệt `spec.md` + `plan.md` (milestone trước khi code).

## 1. Phân loại nhiệm vụ

| Mục | Giá trị | Nhãn |
|---|---|---|
| Loại | Frontend cho backend có sẵn (FastAPI + LangGraph) + phần backend mới tối thiểu trong `src/api/` (auth, lưu trữ) | confirmed (prompt nhiệm vụ) |
| Mức hoàn thành | Real integration, chạy local (production-local = một lệnh uvicorn) | confirmed |
| Phạm vi | MVP A–H trong prompt; ngoài phạm vi ghi ở `decisions.md` | confirmed |
| Phase vào | Phase 1 (đã có) → 2 (journey/màn hình trong `spec.md`) → 3 rút gọn (một hướng, shadcn, teal) → 4 → 5 | confirmed |
| Độ sâu | Standard: nhiều journey, có hợp đồng mới; KHÔNG khám phá hướng thiết kế (một hướng duy nhất) | confirmed |
| Milestone review | Chỉ duyệt spec+plan trước khi code, rồi C (integrated journey review) ở cuối. Bỏ A, B. | confirmed |

## 2. Mục tiêu và người dùng

- Mục tiêu: một web app đủ chức năng để chủ dự án và nhóm **test và đánh giá** pipeline (chẩn đoán → hỏi ý → xử lý → phiên bản → tinh chỉnh), lưu lại mọi phiên và đánh giá để phân tích về sau. Cải thiện chất lượng pipeline KHÔNG thuộc nhiệm vụ. — confirmed
- Người dùng: vài thành viên nhóm / giảng viên trên máy local hoặc LAN, tiếng Việt. Không có nghiên cứu người dùng; persona không được dựng. — assumed
- Ưu tiên: chạy được, đúng, trung thực, ít phức tạp > độ bóng bẩy. — confirmed

## 3. Hệ thống hiện có (đã đọc)

- API: `src/api/main.py`, `routes.py`, `schemas.py`. Route `/api/v1/{health,diagnose,process,render,refine,sessions,sessions/{id}/answers,memory}`; ảnh trả base64 PNG; không auth; `user_id` do client gửi. — evidence-backed
- Phiên LangGraph: `src/agent/session.py` — `start_session` / `answer_session` / `end_session`, `MemorySaver` trong RAM, giới hạn `MAX_SESSIONS = 32` (thread cũ nhất bị xóa). — evidence-backed (`session.py:27-62`)
- Pipeline: `src/agent/graph.py::run_pipeline(image, ground_truth, is_synthetic, max_iterations, num_variants, intent, user_id)`. — evidence-backed (`graph.py:555`)
- Render: `planner.sanitize_actions` + `executor.execute_plan` (giữ thứ tự). Refine: `refine.refine(original, actions, feedback) -> RefineResult`. — evidence-backed
- Case memory: `src/agent/memory.py` (SQLite `data/memory/cases.sqlite`, tách theo `user_id`, `remember_run`, `record_choice`, `record_feedback`, `stats`). — evidence-backed
- Ý định: `src/agent/intent.py` — câu hỏi tất định (≤ 3), `intent_from_answers(questions, answers, notes)`. — evidence-backed
- UI hiện tại: Gradio `frontend/app.py` (924 dòng), giữ nguyên. Bài học: nhãn trung thực, theme tối từng mất chữ, không tiến độ giả. — evidence-backed (`frontend/README.md`, `docs/plan_ui_progress_demo.md`)
- Ảnh mẫu: `data/real/coffee_underexposed.jpg`, `astro_noisy.jpg`, `cat_warm_cast.jpg` có trên máy này, **nhưng không được track** (`.gitignore:416`). — evidence-backed

## 4. Năng lực môi trường (kiểm tra 2026-10-22)

| Năng lực | Trạng thái |
|---|---|
| Node / npm | v23.6.0 / 11.4.2 — có |
| Python local | 3.12.10 (CI là 3.10 → không dùng cú pháp 3.11+) |
| python-multipart, SQLAlchemy, argon2-cffi | Đã cài trong môi trường local (0.0.32 / 2.0.54 / 25.1.0) nhưng **chưa khai báo** trong `requirements.txt` |
| Skill `frontend-delivery` | Đã nạp. `shadcn` CLI dùng qua `npx shadcn` khi scaffold (cần mạng npm). `impeccable` không dùng (một hướng, không polish riêng). |
| Trình duyệt kiểm tra | Built-in browser pane (Claude) có sẵn; Playwright chưa có trong repo (`tests/browser/frontend_layout.js` là hàm cho Playwright MCP) — xem câu hỏi Q3 |
| Gemini | Không dùng trong test; kịch bản chấp nhận chạy rule-based (không API key) |

## 5. Hướng thiết kế (một hướng)

- shadcn/ui (Radix + Tailwind), màu nhấn teal, nền trung tính, font hệ thống (không tải font ngoài). Theme sáng/tối bằng biến CSS của shadcn; kiểm tra tương phản cả hai. — confirmed
- Bố cục: sidebar trái (phiên xử lý) + vùng chính dạng hội thoại; kết quả là các card trong luồng hội thoại; chi tiết kỹ thuật thu gọn. Mobile 390px: sidebar thành Sheet. — proposed (trong `spec.md`)
- Tiếng Việt toàn bộ copy UI. — confirmed

## 6. Ràng buộc chính (tóm tắt, chi tiết ở CLAUDE.md và prompt)

ADR-001 (không generative pixel); không sửa module 1–3 và thuật toán/prompt/ngưỡng của agent; chỉ thêm module mới trong `src/api/` + thay đổi tối thiểu `main.py`; không đổi hợp đồng route cũ; test offline; numpy<2; không CUDA; UI trung thực; dữ liệu upload không tin cậy; không commit/push khi chưa được yêu cầu.

## 7. Điều chưa biết mang theo

- Thời gian chạy thật trên CPU của 3/3 vòng/phiên bản với ảnh mẫu — unresolved (đo ở lát cuối).
- Hành vi với Gemini thật — unresolved, ngoài phạm vi kiểm chứng tự động.
