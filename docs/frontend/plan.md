# Plan — Web app "AI Image Doctor"

Trạng thái: **accepted** (chủ dự án duyệt 2026-10-22). Triển khai tuần tự S0 → S8; mỗi lát phải đạt "Xong khi" và chạy "Kiểm tra" trước khi sang lát sau. Mỗi lát kết thúc bằng một báo cáo ngắn (implemented / technically verified / real-mock, kiểm tra đã chạy). Không commit/push trừ khi chủ dự án yêu cầu (khi đó: nhánh `feat/ui-web` từ `develop`, Conventional Commits scope `ui`/`api`, không có Claude làm author/co-author).

Lệnh kiểm tra backend chung (PowerShell, từ repo root, không API key):
```powershell
$env:GEMINI_API_KEY = $null
ruff check src/ tests/; ruff format --check src/ tests/
pytest tests/unit/ -q
```
Lệnh frontend chung (trong `web/`): `npm run typecheck`, `npm test -- --run`, `npm run build`.

---

## S0 — Khung dự án
- Backend: `config.py`, `db.py` (engine WAL, `init_db`), `models.py` (toàn bộ bảng ở spec §2), `errors.py` (envelope + `route_class`), `web_static.py`. `main.py`: startup `init_db()`, CORS theo `CORS_ORIGINS`, `mount_web`. `requirements.txt` + `.gitignore`.
- Frontend: scaffold Vite React TS trong `web/`, Tailwind + `npx shadcn init` (teal, CSS variables, dark mode `class`), proxy `/api` với timeout 0, `api/client.ts` (fetch + envelope lỗi → `ApiError` tiếng Việt), ThemeProvider, AppShell rỗng.
- **Xong khi:** `uvicorn src.api.main:app` khởi động, tạo `data/app/app.sqlite` (WAL); `/api/v1/health` trả body cũ; `npm run dev` hiện AppShell qua proxy; `npm run build` rồi uvicorn phục vụ `/` và deep link `/s/abc` → `index.html`, `/api/v1/nope` vẫn 404 JSON.
- **Kiểm tra:** test `test_api_app.py` (health không đổi, static fallback với dist giả trong tmp_path, `/api/*` không bị fallback nuốt); typecheck/build; curl hai chế độ.

## S1 — Tài khoản (A)
- `auth.py`: register/login/logout/me/settings, argon2, token hash, cookie, `current_user`.
- UI: trang Đăng nhập/Đăng ký, route guard, menu người dùng + đăng xuất, xử lý 401 toàn cục.
- **Xong khi:** đăng ký → vào app; F5 vẫn đăng nhập; đăng xuất → về `/login`; lỗi trùng tên / sai mật khẩu hiện câu tiếng Việt.
- **Kiểm tra:** `test_api_auth.py` (spec §7.1 mục 1–3); vitest form auth; browser: luồng đăng ký/đăng nhập/đăng xuất, cookie httpOnly (không đọc được từ `document.cookie`).

## S2 — Phiên xử lý: tạo, danh sách, đổi tên, xóa (B)
- `storage.py` (kiểm tra upload spec §3.5, PNG + thumbnail, xóa thư mục an toàn), endpoint sessions list/get/patch/delete, artifact GET (kiểm quyền, download), samples.
- Ở lát này `POST sessions` đã gọi `start_session` thật (rule-based) và lưu run `analyze` + messages `diagnosis`/`questions` — để S3 chỉ còn phần trả lời.
- UI: Phiên mới (dropzone, mẫu, tùy chọn, ground truth), sidebar, đổi tên inline, xóa có xác nhận, trang phiên hiển thị ảnh gốc + card chẩn đoán.
- **Xong khi:** tạo phiên từ upload và từ mẫu; sidebar cập nhật; đổi tên; xóa xong thư mục mất; ảnh qua URL có kiểm quyền.
- **Kiểm tra:** `test_api_sessions.py` (§7.1 mục 4, 5, 11 + phần tạo của 6); vitest tạo phiên; browser: kéo thả, chọn mẫu, file sai định dạng.

## S3 — Hỏi ý và chạy pipeline (C, F)
- `workspace_service.py`: semaphore, trạng thái `running`/khóa phiên, answers (nhánh thường `answer_session`; nhánh ground truth P2), clarify, tuần tự hóa `DoctorState` → run `process` + artifacts (`final`, `intermediate`, `variant_preview`) + variants + treatment v1, `remember_run(state, str(user.id))`, `end_session` sau khi xong.
- UI: card câu hỏi (RadioGroup + ghi chú), card chờ trung thực, notice hết hạn + "Hỏi ý lại", card kết quả tối thiểu (ảnh sau, badge quyết định, nguồn).
- **Xong khi:** luồng đầy đủ rule-based ra kết quả; F5 giữa chừng không vỡ; restart server khi đang chờ trả lời → "Phiên hỏi ý đã hết hạn" → hỏi ý lại → chạy được; nút khóa khi chạy, mở lại khi lỗi.
- **Kiểm tra:** `test_api_flow.py` (§7.1 mục 6, 7, 9, 12 + ground truth: PSNR/SSIM có trong `final_evaluation`); browser: luồng đầy đủ + restart server thật.

## S4 — Hiển thị kết quả và render (D)
- Endpoint render phiên bản (`sanitize_actions` + `execute_plan`, `record_choice`).
- UI: so sánh slider + zoom dialog, lưới phiên bản (hạng, đề xuất, lý do, điểm, critic_note, nhãn preview), panel chẩn đoán đầy đủ, timeline các vòng, nhãn nguồn, "Điểm đánh giá ước lượng"/PSNR-SSIM, Nâng cao JSON, Tải về.
- **Xong khi:** chọn phiên bản → render full-res → tải PNG đúng kích thước gốc; mọi nhãn trung thực theo spec §5.5.
- **Kiểm tra:** test render (§7.1 mục 8 phần render); vitest card kết quả với fixture (badge "Dữ liệu mẫu" chỉ trong test); browser: slider bằng bàn phím, zoom, tải file.

## S5 — Tinh chỉnh và lịch sử phiên bản (E)
- Endpoint refine + activate; message `feedback`/`refine_result`.
- UI: ô góp ý, card điều chỉnh, danh sách phiên bản phác đồ, "Dùng bản này", so sánh theo treatment active.
- **Xong khi:** góp ý "da hơi vàng" tạo v2 với điều chỉnh hiển thị; quay về v1 đổi ảnh so sánh; mở lại phiên thấy đủ.
- **Kiểm tra:** test refine/activate (§7.1 mục 8); browser.

## S6 — Đánh giá và xuất JSON (G)
- Endpoint ratings (upsert) + export.
- UI: Tốt/Chưa tốt + ghi chú + Lưu (hiện trạng thái đã lưu), nút Xuất JSON.
- **Xong khi:** đánh giá lưu và hiện lại sau F5; file JSON tải về đúng schema spec §6.
- **Kiểm tra:** §7.1 mục 10; browser tải file, mở xem nội dung.

## S7 — Hệ thống, bộ nhớ, cài đặt (H)
- Endpoint system + memory theo user; dùng `PATCH /auth/me/settings`.
- UI: chỉ báo backend/chế độ ở header (poll KHÔNG dùng; kiểm tra khi tải trang và khi request lỗi mạng), trang Bộ nhớ của tôi, trang Cài đặt (mặc định vòng/phiên bản dùng cho Phiên mới, theme).
- **Xong khi:** tắt backend → chỉ báo "mất kết nối" ở lần request kế tiếp, bật lại → "hoạt động" khi thử lại; memory chỉ hiện ca của user hiện tại.
- **Kiểm tra:** test memory tách user (cấu hình CaseMemory trong tmp_path), test system không lộ khóa; browser.

## S8 — Xác minh tổng thể và bàn giao
- Chạy kịch bản chấp nhận spec §7.3 trong browser pane (backend production-local, không API key) ở 1366×768 và 390×844, sáng và tối; theo dõi console + network; lưu ảnh chụp vào `scratch/` (đã gitignore).
- Kiểm tra bàn phím: Tab qua toàn bộ luồng chính, focus ring, Esc đóng dialog.
- Đo thời gian thực tế 3 vòng/3 phiên bản trên ảnh mẫu, ghi vào README.
- Cập nhật `README.md` (gốc) + `web/README.md`: chạy dev (2 lệnh), production-local (build + 1 lệnh uvicorn), biến môi trường, vị trí dữ liệu (`data/app/`, `data/memory/`), hạn chế đã biết (thread mất khi restart/vượt 32, run đồng bộ không tiến độ, chất lượng pipeline chưa đánh giá, route cũ không auth, EXIF khác route cũ, ảnh mẫu không track trong git).
- Báo cáo cuối theo mẫu frontend-delivery.
- **Xong khi:** mọi mục spec §7 có bằng chứng hoặc được ghi rõ là chưa đạt.

---

## Checklist kiểm tra trình duyệt (dùng ở S1–S8)
- [ ] Không lỗi console; không request thất bại ngoài lỗi chủ động kiểm tra
- [ ] 1366×768 và 390×844: không tràn ngang (`document.documentElement.scrollWidth <= innerWidth`)
- [ ] Sáng và tối: tiêu đề, nhãn, badge, chữ trong card đọc được (tương phản ≥ 4.5:1 cho chữ thường)
- [ ] Bàn phím: mọi nút/điều khiển truy cập được, focus nhìn thấy
- [ ] Nút gây ghi khóa khi đang chạy, mở lại khi lỗi
- [ ] Không có nhãn "độ chính xác"/"%" cho điểm; quyết định và nguồn đúng spec

## Rủi ro
- Pipeline chậm trên CPU → request dài vài phút: đã chấp nhận (D3); kiểm tra không có timeout ở proxy/uvicorn ở S3.
- `npx shadcn` cần mạng npm khi scaffold; nếu thất bại, chép component thủ công từ registry đã đọc (vẫn là shadcn).
- Test pipeline thật làm suite chậm: dùng ảnh tổng hợp ≤ 96×96, `max_iterations=1`, chỉ vài test chạy pipeline thật, còn lại patch hàm agent.
