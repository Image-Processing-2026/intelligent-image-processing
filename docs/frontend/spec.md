# Spec — Web app "AI Image Doctor" (React + FastAPI, local)

Trạng thái: **accepted** (chủ dự án duyệt 2026-10-22). Quyết định nền: `decisions.md` (D1–D11, P1–P11, Q1–Q3 đều confirmed). Context: `context.md`.

---

## 1. Kiến trúc

```
Trình duyệt ──(dev) Vite :5173 ── proxy /api ──┐
           └─(prod-local) ─────────────────────┴─▶ FastAPI :8000
                                                    ├─ /api/v1/{health,diagnose,…}   route cũ, KHÔNG đổi
                                                    ├─ /api/v1/auth/*                 mới
                                                    ├─ /api/v1/workspace/*            mới (yêu cầu đăng nhập)
                                                    └─ /*  → web/dist (nếu có), fallback index.html
FastAPI ─▶ src.agent (run_pipeline, start_session/answer_session/end_session, sanitize_actions+execute_plan, refine, memory)
        ─▶ data/app/app.sqlite (SQLAlchemy, WAL)      ─▶ data/app/users/<uid>/sessions/<sid>/*.png
        ─▶ data/memory/cases.sqlite (CaseMemory, không đổi)
```

### 1.1 Module backend mới (chỉ trong `src/api/`)

| File | Nội dung |
|---|---|
| `src/api/config.py` | Đọc env (bảng 1.3), giá trị mặc định. |
| `src/api/db.py` | Engine SQLite (WAL, `foreign_keys=ON`, `check_same_thread=False`), `SessionLocal`, `init_db()` (`create_all` + reset phiên `running` → `failed`, P8), dependency `get_db`. |
| `src/api/models.py` | Model SQLAlchemy 2.0 (mục 2). |
| `src/api/errors.py` | `ApiError(status, code, message_vi)`, `route_class` bọc lỗi thành envelope (P5), bảng mã lỗi (mục 4). |
| `src/api/auth.py` | Băm/kiểm mật khẩu (argon2), tạo/kiểm token, dependency `current_user`, router `/api/v1/auth`. |
| `src/api/storage.py` | Kiểm tra + chuẩn hóa ảnh upload, ghi PNG/thumbnail, đường dẫn do server sinh, xóa thư mục phiên an toàn. |
| `src/api/workspace.py` | Router `/api/v1/workspace` (phiên, luồng, artifact, đánh giá, xuất, memory, system). |
| `src/api/workspace_service.py` | Bọc hàm agent + semaphore + tuần tự hóa kết quả (bỏ base64, ảnh → artifact). |
| `src/api/workspace_schemas.py` | Pydantic v2 request/response của API mới. |
| `src/api/web_static.py` | `mount_web(app, dist_dir)`: phục vụ `web/dist` nếu tồn tại, fallback `index.html` cho path không bắt đầu bằng `/api`. |

Thay đổi `src/api/main.py` (tối thiểu): `init_db()` khi khởi động (lifespan/startup), `include_router(auth_router)`, `include_router(workspace_router)`, CORS theo `CORS_ORIGINS` (Q1), `mount_web(app)` sau cùng. Route cũ, `routes.py`, `schemas.py` không đổi.

`requirements.txt`: thêm `python-multipart`, `sqlalchemy>=2.0`, `argon2-cffi`. `.gitignore`: thêm `data/app/`, `web/node_modules/`, `web/dist/`.

### 1.2 Frontend `web/`

Vite 8 + React 19 + TypeScript 7 (strict) + Tailwind 4 + shadcn/ui (new-york, radix-ui); react-router, TanStack Query, lucide-react, sonner; vitest + Testing Library. `vite.config.ts`: proxy `/api` → `http://127.0.0.1:8000`, `timeout: 0`, `proxyTimeout: 0`. `fetch` không đặt AbortSignal timeout. Mọi request `credentials: "same-origin"`.

### 1.3 Biến môi trường mới

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `APP_DATA_DIR` | `data/app` | Gốc DB + file ảnh |
| `PIPELINE_CONCURRENCY` | `1` | Số lời gọi pipeline đồng thời (1–4) |
| `MAX_UPLOAD_MB` | `20` | Dung lượng file tối đa |
| `MAX_IMAGE_PIXELS` | `40000000` | Số pixel tối đa (chống decompression bomb) |
| `MIN_IMAGE_SIDE` | `16` | Cạnh nhỏ nhất |
| `AUTH_TOKEN_TTL_DAYS` | `30` | Hạn token |
| `COOKIE_SECURE` | `0` | `1` khi chạy sau HTTPS |
| `ALLOW_REGISTRATION` | `1` | `0` để tắt đăng ký |
| `CORS_ORIGINS` | rỗng | Danh sách origin, phân cách bằng dấu phẩy (Q1) |
| `WEB_DIST_DIR` | `web/dist` | Thư mục bản build |
| `GEMINI_API_KEY`, `GEMINI_MODEL`, `CASE_MEMORY_PATH` | như hiện có | Không đổi |

---

## 2. Mô hình dữ liệu (SQLite, `data/app/app.sqlite`)

Thời gian: ISO-8601 UTC. Id: UUID hex (P1) trừ khi ghi khác. `ON DELETE CASCADE` cho mọi khóa ngoại tới `processing_sessions` và `users`.

| Bảng | Cột | Ghi chú |
|---|---|---|
| `users` | `id` PK, `username` UNIQUE (lưu chữ thường), `password_hash`, `settings` JSON `{max_iterations:3, num_variants:3}`, `created_at` | `str(id)` = `user_id` của CaseMemory (D8) |
| `auth_tokens` | `token_hash` PK (SHA-256 hex), `user_id` FK, `created_at`, `expires_at` | Token thô chỉ có trong cookie `iid_token` |
| `processing_sessions` | `id` PK, `user_id` FK, `title`, `status`, `options` JSON `{max_iterations, num_variants, has_ground_truth}`, `thread_id` NULL, `case_id` NULL, `questions` JSON NULL, `active_treatment_id` NULL, `error_code` NULL, `created_at`, `updated_at` | `status` ∈ `uploaded`, `awaiting_answers`, `running`, `done`, `failed`. `thread_id` không bao giờ gửi ra client. |
| `messages` | `id` INTEGER PK tự tăng, `session_id` FK, `role` ∈ `system`,`user`,`assistant`, `kind`, `content` JSON, `created_at` | `kind` ∈ `diagnosis`, `questions`, `answers`, `result`, `feedback`, `refine_result`, `render_result`, `notice` |
| `runs` | `id` PK, `session_id` FK, `kind` ∈ `analyze`,`process`,`render`,`refine`, `status` ∈ `done`,`failed`, `result` JSON, `error` (nội bộ, không trả nguyên văn), `duration_ms`, `created_at` | `analyze` = start_session (chẩn đoán + câu hỏi) |
| `artifacts` | `id` PK, `session_id` FK, `run_id` FK NULL, `kind` ∈ `original`,`ground_truth`,`thumbnail`,`final`,`variant_preview`,`intermediate`,`render`,`refine`, `rel_path` (tương đối với thư mục phiên, do server sinh), `width`, `height`, `bytes`, `created_at` | Ảnh PNG; thumbnail JPEG 256px |
| `variants` | `id` PK, `session_id` FK, `run_id` FK, `style_id` (balanced/natural/vivid), `label`, `description`, `rank`, `quality_score`, `quality_improved`, `critic_note`, `actions` JSON, `preview_artifact_id` FK, `recommended` bool, `chosen` bool | |
| `treatments` | `id` PK, `session_id` FK, `parent_id` FK NULL, `version` (1,2,3… trong phiên), `source` ∈ `pipeline`,`variant`,`refine`, `variant_id` NULL, `actions` JSON, `feedback_text` NULL, `adjustments` JSON, `notes` JSON, `refine_source` NULL (`rules`/`vlm`/`none`), `quality_score` NULL, `artifact_id` FK NULL, `run_id` FK, `created_at` | Phiên bản phác đồ; `pipeline` = treatment của run process (ảnh `final`) |
| `ratings` | `id` PK, `session_id` FK, `run_id` FK, `treatment_id` FK NULL, `rating` ∈ `good`,`bad`, `note`, `created_at`, `updated_at` | UNIQUE(`run_id`, `treatment_id`): đánh giá lại = cập nhật |

Thư mục phiên: `data/app/users/<user_id>/sessions/<session_id>/` chứa `original.png`, `thumb.jpg`, `ground_truth.png`, `runs/<run_id>/{final,variant_<style>,iter_<n>,render,refine}.png`. Mọi đường dẫn ghép từ id do server sinh; khi đọc kiểm tra `resolve()` nằm trong thư mục phiên.

`result` JSON của run `process` (tuần tự hóa từ `DoctorState`, không base64):
```json
{
  "total_iterations": 2, "final_decision": "SHIP|STOP_BEST_EFFORT|RE_PROCESS",
  "final_evaluation": {...}, "history": [HistoryItem.model_dump(mode="json"), ...],
  "treatment": [RegionOperation...], "intent": IntentProfile|null,
  "recommended_variant": "balanced", "recommended_reason": "intent|preference|critic|score",
  "variant_ranking_source": "critic|score", "case_id": "...",
  "artifacts": {"final": "<artifact_id>", "intermediate": ["<id>", ...]},
  "sources": {"diagnosis": ["rule_based", ...], "plan": ["rule_based", ...]}
}
```
`sources` = `history[i].diagnosis.source` và `history[i].plan.source` theo vòng (nguồn THẬT, không suy từ API key).

---

## 3. Hợp đồng API mới

Tất cả dưới `/api/v1`. Body JSON trừ khi ghi `multipart`. Endpoint `workspace/*` cần cookie hợp lệ (401 nếu không). Mọi `{sid}` không thuộc user hiện tại → 404 `not_found` (không phân biệt "không tồn tại" và "của người khác").

### 3.1 Auth — `src/api/auth.py`

| Method + path | Request | Response | Lỗi |
|---|---|---|---|
| `POST /auth/register` | `{username, password}` — username 3–32 ký tự `[a-z0-9_.-]` (không phân biệt hoa thường), password 8–128 | 201 `User` + Set-Cookie (tự đăng nhập) | 409 `username_taken`, 422 `validation`, 403 `registration_closed` |
| `POST /auth/login` | `{username, password}` | 200 `User` + Set-Cookie | 401 `invalid_credentials` (cùng thông báo cho sai tên và sai mật khẩu; vẫn chạy verify với hash giả để thời gian tương đương) |
| `POST /auth/logout` | — | 204, xóa bản ghi token + xóa cookie | luôn 204 |
| `GET /auth/me` | — | 200 `User` | 401 `unauthenticated` |
| `PATCH /auth/me/settings` | `{max_iterations?: 1–5, num_variants?: 1–3}` | 200 `User` | 422 |

`User = {id, username, settings: {max_iterations, num_variants}, created_at}`. Cookie `iid_token`: httpOnly, SameSite=Lax, Path=/, Max-Age = TTL, Secure theo `COOKIE_SECURE`. Token: `secrets.token_urlsafe(32)`; DB lưu SHA-256. Token hết hạn bị xóa khi gặp. Giới hạn 1–5 vòng và 1–3 phiên bản lấy đúng theo Gradio (`frontend/app.py:619-624`) và `variants.MAX_VARIANTS = 3`; không đặt ngưỡng mới cho pipeline.

### 3.2 Phiên xử lý — `src/api/workspace.py`

| Method + path | Request | Response | Bọc hàm có sẵn |
|---|---|---|---|
| `GET /workspace/sessions` | — | `SessionSummary[]` mới nhất trước | — |
| `POST /workspace/sessions` | multipart: `file` **hoặc** `sample` (tên mẫu), `ground_truth?` (file), `max_iterations?`, `num_variants?`, `title?` | 201 `SessionDetail` (status `awaiting_answers`) | `storage` + `start_session(img, max_iterations, num_variants, user_id=str(user.id))` → lưu run `analyze`, messages `diagnosis` + `questions` |
| `GET /workspace/sessions/{sid}` | — | `SessionDetail` (đọc từ DB, không chạy pipeline) | — |
| `PATCH /workspace/sessions/{sid}` | `{title}` 1–120 ký tự | `SessionSummary` | — |
| `DELETE /workspace/sessions/{sid}` | — | 204; xóa bản ghi (cascade) + thư mục phiên + `end_session(thread_id)` nếu có | `end_session` |
| `POST /workspace/sessions/{sid}/answers` | `{answers: {qid: value}, notes: str ≤ 1000}` | `SessionDetail` (status `done`) | Không có ground truth: `answer_session(thread_id, answers, notes)`. Có ground truth: `intent_from_answers(questions, answers, notes)` + `run_pipeline(img, gt, is_synthetic=True, intent=…, user_id)` + `end_session` (P2). Sau đó `remember_run(state, str(user.id))`, ghi run `process`, artifacts, variants, treatment v1 (`pipeline`), message `answers` + `result`. |
| `POST /workspace/sessions/{sid}/clarify` | — | `SessionDetail` (status `awaiting_answers`) | Hỏi ý lại khi thread hết hạn: `start_session` trên `original.png` đã lưu; chỉ hợp lệ khi status `awaiting_answers`/`failed` và chưa có run `process` xong |
| `POST /workspace/sessions/{sid}/variants/{vid}/render` | — | `SessionDetail`; thêm treatment `variant` + artifact `render`, đặt active, `chosen=true` | `sanitize_actions(variant.actions)` + `execute_plan(original, TreatmentPlan(reasoning="render", actions=…))`; `memory.record_choice(case_id, style_id)` |
| `POST /workspace/sessions/{sid}/refine` | `{feedback: str 1–500, treatment_id?: str}` (mặc định active) | `SessionDetail`; thêm treatment `refine` (parent = treatment gốc), message `feedback` + `refine_result`, đặt active | `refine(original, treatment.actions, feedback)`; `memory.record_feedback(case_id, feedback, adjustments)` nếu có adjustments |
| `POST /workspace/sessions/{sid}/treatments/{tid}/activate` | — | `SessionDetail` | Quay về phiên bản trước; không chạy pipeline |
| `GET /workspace/sessions/{sid}/artifacts/{aid}` | query `download=1?` | `FileResponse` (image/png hoặc image/jpeg), `Cache-Control: private, max-age=31536000, immutable`; `download=1` → `Content-Disposition: attachment; filename="<title>-<kind>.png"` (tên đã làm sạch) | — |
| `PUT /workspace/sessions/{sid}/ratings` | `{run_id, treatment_id?, rating: good|bad, note ≤ 2000}` | `Rating` | — |
| `GET /workspace/sessions/{sid}/export` | — | JSON tải về (`Content-Disposition: attachment; filename="session-<sid>.json"`) | — |
| `GET /workspace/samples` | — | `[{id, label, thumbnail_url}]` chỉ các file thực sự tồn tại trong `data/real/` | — |
| `GET /workspace/samples/{id}/thumbnail` | — | JPEG | — |
| `GET /workspace/memory` | — | `MemoryStats` của `str(user.id)` | `get_case_memory().stats(str(user.id))` |
| `GET /workspace/system` | — | `{backend:"ok", gemini_key_configured: bool, gemini_model: str, memory_enabled: bool, pipeline_concurrency: int, registration_open: bool}` | Không gọi model. UI ghi rõ "có khóa ≠ đã dùng Gemini". |

Mẫu (allowlist cứng, không nhận đường dẫn từ client): `underexposed` → `coffee_underexposed.jpg` "Thiếu sáng"; `noisy` → `astro_noisy.jpg` "Nhiễu"; `warm_cast` → `cat_warm_cast.jpg` "Ám màu".

Kiểu dữ liệu chính:
```ts
SessionSummary = { id, title, status, created_at, updated_at, thumbnail_url: string|null,
                   final_decision: string|null }
SessionDetail  = SessionSummary & {
  options: {max_iterations, num_variants, has_ground_truth},
  original: Artifact, ground_truth: Artifact|null,
  messages: Message[],                      // hội thoại theo thời gian
  diagnosis: DiagnosisReport|null,          // từ run analyze (vòng 1)
  questions: Question[]|null,               // khi awaiting_answers
  process: ProcessResult|null,              // run process đã xong
  variants: Variant[],                      // xếp theo rank; preview_url, recommended, chosen
  treatments: Treatment[], active_treatment_id: string|null,
  ratings: Rating[], error: {code, message}|null }
Artifact = { id, kind, url, download_url, width, height }
```
`url` là đường dẫn tương đối `/api/v1/workspace/sessions/{sid}/artifacts/{aid}`.

### 3.3 Chuyển trạng thái phiên xử lý

```
POST sessions ─▶ running ─▶ awaiting_answers ──answers──▶ running ─▶ done
                   │              │  (thread hết hạn: 409 thread_expired → clarify)
                   └─▶ failed ◀───┘  failed ──clarify──▶ running ─▶ awaiting_answers
```
- `running` set trong cùng transaction kiểm tra trạng thái; request thứ hai trên cùng phiên khi đang `running` → 409 `session_busy`.
- `done`: render/refine/activate/ratings/export được phép; answers/clarify → 409 `invalid_state`.
- Lỗi pipeline: run `failed` + `error` (log server với traceback), phiên quay về trạng thái trước (`awaiting_answers` nếu thread còn) hoặc `failed`; message `notice` tiếng Việt; client nhận 500 `pipeline_failed`.
- Lỗi render/refine không đổi trạng thái `done`; chỉ thêm run `failed`.

### 3.4 Thread LangGraph hết hạn

`answers` khi `SessionNotFoundError` (restart hoặc vượt `MAX_SESSIONS`) → giữ `awaiting_answers`, thêm message `notice` "Phiên hỏi ý đã hết hạn", trả 409 `thread_expired`. UI: thông báo + nút **Hỏi ý lại** → `POST …/clarify` (chạy lại chẩn đoán trên ảnh đã lưu, câu hỏi mới thay câu hỏi cũ; câu trả lời đang nhập được giữ trong form nếu id câu hỏi trùng). Không vỡ trang. Nhánh ground truth (P2) không phụ thuộc thread khi trả lời nên không gặp lỗi này.

### 3.5 Kiểm tra upload (`storage.py`)

1. Content-Length / số byte đọc > `MAX_UPLOAD_MB` → 413 `file_too_large` (đọc theo khối, dừng sớm).
2. Mở bằng Pillow với `Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS`; `format` ∈ {JPEG, PNG, WEBP} (không tin đuôi file hay Content-Type) → khác: 415 `unsupported_format`; lỗi giải mã / bomb → 422 `invalid_image`.
3. `ImageOps.exif_transpose`, chuyển RGB (grayscale/alpha → RGB, nền trắng cho alpha), `np.uint8 (H, W, 3)`.
4. Cạnh < `MIN_IMAGE_SIDE` → 422 `image_too_small`.
5. Ground truth phải cùng kích thước ảnh gốc (sau xoay) → 422 `ground_truth_mismatch`.
6. Lưu `original.png` + `thumb.jpg`; tên file client chỉ dùng làm `title` mặc định (cắt 120 ký tự, bỏ ký tự điều khiển).

---

## 4. Mã lỗi → câu tiếng Việt

Envelope: `{"error": {"code": "...", "message": "..."}}`. Không bao giờ chứa traceback, chuỗi exception thô, đường dẫn đĩa hay khóa API. Exception không lường trước → log `logger.exception(...)` (tiếng Anh) + 500 `internal`.

| HTTP | code | message (UI hiện nguyên văn) |
|---|---|---|
| 401 | `unauthenticated` | Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại. |
| 401 | `invalid_credentials` | Tên đăng nhập hoặc mật khẩu không đúng. |
| 403 | `registration_closed` | Máy chủ đang tắt đăng ký tài khoản mới. |
| 404 | `not_found` | Không tìm thấy phiên xử lý hoặc tệp yêu cầu. |
| 409 | `username_taken` | Tên đăng nhập đã được dùng. |
| 409 | `session_busy` | Phiên này đang được xử lý. Vui lòng đợi. |
| 409 | `invalid_state` | Thao tác không phù hợp với trạng thái hiện tại của phiên. |
| 409 | `thread_expired` | Phiên hỏi ý đã hết hạn (máy chủ khởi động lại hoặc quá nhiều phiên đang chờ). Hãy hỏi ý lại. |
| 413 | `file_too_large` | Tệp vượt quá {MAX_UPLOAD_MB} MB. |
| 415 | `unsupported_format` | Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP. |
| 422 | `invalid_image` | Không đọc được ảnh, hoặc ảnh quá lớn. |
| 422 | `image_too_small` | Ảnh quá nhỏ để xử lý. |
| 422 | `ground_truth_mismatch` | Ảnh chuẩn phải cùng kích thước với ảnh gốc. |
| 422 | `validation` | Dữ liệu gửi lên không hợp lệ. (+ `fields` tùy chọn: tên trường → câu ngắn) |
| 500 | `pipeline_failed` | Xử lý ảnh không thành công. Bạn có thể thử lại. |
| 500 | `internal` | Có lỗi phía máy chủ. Vui lòng thử lại. |

Frontend: lỗi mạng / không phải JSON → "Không kết nối được máy chủ. Kiểm tra backend đang chạy rồi thử lại." 401 ở bất kỳ request nào → xóa cache, chuyển `/login?next=…`.

---

## 5. Màn hình và trạng thái

Route frontend: `/login`, `/register`, `/` (phiên mới), `/s/:sid`, `/memory`, `/settings`. Route cần đăng nhập chuyển `/login` khi `GET /auth/me` = 401.

### 5.1 Khung chung (AppShell)
- Header: tên app, chỉ báo backend (`/health` + `/workspace/system`: "Backend: hoạt động / mất kết nối"; "Chế độ: có khóa Gemini (nguồn thật xem ở từng kết quả)" hoặc "Chế độ: rule-based (không có khóa Gemini)"), nút theme sáng/tối/hệ thống, menu người dùng (Bộ nhớ của tôi, Cài đặt, Đăng xuất).
- Sidebar: nút "Phiên mới", danh sách `SessionSummary` (thumbnail, tên, badge trạng thái, thời điểm tương đối + tooltip tuyệt đối). Trạng thái: loading (skeleton), rỗng ("Chưa có phiên nào"), lỗi (thông báo + Thử lại). ≤ 767px: sidebar thành Sheet mở bằng nút menu.

### 5.2 Đăng nhập / Đăng ký
Form username + password (+ xác nhận mật khẩu khi đăng ký). Trạng thái: idle, submitting (nút khóa + spinner), lỗi trường (inline), lỗi server (Alert). Enter submit; focus vào trường lỗi đầu tiên.

### 5.3 Phiên mới (`/`)
Dropzone (kéo thả + nút chọn file, `accept="image/jpeg,image/png,image/webp"`, kiểm tra dung lượng phía client trước), ảnh mẫu (chỉ những mẫu backend trả về; ẩn nếu rỗng), "Tùy chọn" (Accordion, đóng mặc định): số vòng tối đa (1–5, mặc định từ settings = 3), số phiên bản (1–3, mặc định 3), "Chế độ nâng cao: ảnh chuẩn (ground truth) để đo PSNR/SSIM" + dropzone thứ hai + chú thích "Chỉ dùng khi có ảnh chuẩn tương ứng để đánh giá thử nghiệm; khi có ảnh chuẩn, chẩn đoán vòng 1 chạy hai lần". Nút **Bắt đầu chẩn đoán**. Trạng thái: chưa chọn ảnh (nút khóa), đã chọn (preview), đang gửi (spinner + câu chờ chuẩn, khóa toàn form), lỗi (Alert, form giữ nguyên, thử lại). Thành công → điều hướng `/s/:sid`.

### 5.4 Phiên xử lý (`/s/:sid`) — luồng hội thoại
Tiêu đề có thể đổi tên (inline edit, Enter lưu, Esc hủy), nút Xuất JSON, nút Xóa (AlertDialog xác nhận: "Xóa phiên và mọi ảnh trên đĩa? Không thể hoàn tác."). Nội dung là danh sách message theo thời gian, render theo `kind`:

| kind | Card |
|---|---|
| `diagnosis` | Ảnh gốc + **Chẩn đoán**: cảnh, ánh sáng, tóm tắt, lỗi theo mức độ (badge 0–3 + vùng + bằng chứng + origin), cần giữ (aspect @ vùng + lý do), số đo theo vùng (bảng `region_metrics`), nhãn nguồn `diagnosis.source` |
| `questions` | Câu hỏi (RadioGroup cho mỗi câu, chọn sẵn `default`), ô ghi chú tự do, nút **Chạy xử lý**. Chỉ tương tác được nếu là bộ câu hỏi hiện hành và status `awaiting_answers`; bộ cũ hiển thị chỉ đọc. |
| `answers` | Bong bóng người dùng: các lựa chọn (nhãn), ghi chú |
| `result` | Card kết quả (mục 5.5) |
| `feedback` | Bong bóng người dùng: nội dung góp ý |
| `refine_result` | Ảnh mới (thumbnail, mở so sánh), "Đã điều chỉnh: …" (`notes`), nguồn hiểu góp ý (`rules`/`vlm`/`none` → "luật" / "Gemini" / "không hiểu góp ý — giữ nguyên phác đồ"), điểm ước lượng nếu có |
| `render_result` | "Đã render phiên bản {label} ở độ phân giải gốc {w}×{h}" + nút Tải về |
| `notice` | Alert trung tính (hết hạn thread, lỗi lần chạy) + hành động liên quan (Hỏi ý lại / Thử lại) |

Trạng thái trang: loading (skeleton), 404 ("Không tìm thấy phiên xử lý" + về Phiên mới), đang chạy (card chờ cuối luồng: spinner + "Đang xử lý, thời gian phụ thuộc ảnh và số vòng", mọi nút gây ghi bị khóa, sidebar vẫn dùng được; nếu mở phiên khác giữa chừng, request vẫn chạy và kết quả hiện khi quay lại), lỗi request (toast + notice, nút mở lại).

### 5.5 Card kết quả
- Badge quyết định: `SHIP` → "Đạt ngưỡng đánh giá của hệ thống"; `STOP_BEST_EFFORT` → "Đã dừng — xem kết quả và lịch sử"; `RE_PROCESS` → "Tiếp tục lặp"; số vòng đã chạy.
- Nhãn nguồn theo vòng: "Chẩn đoán: Gemini / Luật (rule_based) / Luật dự phòng (vlm_fallback)" và tương tự cho phác đồ — từ `sources`. Nếu có `vlm_fallback`: ghi chú "Gọi Gemini lỗi, đã dùng luật dự phòng".
- **So sánh trước/sau**: ảnh gốc vs ảnh của treatment đang active (mặc định `final`); slider bằng `<input type="range">` (phím ←/→, Home/End) + nút Zoom mở Dialog (Vừa khung / 100% / 200%, cuộn để pan). Ảnh `object-contain`, không méo.
- "Điểm đánh giá ước lượng": `estimated_quality_score` (+ BRISQUE/NIQE nếu có); chế độ ground truth: PSNR / SSIM / MSE với nhãn "so với ảnh chuẩn".
- **Lưới phiên bản** (khi > 1): mỗi ô preview (nhãn "Xem trước ≤ 1024px"), hạng, nhãn "Đề xuất" cho `recommended`, lý do đề xuất (`intent` → "theo ý định của bạn", `preference` → "theo lựa chọn trước đây của bạn", `critic` → "Gemini nhận xét", `score` → "theo điểm ước lượng"), điểm, `critic_note`, nút **Chọn và render** → sau đó **Tải về** (full-res).
- **Timeline các vòng** (Accordion): mỗi vòng — chẩn đoán vòng đó (tóm tắt), phác đồ (thao tác @ vùng (tham số)), lý do (`plan.reasoning`), id thẻ tri thức (`plan.knowledge`), chỉ số trước/sau (độ sáng, tương phản, nhiễu, độ nét…), điểm, quyết định, "đã rollback" nếu có, ảnh trung gian.
- **Tinh chỉnh**: Textarea góp ý (placeholder "Ví dụ: da hơi vàng, trời gắt quá"), nút Gửi góp ý; **Lịch sử phiên bản phác đồ**: danh sách v1, v2… (nguồn, góp ý, thời điểm, nút "Dùng bản này" → activate, nút Tải về).
- **Đánh giá**: Tốt / Chưa tốt (ToggleGroup) + ghi chú + Lưu; áp cho treatment đang active của run process.
- **Nâng cao** (Collapsible, đóng): JSON kỹ thuật (`process.result`, `diagnosis`, treatment active) dạng `<pre>` cuộn được, nút sao chép.

### 5.6 Bộ nhớ của tôi (`/memory`)
`enabled=false` → "Bộ nhớ ca bệnh đang tắt trên máy chủ". Ngược lại: số ca, lựa chọn phong cách (bảng đếm), góp ý hay gặp, phong cách ưa thích (hoặc "chưa đủ dữ liệu: cần ≥ 3 lần chọn"). Trạng thái loading / lỗi.

### 5.7 Cài đặt (`/settings`)
Số vòng mặc định, số phiên bản mặc định (lưu server), theme (localStorage, try/catch). Thông tin hệ thống từ `/workspace/system`.

### 5.8 Yêu cầu xuyên suốt
- Không giả tiến độ; không có nhãn "độ chính xác"/"%" cho điểm.
- Không có dữ liệu mock trong bản chạy thật. Nếu bất kỳ màn hình nào dùng dữ liệu mẫu (chỉ trong test/Storybook), phải có badge "Dữ liệu mẫu".
- Theme sáng và tối dùng token shadcn; không màu chữ/nền cố định cục bộ.
- Mọi điều khiển truy cập được bằng bàn phím, focus ring nhìn thấy được; ảnh có `alt` mô tả vai trò ("Ảnh gốc", "Kết quả phiên bản Cân bằng").
- 390px: không tràn ngang; lưới phiên bản 1 cột; JSON cuộn trong khung.
- `prefers-reduced-motion`: tắt chuyển động không cần thiết (spinner giữ nguyên là chỉ báo trạng thái).

---

## 6. Xuất JSON phiên

```json
{
  "schema": "iid.session-export/1", "exported_at": "...",
  "app": {"gemini_key_configured": false, "gemini_model": "..."},
  "session": {"id", "title", "status", "options", "created_at", "updated_at", "case_id"},
  "messages": [...], "runs": [{"id","kind","status","duration_ms","result","created_at"}],
  "variants": [...], "treatments": [...], "ratings": [...],
  "artifacts": [{"id","kind","width","height","bytes","run_id"}]
}
```
Không gồm ảnh, đường dẫn đĩa, `thread_id`, `error` thô, username của người khác.

---

## 7. Tiêu chí chấp nhận

### 7.1 Backend (pytest offline, `tests/unit/test_api_*.py`, DB + data dir trong `tmp_path`, case memory tắt trừ test memory)
1. Đăng ký → cookie đặt, `GET /auth/me` 200; đăng xuất → token xóa trong DB, `me` 401.
2. Trùng username (khác hoa thường) → 409; mật khẩu sai / user không tồn tại → 401 cùng message; password không lưu dạng thô, token chỉ lưu hash.
3. Token hết hạn → 401.
4. User B: GET/PATCH/DELETE/answers/render/refine/artifact/export trên phiên của A → 404; danh sách của B không chứa phiên của A; `/workspace/memory` của B không thấy ca của A.
5. Upload: file không phải ảnh / GIF → 415; > giới hạn → 413; ảnh hỏng → 422; ảnh có EXIF Orientation=6 → `original` đã xoay (W/H hoán đổi); grayscale/RGBA → lưu RGB.
6. Luồng đầy đủ rule-based (ảnh tổng hợp nhỏ, 1–2 vòng): tạo → `awaiting_answers` + câu hỏi → answers → `done`, có artifact `final`, variants đúng số lượng, treatment v1, messages theo thứ tự `diagnosis, questions, answers, result`.
7. Thread hết hạn (gọi `end_session` thủ công) → answers 409 `thread_expired`, status giữ `awaiting_answers`; `clarify` → câu hỏi mới, answers thành công.
8. Render phiên bản → treatment `variant` + artifact cùng kích thước ảnh gốc; refine "da hơi vàng" → treatment `refine` với parent đúng, notes không rỗng hoặc `refine_source="none"`; activate đổi `active_treatment_id`.
9. Mở lại `GET sessions/{sid}` sau khi tạo engine/app mới trên cùng data dir → đủ messages, variants, treatments (không gọi pipeline — kiểm bằng patch `run_pipeline`/`answer_session` raise nếu bị gọi).
10. Rating PUT hai lần → một bản ghi được cập nhật; export chứa ratings, sources, không chứa `thread_id`/đường dẫn đĩa.
11. Xóa phiên → thư mục phiên không còn trên đĩa, bản ghi cascade biến mất, artifact URL → 404.
12. Lỗi pipeline (patch raise) → 500 `pipeline_failed`, body không chứa chuỗi exception, run `failed`, phiên không kẹt `running`; request thứ hai khi `running` → 409 `session_busy`.
13. Route cũ: `/api/v1/health` trả đúng body cũ; test cũ vẫn xanh.
14. `ruff check src/ tests/` và `pytest tests/unit/` xanh.

### 7.2 Frontend
1. `npm run typecheck` và `npm run build` thành công; `npm test` (vitest) xanh: auth form (lỗi hiển thị, submit khóa nút), tạo phiên (nút khóa khi chưa có ảnh, lỗi 415 hiện câu tiếng Việt), card kết quả với dữ liệu fixture (nhãn quyết định đúng, nguồn đúng, "Điểm đánh giá ước lượng", lý do đề xuất).
2. Trình duyệt (backend thật, rule-based) ở 1366×768 và 390×844, sáng và tối: không tràn ngang, chữ đọc được, không lỗi console, không request thất bại ngoài các lỗi chủ động kiểm tra (401 trước đăng nhập, 409/415 của kịch bản lỗi).

### 7.3 Kịch bản chấp nhận end-to-end (không API key)
Đăng ký A → đăng nhập → chọn mẫu "Thiếu sáng" → xem chẩn đoán → trả lời câu hỏi → kết quả có 3 phiên bản → chọn một bản → render và tải về (file PNG đúng kích thước gốc) → góp ý "da hơi vàng" → tải lại trang, mở lại phiên: đủ hội thoại, kết quả, phiên bản → đánh giá + xuất JSON → xóa phiên (thư mục trên đĩa mất) → đăng ký B: không thấy phiên của A, truy cập URL phiên/artefact của A → 404.
