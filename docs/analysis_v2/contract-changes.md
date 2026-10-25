# Đề xuất đổi hợp đồng

Lane ghi đề xuất ở đây; lead quyết định cùng người dùng và ghi **Quyết định** dưới mỗi mục.

## 1. C2: tên `capabilities` trùng export hiện có (lane `region`, 2026-10-25)

- **Vấn đề:** `contracts.md` §C2 yêu cầu `region_engine.capabilities() -> Dict[str, bool]`
  với khóa `face` và `semantic_models`. Trong repo, `src/region_engine/controller.py::capabilities()`
  đã được export từ `region_engine/__init__.py` với cấu trúc lồng nhau (`geometric`, `face`,
  `face_oval`, `semantic`…). `scripts/evaluate_person_image.py` và `tests/unit/test_region_controller.py`
  đang dùng bản đó.
- **Đã làm (không phá API cũ):** cài bản C2 trong `src/region_engine/subject.py::capabilities`,
  export từ package dưới tên `subject_capabilities`. `region_engine.capabilities` giữ nguyên.
- **Lựa chọn cho lead:**
  - (A) Giữ như hiện tại. Lane `agent` import `subject_capabilities` (khuyến nghị, không đổi
    code đang chạy).
  - (B) Đổi tên bản C2 thành `semantic_capabilities` trong `contracts.md` và `subject.py`.
  - (C) Đổi `controller.capabilities()` thành phẳng và sửa hai nơi dùng. Không khuyến nghị:
    thay đổi hành vi API hiện có.
- **Quyết định (người dùng duyệt, 2026-10-25): (A).** `contracts.md` §C2 đã sửa theo
  `subject_capabilities`; lane `agent` import tên này.

## 2. C2: hành vi khi có model nhưng không có bbox (quyết định của lane `region`)

- `contracts.md` và `plan_analysis_v2.md` §Phase 4 ghi "không bbox → quadrant → full".
- Lane `region` làm thêm: có model và `box_2d=None` thì DINO vẫn chạy theo tên vùng, bỏ detection
  phủ ≥ 90 % khung, lấy detection có điểm cao nhất rồi chạy MobileSAM; `confidence` bị chặn ở
  `UNANCHORED_DINO_CONFIDENCE = 0.5`. Không có detection nào thì rơi về quadrant/full.
- Lý do: `region.md` yêu cầu test real_model `resolve_subject_mask(..., "sky", box_2d=None)` phải
  chạy DINO và không được trả vùng "sky" ≥ 90 %.
- **Quyết định (người dùng duyệt, 2026-10-25): chấp nhận.** `contracts.md` §C2 và
  `plan_analysis_v2.md` §Phase 4 đã sửa theo hành vi này.

## 3. Backend: encoder SAM dùng chung (không đổi chữ ký Protocol)

- `segmentation_backend.py` thêm `_ensure_encoder_locked` và `_reset_encoder_locked`: gọi
  `set_image` một lần cho mỗi digest ảnh; `reset_image` xóa digest. `SegmentationBackend` Protocol
  không đổi. Fake backend cũ vẫn chạy (test `test_segment_by_prompt.py` vẫn xanh).
- **Quyết định:** chấp nhận (không đổi hợp đồng, nằm trong phạm vi lane `region`).

## 4. [meta] Metadata: nối `image_metadata` vào luồng phiên tương tác và giao diện Gradio

1. `src/agent/session.py::start_session` gọi `initial_doctor_state` không truyền `image_metadata`.
   Vì vậy `POST /api/v1/sessions` (`src/api/workspace.py::create_session`) chưa đưa metadata vào
   state. Cần thêm `image_metadata: Optional[ImageMetadata] = None` cho `start_session` rồi truyền
   xuống; route `create_session` đọc bytes upload bằng `read_metadata`. `session.py` không thuộc
   phạm vi lane `meta`.
   - **Quyết định (người dùng duyệt, 2026-10-25):** giao cho lane `agent` (đã ghi vào
     `agents/agent.md`, phạm vi thêm `session.py` và route `create_session`).
2. `frontend/app.py`: `gr.Image(type="numpy")` không giữ bytes file, nên không đọc được EXIF.
   Muốn đọc EXIF phải đổi input sang `type="filepath"`, kéo theo `process_job`, `render_chosen_variant`,
   `image.change`, và mẫu ảnh (đang gán đường dẫn).
   - **Quyết định (người dùng duyệt, 2026-10-25): HOÃN.** Giữ `type="numpy"`; UI Gradio chưa có
     metadata. Ghi trong `progress.md` mục Hoãn.
3. `src/agent/perception.py` và `src/agent/planner.py` chưa đọc `image_metadata`. Đó là việc của lane
   `agent` (theo brief `meta.md`).

## 5. [agent] Sửa nhỏ ngoài phạm vi lane (cần lead duyệt)

Lane `agent` sửa các file sau ngoài danh sách sở hữu, vì brief yêu cầu hoặc vì cần plumbing tối
thiểu. Không đổi chữ ký hiện có: mọi tham số mới đều tùy chọn, có default.

1. `src/agent/graph.py` (2 chỗ):
   - `perceive_node` truyền `metadata=state.get("image_metadata")` vào `perceive` (brief mục 7:
     prior EXIF vào prompt).
   - `process_node` truyền `subject_boxes=diagnosis.subject_boxes` và
     `cache_key=image_cache_key(original_image)` vào `execute_plan` (brief mục 3: executor dùng
     chung resolver và cache mask của Perceive).
2. `src/api/routes.py`: `/diagnose` truyền metadata đã đọc vào `perceive`.
3. `src/api/main.py` (lifespan) và `frontend/app.py` (`__main__`): gọi
   `perception.preload_region_models()` (brief mục 5). Nạp ở luồng nền, không chặn khởi động.
4. `tests/conftest.py`: fixture autouse đặt `perception.PERCEIVE_SAMPLES = 1` (các test cũ mock một
   phản hồi Perceive mỗi vòng) và xóa cache mask chủ thể của `region_engine` trước/sau mỗi test.
   Test self-consistency tự đặt lại 3.
5. Thêm trường có default (được phép theo quy tắc): `DiagnosisReport.subject_boxes`
   (bbox Gemini theo tên subject, không đưa vào prompt Plan) và `DiagnosisReport.measurement_maps`
   (`exclude=True`, không ra JSON/HistoryItem/API).
6. Chữ ký mở rộng (tham số tùy chọn): `perceive(..., metadata=None, cache_key=None)`,
   `execute_plan(image, plan, subject_boxes=None, cache_key=None)`,
   `resolve_region_mask(..., box_2d=None, cache_key=None)`,
   `call_gemini_json(..., temperature=None)`.

- **Quyết định:** chờ lead.
