# Lane `meta` — P3: metadata EXIF (S0)

## Đã làm
- `src/analyzer_evaluator/metadata.py` (chỉ Pillow): `read_metadata` không raise (bytes hỏng →
  mọi trường None; mỗi phần EXIF / bảng lượng tử / ICC được bảo vệ riêng). `estimate_jpeg_quality`
  khớp bảng luma IJG tỉ lệ theo q. `metadata_priors` cho 5 loại prior theo ngưỡng trong brief,
  `strength` tăng dần và kẹp trong [0, 1].
- Plumbing `image_metadata`: `initial_doctor_state` / `run_pipeline` nhận tham số mặc định None;
  `DoctorState["image_metadata"]`; `/api/v1/diagnose` và `/api/v1/process` đọc bytes upload bằng
  `read_metadata`, trả về trường `image_metadata` (tùy chọn) trong response.
- Perceive, planner và VLM chưa đọc metadata (đúng phạm vi; lane `agent` làm).

## Tiêu chí
- [x] `read_metadata` không raise với bytes rỗng, hỏng, cắt cụt, PNG không metadata.
- [x] Ước lượng q JPEG ∈ {30, 60, 90}: sai ≤ 10 (thực tế khớp đúng cả ba).
- [x] Priors đúng ngưỡng: ISO 1600, phơi chậm hơn 1/35mm (đúng ngưỡng không sinh prior), f ≤ 2.8, q < 60, có Software.
- [x] API test: `/process` đưa metadata vào `run_pipeline` và vào state (qua response); `/diagnose` trả metadata; ảnh không EXIF vẫn chạy.
- [x] Không có metadata thì luồng như cũ (toàn bộ test cũ vẫn xanh).
- [x] `ruff check src/ tests/` sạch; `pytest tests/unit/` → **1180 passed** (baseline 1151 + 29 mới), không gọi Gemini.

## Số đo
- Test mới: 29 (`tests/unit/test_metadata.py`, `tests/unit/test_api_metadata.py`), chạy offline ~27 s.

## Chưa làm / cần lead quyết định (chi tiết trong `docs/analysis_v2/contract-changes.md`)
1. **Phiên tương tác** (`POST /api/v1/sessions`): `session.start_session` chưa nhận `image_metadata`.
   `session.py` không thuộc phạm vi lane.
2. **Gradio** (`frontend/app.py`): input là `gr.Image(type="numpy")` nên không có bytes để đọc EXIF.
   Đổi sang `type="filepath"` kéo theo nhiều hàm UI và sự kiện; chưa làm.

## Commit
- `15f6642` feat(analyzer): read image metadata and derive diagnostic priors
- `169d5b6` feat(api): pass image_metadata from uploads into the pipeline state

Chưa push, chưa mở PR, chưa merge.

LANE DONE
