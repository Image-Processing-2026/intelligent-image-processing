# Lane `meta` — P3: metadata EXIF (S0)

Nhánh `feat/analysis-v2-meta`. Đọc trước: `CLAUDE.md`, `docs/plan_analysis_v2.md` §Phase 3,
`docs/analysis_v2/contracts.md` §C3, `docs/analysis_v2/orchestration.md`.

## Việc
1. `src/analyzer_evaluator/metadata.py` theo C3, chỉ dùng Pillow (`getexif()`, IFD Exif, bảng
   `quantization` của JPEG → ước lượng chất lượng kiểu IJG). `read_metadata` không bao giờ raise.
2. `metadata_priors`: ISO ≥ 1600 → noise; màn trập chậm hơn 1/(tiêu cự 35 mm) → motion_blur;
   f ≤ 2.8 → bokeh; JPEG q < 60 → no_sharpen; có `Software` → edited. `strength` tăng dần theo mức.
3. Plumbing: `run_pipeline(..., image_metadata=None)`, `DoctorState["image_metadata"]`;
   API `/diagnose`, `/process` (và các route nhận upload khác) đọc bytes upload → `read_metadata`;
   Gradio đọc từ file path. Chỉ truyền và lưu — chưa dùng trong perceive (lane `agent` làm).
   Không có metadata thì mọi thứ chạy như cũ.
4. Test offline: JPEG tạo trong test với EXIF giả (Pillow ghi EXIF), JPEG không EXIF, PNG, bytes
   hỏng; ước lượng chất lượng JPEG q ∈ {30, 60, 90} sai ≤ 10; priors đúng ngưỡng; API test hiện có
   vẫn xanh; một test API xác nhận metadata đến được state.

## Lưu ý phạm vi
Bạn là lane duy nhất ở wave 1 được sửa `src/agent/state.py`, `graph.py` và `src/api/` — chỉ cho
plumbing `image_metadata`, giữ diff nhỏ.

## Tiêu chí xong
Test xanh; `docs/analysis_v2/status/meta.md` kết thúc bằng `LANE DONE`.
