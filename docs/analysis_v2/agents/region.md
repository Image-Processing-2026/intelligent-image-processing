# Lane `region` — P4a: phân vùng SAM-first (Module 2)

Nhánh `feat/analysis-v2-region`. Đọc trước: `CLAUDE.md`, `docs/plan_analysis_v2.md` §Phase 4 và
§7 D1 (số đo thật), `docs/analysis_v2/contracts.md` §C2, `docs/analysis_v2/orchestration.md`.

Module 2 thuộc Person 2 (Le Viet Xuan). Giữ nguyên hành vi API hiện có (`segment_by_prompt`,
`detect_faces`, …); chỉ thêm và tối ưu. Ghi rõ thay đổi trong `docs/analysis_v2/status/region.md`
để người dùng gửi Person 2 review.

## Môi trường
Model thật có sẵn qua biến môi trường `REGION_DINO_MODEL_PATH`, `REGION_MOBILE_SAM_CHECKPOINT`
(trỏ về repo chính). Gói `mobile_sam` đã cài. Fixture Penn-Fudan đã copy vào worktree.
Test real-model: `python -m pytest -o addopts= -m real_model tests/integration/test_prompt_segmentation_model.py -q`.

## Việc
1. Backend: tách `set_image` khỏi từng prompt — encoder SAM chạy một lần cho mỗi ảnh (cache theo
   id/hash ảnh trong backend), decoder chạy cho từng bbox.
2. Cài C2: `capabilities`, `box2d_to_xyxy`, `segment_box`, `refine_bbox_mask` (GrabCut trên ảnh
   thu về `max_side`, upscale mask, feather), `resolve_subject_mask` đúng thứ tự fallback,
   `SubjectMask`, cache, `clear_subject_mask_cache`. Mọi mask qua `mask_utils.create_soft_mask`.
3. DINO chạy riêng từng tên vùng (không gộp prompt). Loại bbox phủ ≥ 90 % khung. Khớp với bbox
   Gemini bằng IoU ≥ 0.3.
4. Dependency: `requirements-semantic.txt` (torch CPU index, transformers, `mobile_sam @ git+…@f706ad9c4eb7f219c00d9050e46328518ffb65d2`);
   `scripts/fetch_segmentation_models.py` tải DINO `IDEA-Research/grounding-dino-tiny` revision
   `a2bb814dd30d776dcf7e30523b00659f4f141c71` và `mobile_sam.pt` theo commit trên, kiểm sha256
   (`model.safetensors` = `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`,
   `mobile_sam.pt` = `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`);
   điền hash/revision vào `tests/fixtures/prompt_segmentation/assets.json`; thêm
   `models/segmentation/` vào `.gitignore`.
5. Test:
   - Unit offline với backend giả: thứ tự fallback (có/không model, có/không bbox), loại bbox toàn
     khung, không khớp → bbox Gemini, cache trúng lần 2, encoder SAM gọi đúng 1 lần cho nhiều bbox,
     mask hợp lệ theo `executor._validate_mask`.
   - GrabCut: vật thể rõ trên nền synthetic → IoU ≥ 0.7.
   - `real_model` mới: trên `data/real/benchmark/` — `backlit_02` và `multi_defect_01` khi
     `resolve_subject_mask(..., "sky", box_2d=None)` với DINO không được trả vùng "sky" lớn
     (≥ 90 % khung); mask người ở `group_small_faces_01` không rỗng. Đo thời gian, ghi vào status.

## Tiêu chí xong
Unit + real_model xanh; RSS và thời gian đo lại ghi trong status; `docs/analysis_v2/status/region.md`
kết thúc bằng `LANE DONE`.
