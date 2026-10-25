# Đề xuất đổi hợp đồng (chờ lead quyết định)

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

## 2. C2: hành vi khi có model nhưng không có bbox (quyết định của lane `region`)

- `contracts.md` và `plan_analysis_v2.md` §Phase 4 ghi "không bbox → quadrant → full".
- Lane `region` làm thêm: có model và `box_2d=None` thì DINO vẫn chạy theo tên vùng, bỏ detection
  phủ ≥ 90 % khung, lấy detection có điểm cao nhất rồi chạy MobileSAM; `confidence` bị chặn ở
  `UNANCHORED_DINO_CONFIDENCE = 0.5`. Không có detection nào thì rơi về quadrant/full.
- Lý do: `region.md` yêu cầu test real_model `resolve_subject_mask(..., "sky", box_2d=None)` phải
  chạy DINO và không được trả vùng "sky" ≥ 90 %.
- Lead cần xác nhận hoặc yêu cầu bỏ nhánh này (chỉ quadrant/full khi không có bbox).

## 3. Backend: encoder SAM dùng chung (không đổi chữ ký Protocol)

- `segmentation_backend.py` thêm `_ensure_encoder_locked` và `_reset_encoder_locked`: gọi
  `set_image` một lần cho mỗi digest ảnh; `reset_image` xóa digest. `SegmentationBackend` Protocol
  không đổi. Fake backend cũ vẫn chạy (test `test_segment_by_prompt.py` vẫn xanh).
