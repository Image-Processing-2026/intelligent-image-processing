# Lane `region` — trạng thái (P4a, Module 2)

Nhánh: `feat/analysis-v2-region`. Brief: `docs/analysis_v2/agents/region.md`.
Commit chính: `7f9198c feat(region): add C2 subject mask resolver with shared SAM encoder`.

## Tiêu chí xong

| Tiêu chí | Kết quả |
|---|---|
| Unit offline xanh | Đạt. `pytest tests/unit/ -q`: 1175 passed (1151 trước lane + 24 mới). |
| `ruff check src/ tests/` và `ruff format --check` | Đạt. |
| real_model xanh | Đạt. 4 passed: test cũ `person_penn_fudan_00046` + 3 test mới (xem bên dưới). |
| RSS và thời gian đo lại | Có, xem mục Số đo. |
| Kết thúc bằng `LANE DONE` | Có ở cuối file. |

## Đã làm

1. **Backend, encoder dùng chung** (`src/region_engine/segmentation_backend.py`):
   `_ensure_encoder_locked` gọi `set_image` một lần cho mỗi digest ảnh (blake2b trên bytes + shape);
   `_reset_encoder_locked` xóa digest. `detector.resolve_prompt_instances` đổi sang hai hàm này và
   vẫn reset sau mỗi prompt như cũ, nên `segment_by_prompt` không đổi hành vi.
   Protocol `SegmentationBackend` không đổi.
2. **C2** (`src/region_engine/subject.py`, mới):
   - `box2d_to_xyxy`, `segment_box`, `refine_bbox_mask` (GrabCut trên ảnh thu về `max_side`,
     upscale nhị phân, khóa trong bbox, `create_soft_mask`), `resolve_subject_mask`,
     `SubjectMask`, `MaskSource`, `BoxXYXY`, `Box2D`, `clear_subject_mask_cache`.
   - Chuỗi: exact command (full/top/…) → `face` (MediaPipe; không thấy mặt thì mặt nạ rỗng,
     confidence 0) → có model: DINO theo tên, bỏ detection phủ ≥ 90 % khung, khớp bbox Gemini
     IoU ≥ 0.3 → `dino_sam`; không khớp → `gemini_sam`; không bbox → `dino_sam` không đối chiếu
     (xem `contract-changes.md` mục 2) → không model: `gemini_grabcut` → từ khóa (sky/trời → top,
     ground/đất → bottom) → `full`.
   - Cache LRU 16 mục theo `(cache_key, name, box_2d)`, chỉ bật khi có `cache_key`.
   - Ngưỡng: `DINO_MATCH_IOU = 0.3`, `DINO_MAX_BOX_FRACTION = 0.9` (hằng số module).
   - Confidence cố định (heuristic, chưa hiệu chỉnh): face 0.9, dino_sam = điểm DINO (không bbox:
     tối đa 0.5), gemini_sam 0.6, gemini_grabcut 0.4, quadrant 0.3, full 0.1.
3. **Export** (`src/region_engine/__init__.py`): thêm các tên C2. `capabilities` cũ giữ nguyên;
   bản C2 export là `subject_capabilities` (xem `contract-changes.md` mục 1).
4. **Phụ thuộc và model**:
   - `requirements-semantic.txt`: index CPU của PyTorch, `torch`, `transformers`,
     `mobile_sam @ git+…@f706ad9c4eb7f219c00d9050e46328518ffb65d2`.
   - `scripts/fetch_segmentation_models.py`: tải DINO bằng `huggingface_hub.snapshot_download`
     theo revision ghim, tải `mobile_sam.pt` bằng URL ghim, kiểm sha256, ghi qua file `.part`.
   - `tests/fixtures/prompt_segmentation/assets.json`: điền `repo_id`, `revision`, `files`
     (sha256 `model.safetensors`), `url` và `sha256` của `mobile_sam.pt`.
   - `.gitignore`: thêm `models/segmentation/`.
5. **Test**:
   - `tests/unit/test_region_subject.py` (24 test, offline, backend giả): thứ tự fallback có/không
     model và bbox, loại bbox phủ toàn khung, không khớp → bbox Gemini, cache trúng lần 2,
     `set_image` đúng 1 lần cho nhiều bbox trên cùng ảnh (và chạy lại khi ảnh đổi), lỗi
     backend → GrabCut, khuôn mặt (union, không thấy mặt, MediaPipe thiếu), ảnh grayscale,
     GrabCut IoU ≥ 0.7 trên vật thể rõ, mask float32 [0, 1] đã feather.
   - `tests/integration/test_prompt_segmentation_model.py` (real_model):
     - `test_sky_without_bbox_is_never_a_frame_covering_dino_box` trên `backlit_02` và
       `multi_defect_01`: coverage < 0.9.
     - `test_person_subject_on_group_photo_is_detected` trên `group_small_faces_01`: source
       `dino_sam`, mask không rỗng; sau đó gọi `sky` trên cùng ảnh để kiểm tra tái dùng encoder.
     - Đường dẫn model ưu tiên biến môi trường `REGION_DINO_MODEL_PATH` /
       `REGION_MOBILE_SAM_CHECKPOINT` (worktree không có `models/segmentation/`).

## Kết quả real_model

| Ca | Kết quả | Ghi chú |
|---|---|---|
| `backlit_02`, "sky", không bbox | `dino_sam`, conf 0.41, coverage < 0.9 | DINO trả vùng trời không phủ cả khung |
| `multi_defect_01`, "sky", không bbox | `quadrant` (top), conf 0.30 | DINO không cho detection hợp lệ, rơi về quadrant |
| `group_small_faces_01`, "person", không bbox | `dino_sam`, coverage 0.062 | mask không rỗng |
| `group_small_faces_01`, "sky" sau đó | `quadrant` | DINO không thấy trời, encoder không chạy lại |
| Penn-Fudan `FudanPed00046`, "person" (test cũ) | pass | |

## Số đo (máy: Ryzen 7 8845HS, CPU, torch 2.12.0+cpu, transformers 5.17.0, mobile_sam @ f706ad9c)

Đo ngoài repo bằng `python -I` trên `backlit_02` (1600 × 902):

| Bước | Thời gian | Ghi chú |
|---|---|---|
| Lần gọi đầu, gồm nạp model và DINO | 34.0 s | tính cả nạp torch và DINO lần đầu |
| Subject có bbox Gemini (DINO + SAM encoder + decoder) | 6.8 s | model đã nạp |
| `segment_box` khi encoder đã có ảnh | 0.081 s | encoder dùng chung, không chạy lại |
| GrabCut 1024 px trên bbox lớn, `backlit_02` | 2.5 s | coverage 0.45 |
| GrabCut 1024 px, `group_small_faces_01` (1200 × 1600), bbox giữa ảnh | 9.2 s | coverage 0.14 |

RSS (psutil): 44 MB ban đầu → 2666 MB sau lần gọi đầu có nạp model → 2716 MB cuối.
Plan ghi nạp model ~1.2 GB và đỉnh ~3.2 GB; số này gồm cả torch và ảnh 1600 px nên cao hơn.

Thời gian trong pytest (mỗi test nạp lại model vì `reset_segmentation_backend`): 8–11 s mỗi test.

## Thay đổi cần người dùng gửi Person 2 review

- Module 2 chỉ thêm API: `resolve_subject_mask`, `segment_box`, `refine_bbox_mask`,
  `box2d_to_xyxy`, `SubjectMask`, `clear_subject_mask_cache`, `subject_capabilities`.
- Hành vi cũ không đổi: `segment_by_prompt`, `resolve_prompt_instances`, `capabilities`
  (controller), `detect_faces`. Chỉ thêm cách dùng chung encoder SAM (xem mục 1 ở trên).
- Test real_model đọc đường dẫn model từ biến môi trường trước khi dùng `models/segmentation/`.

## Hạn chế đã biết

- GrabCut chỉ tách được khi vật thể có kết cấu. Trên ảnh phẳng hoặc ô kẻ quá lớn, GrabCut có thể
  trả mặt nạ rỗng; `resolve_subject_mask` khi đó chuyển xuống quadrant/full (đã kiểm chứng bằng
  thử nghiệm: ảnh phẳng 1200 × 1600 ra mask 0 ở `max_side=256`).
- Chất lượng GrabCut trên ô kẻ 64 px đã thu nhỏ: IoU khoảng 0.35–0.5 (đo nhanh, không phải test).
- `UNANCHORED_DINO_CONFIDENCE`, các confidence cố định và ngưỡng `DINO_*` chưa được hiệu chỉnh
  trên benchmark. Việc này thuộc lane `calib`/`agent` khi đo.
- `scripts/prepare_prompt_segmentation_assets.py` chưa xử lý được mục `snapshot` (không có `url`).
  Dùng `scripts/fetch_segmentation_models.py` cho DINO. Chưa chạy script này với mạng thật
  (tải ~690 MB); đã kiểm sha256 của hai file có sẵn trong repo chính và HEAD được URL MobileSAM
  (200 OK) và revision DINO trên Hugging Face (ETag khớp sha256).
- `capabilities()` của C2 đặt tên khác với export hiện có: xem `contract-changes.md` mục 1.

## Việc còn lại

- Lead quyết định mục 1 và mục 2 trong `docs/analysis_v2/contract-changes.md`.
- Lane `agent` nối `resolve_subject_mask` vào `executor.resolve_region_mask` và `perceive`;
  lane `region` không sửa hai file này.
- Cập nhật `docs/interfaces.md` §3.2 theo C2 (lead, P8).

## Commit

- `7f9198c` — `feat(region): add C2 subject mask resolver with shared SAM encoder` (code, test, model
  pins, contract-changes).
- Status này được commit sau đó (xem `git log feat/analysis-v2-region`).

LANE DONE
