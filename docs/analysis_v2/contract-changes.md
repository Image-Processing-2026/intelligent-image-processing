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

## 5. [calib] C4: tham số tùy chọn `long_side` cho `severity_from_measurement` (2026-10-25)

- **Vấn đề:** severity mờ phải quy σ defocus (px) về cạnh dài tham chiếu của bộ hiệu chỉnh
  (1024 px), nếu không ảnh 12 MP nét ở mức pixel sẽ bị báo mờ. `MeasurementReport` có
  `width/height`, nhưng `RegionStats` không có kích thước ảnh.
- **Đã làm (tương thích ngược):** `severity_from_measurement(defect_type, stats, *, long_side=None)`.
  Báo cáo toàn ảnh tự lấy `max(width, height)`; `RegionStats` không truyền `long_side` thì không quy
  đổi (giữ nguyên hành vi theo chữ ký C4 gốc). Lời gọi hai tham số như C4 vẫn chạy.
- **Đề xuất cho lead:** ghi tham số này vào `contracts.md` §C4; lane `agent`/`fusion` truyền
  `long_side` của ảnh gốc khi tính severity vùng. Phương án khác: thêm trường có default
  `image_long_side` vào `RegionStats` (schema của lane `measure`).
- **Quyết định (người dùng duyệt, 2026-10-25): chấp nhận.** Đã ghi vào `contracts.md` §C4; `fusion`
  truyền `long_side` của ảnh gốc khi tính severity vùng.

## 6. [calib] Thông báo (không đổi chữ ký)

1. **Key mới trong `MeasurementReport.sharpness`** (C1, chỉ thêm key): `motion_anisotropy`,
   `motion_extent_sigma` (px), `motion_blur_angle_deg` (hướng chuyển động 0–180). Không có trong
   `MAP_KEYS`, nên `RegionStats` không có và `motion_blur` theo vùng trả `(0, 0.0)`.
2. **`TechnicalMetrics` thêm `defect_severity: Dict[str, int]` và `calibration_version: str`**
   (có default). `perception.perceive` đưa nguyên dict metrics vào prompt, nên VLM giờ thấy cả
   severity đã hiệu chỉnh. Lane `agent` lưu ý khi viết prompt perceive v2.
3. **Ngưỡng còn sót ngoài calibration:** `perception._brightness_level` và `_measured_defects`
   vẫn import `_BRIGHTNESS_LOW/HIGH`, `_*_CLIP_*` từ `analyzer.py` (để nguyên vì `perception.py`
   thuộc lane `agent`). Đề xuất lane `fusion` (P6) chuyển sang `severity_from_measurement` trên
   `RegionStats` (`underexposed`/`overexposed` đo được theo vùng qua `exposure.mean_l`).
- **Ghi nhận (lead):** mục 6.3 giao lane `fusion` (P6). Checkpoint sau calib cho thấy 11/17 FP
  còn lại là `backlit_subject` từ các ngưỡng này (`checkpoint_calib.md`).

## 7. [agent] Sửa nhỏ ngoài phạm vi lane (cần lead duyệt)

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

- **Quyết định (lead đã review, người dùng duyệt, 2026-10-25): chấp nhận.** Mọi tham số mới tùy chọn, có default; plumbing đúng
  brief. `feather_radius` của action bị bỏ qua khi dùng resolver chung (mask C2 feather 15 px): chấp
  nhận, ghi cho `fusion`.

## 8. [fusion] Sửa ngoài phạm vi lane và thay đổi hành vi (cần lead duyệt)

Phạm vi lane `fusion` (`orchestration.md`): `fusion.py`, `measurement/diff.py`, `state.py`,
`perception.py`, `planner.py`, `intent.py`, `graph.py` (`decide_node`), knowledge cards,
`SYSTEM_PROMPT`. Ngoài danh sách đó:

1. `src/agent/graph.py::plan_treatment` (không phải `decide_node`): gọi
   `planner.apply_contraindications(plan, fusion.plan_contraindications(diagnosis))` sau preserve
   guard (brief mục 5: planner đọc `contraindications`).
2. `src/agent/knowledge/retriever.py::match_cards`: lấy mọi defect severity ≥ 1 thay cho
   `actionable_defects`, vì `actionable_defects` giờ bỏ lỗi `fixable=False`; card của lỗi không sửa
   được (`cards/limits.yaml`, chỉ tư vấn, không recipe) vẫn được truy xuất cho mục "Tránh".
3. `src/agent/vlm_diagnostician.py`: ngoài `SYSTEM_PROMPT`, thêm `fusion_notes` vào
   `_PLAN_DIAGNOSIS_EXCLUDE` (nhật ký hợp nhất không đưa vào prompt Plan).
4. `tests/conftest.py`: fixture autouse `_no_verify_call` đặt `perception.VERIFY_ENABLED = False`
   (test cũ mock đúng số phản hồi Perceive/Plan; verify là một lời gọi thêm) và xóa env
   `FUSION_WEAK_REJECTED`. Test fusion tự bật lại verify.
5. `scripts/benchmark_analysis.py` (lane `bench`): config `fused`, `fused_noverify`, `fused_n3`,
   `fused_n3_weak`; `vlm_only` giờ tắt hợp nhất (`FUSION_ENABLED = False`); `--set holdout`
   (`data/holdout_manifest.json`, cùng định dạng manifest thật, ảnh ở `image_dir` của manifest hoặc
   `data/real/holdout`); `--vlm-cache DIR` (cache phản hồi Gemini theo băm yêu cầu, chạy lại không
   gọi mạng); `results.json` ghi `ask_user`, `fusion_notes`, `gemini_cache_hits`.
6. Test của lane khác sửa theo hành vi mới: `test_agent_knowledge.py` và `test_agent_perception.py`
   dùng `lens_flare` làm loại lỗi không hợp lệ (`haze` giờ hợp lệ, D2); test `_measured_defects`
   cũ bỏ (hàm đã bỏ, thay bằng bảng fusion, có test riêng); hai test backlit offline dùng ảnh
   640 px (bản đồ đo tile 64 px không tách được chủ thể trên ảnh 100 px); hai test gộp
   self-consistency tắt hợp nhất (`FUSION_ENABLED = False`) vì chỉ kiểm bước gộp.

Thay đổi hành vi đáng chú ý (không đổi chữ ký):
- `DiagnosisReport.actionable_defects` bỏ lỗi `fixable=False` (D2: không lập action). Ảnh chỉ có
  lỗi không sửa được → plan rỗng → SHIP.
- `decide_node`: tác dụng phụ severity ≥ 2 (P7) → suy thoái → rollback (chỉ ảnh thực).
- Perceive vòng ≥ 2 không gọi VLM khi mọi lỗi đã xác nhận đều đo được (P7).
- Trường mới có default: `DiagnosisReport.fusion_notes`, `DiagnosisReport.diff`,
  `HistoryItem.side_effects`. `DEFECT_TYPES` thêm `motion_blur`, `jpeg_artifacts`, `haze`,
  `highlight_clipping` (D2); lead cập nhật `docs/interfaces.md` (P8).
- Brief mục 14: giữ việc executor bỏ `feather_radius` của action khi dùng resolver chung (mask dùng
  chung cho đo và xử lý, C2 feather 15 px); không sửa.

- **Quyết định (lead đã review, 2026-10-26): chấp nhận cả 6 điểm** và các thay đổi hành vi;
  chi tiết ở `progress.md` mục "Review `fusion` trước merge". Merge `a20d674`.

## 9. [color] Key mới trong `MeasurementReport.color` (chỉ thêm key, 2026-10-26)

Đánh số 9 vì nhánh `fusion` đã dùng mục 8.

1. **`illuminant_method`** (Estimate, giá trị 0/1/2, confidence 1.0, `n_samples` = số pixel ước
   lượng): đường ước lượng nguồn sáng trong `measurement/color.py`.
   - 0 = pixel trung tính chuẩn (C\* < 12), như trước.
   - 1 = cổng C\* nới (18 → 24 → 30) vì cổng chuẩn có < `MIN_NEUTRAL_PIXELS` pixel. Confidence các
     key `illuminant_*` nhân thêm 12 / cổng.
   - 2 = mọi pixel không cháy (chỉ Shades-of-Gray + Gray-Edge, không lặp chọn trung tính).
     Confidence nhân thêm `ALL_PIXELS_CONFIDENCE` = 0.2.
   Cả hai đường dự phòng cần ảnh có cạnh (≥ 100 pixel có gradient); một mặt màu phẳng vẫn không có
   ước lượng (như trước), có note `no image structure for a fallback illuminant estimate`.
2. **`illuminant_gate_chroma`** (C\* của cổng đã dùng: 12, 18, 24 hoặc 30): chỉ có ở đường 0 và 1.
3. Không có trong `MAP_KEYS`; `RegionStats` không đổi. `calibration.py` không đọc hai key này
   (severity vẫn từ `illuminant_r/g/b`); confidence thấp đã đi qua `_illuminant` của
   `calibration.py` vào confidence severity.
- **Đề xuất cho lead:** ghi hai key vào bảng §C1 của `contracts.md`. Lane `fusion`: severity ám màu
  có `illuminant_method ≥ 1` (hoặc confidence thấp) nên đi đường verify, không confirm thẳng.
- **Quyết định (lead đã review, 2026-10-26): chấp nhận.** Chỉ thêm key, không đổi `MAP_KEYS`,
  `RegionStats` hay chữ ký; lead đã ghi hai key vào bảng §C1. Đề xuất cho fusion: xem
  `progress.md` mục "Merge `color`" (chưa làm, cần người dùng duyệt).
