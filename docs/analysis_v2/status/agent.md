# Lane `agent` — trạng thái (P4b + P5, wave 2)

Nhánh `feat/analysis-v2-agent` (worktree `iip-wt/agent`), tạo từ `feat/analysis-v2` @ `4571146`.
Brief: `docs/analysis_v2/agents/agent.md`. Hợp đồng: `contracts.md` §C1 §C2 §C3 §C6.

## Đã làm

### P4b — nối đo/vùng vào Perceive
1. **Schema/prompt Perceive** (`perception.py`): `subjects: [{name, box_2d}]` (0–1000, C6). Parse chịu
   cả định dạng cũ (danh sách tên); bbox sai (thiếu tọa độ, ymin ≥ ymax, ngoài 0–1000) bị bỏ.
   `DiagnosisReport.subjects` vẫn là `List[str]`; bbox nằm ở trường mới `subject_boxes`.
   Prompt Plan không nhận `subject_boxes`, `measurement`, `region_stats`, `rejected`, `ask_user`
   (`vlm_diagnostician._PLAN_DIAGNOSIS_EXCLUDE`); `SYSTEM_PROMPT` vẫn cấm bbox.
2. **`measure_regions`**: mask từ `resolve_subject_mask` (bbox Perceive nếu có), `RegionStats` =
   `aggregate` + `mask_source`/`mask_confidence`. Luôn có `full` và `background` (phần bù của hợp
   các subject, `mask_source="complement"`). `RegionMetrics` cũ vẫn tính, trên ảnh thu nhỏ ≤ 1024 px,
   nên `_measured_defects`, UI và API không đổi. Subject rơi về `full` (không định vị được) hoặc
   khuôn mặt rỗng thì bị bỏ, như D1.
   `cache_key` = `subject_cache_key(image_cache_key(ảnh gốc), shape)`. Vòng 2–3 và executor trúng
   cache. Shape nằm trong khóa để ảnh preview của phiên bản phong cách không lấy nhầm mask full-res.
3. **Executor**: vùng ngữ nghĩa có bbox trong `diagnosis.subject_boxes` → `resolve_subject_mask`
   (cùng cache). Resolver trả `full` → bỏ action (D1). Face/full/quadrant, hoặc vùng không có
   bbox → đường controller cũ, không đổi. `resolve_region_mask` nhận thêm `box_2d`, `cache_key`.
4. **`DiagnosisReport`**: `measurement` (MeasurementReport), `region_stats`, `rejected`, `ask_user`,
   `subject_boxes` đều được điền hoặc có default. `measurement_maps` (`exclude=True`) giữ bản đồ tile
   trong tiến trình, không ra JSON, HistoryItem hay API (có test). `Defect` có thêm `confidence`,
   `evidence_items`, `fixable`, `contraindications` (C6).
5. **Nạp sẵn model**: `perception.preload_region_models()` khi
   `subject_capabilities()["semantic_models"]` (luồng nền: một lần `resolve_subject_mask(...,
   use_semantic=True)` trên ảnh 64 px). Gọi trong `lifespan` của API và `__main__` của Gradio.
- **Metadata cho phiên tương tác**: `start_session(..., image_metadata=None)` → `initial_doctor_state`.
  Route `create_session` đọc bytes upload (hoặc ảnh mẫu) bằng `read_metadata`. `/diagnose` cũng
  truyền metadata vào `perceive`. Gradio không đổi (theo quyết định hoãn).
- **Skin locus** (lệch 6 của `measure`): `skin_locus_stats` trên mask mặt (YCrCb, Cr 133–173,
  Cb 77–127, bỏ pixel tối < 30 và cháy ≥ 250). Kết quả ghi vào `RegionStats["face"].values` với các
  key `color.skin_locus_fraction`, `color.skin_cr_mean`, `color.skin_cb_mean`. Ảnh xám không có.
  Chưa sinh defect từ số này: đó là việc của `fusion`.

### P5 — bằng chứng cho VLM
6. **`src/agent/evidence.py`**: `select_evidence(image, measurement, region_masks) ->
   List[EvidenceCrop]` (nhãn A…, `kind`, bbox pixel, crop 512 px ở 100 %, mô tả số đo, vùng).
   Thứ tự chọn: mặt → vùng clip lớn nhất (≥ 5 % pixel) → patch phẳng nhất (gradient sau làm mịn
   σ 1.5, không clip, L* 12–92) → patch σ nhiễu cao nhất (σ ≥ 1.5) → cạnh mạnh nhất trên chủ thể →
   cạnh mạnh nhất trên nền. Crop sau không chồng crop trước quá IoU 0.3. Tối đa 6 crop.
   Chủ thể chưa biết (vòng 1) → khung giữa ảnh, mô tả ghi rõ "chưa rõ chủ thể". Từ vòng 2 dùng
   subject của vòng trước (từ cache). Ảnh có cạnh dài ≤ 768 px thì không cắt crop.
   `build_overlays`: overlay clip (đỏ = cháy, xanh = bệt) và heatmap σ nhiễu theo tile, trên ảnh thu
   nhỏ ≤ 768 px. Chỉ để VLM xem, không đổi pixel đầu ra (ADR-001, có test ảnh vào không đổi).
7. **Prompt Perceive v2**: tóm tắt số đo có cấu trúc (`summarize_measurement`: giá trị + độ tin cậy)
   thay cho dump JSON (vẫn dump JSON khi đo lỗi); thêm số đo vùng mặt, danh sách crop có vị trí,
   mô tả overlay và khối "GỢI Ý TỪ EXIF" (`metadata_priors`, ghi rõ là gợi ý). VLM trả **giả thuyết**
   kèm `crops` dẫn chứng; prompt khuyến khích "không có lỗi". Revision 2 (`ca0d8e1`, sau benchmark
   lần 1): trời hoặc nguồn sáng cháy phía sau chủ thể không phải `overexposed`; mặt sáng đều không
   phải `backlit_subject`.
8. **Self-consistency**: `PERCEIVE_SAMPLES = 3` lần gọi song song, temperature 0.7.
   `merge_samples` gộp: defect (type, region) xuất hiện ≥ ⌈2/3·k⌉ trong k lần hợp lệ thì giữ,
   severity = trung vị, `confidence` = tỉ lệ đồng thuận, `evidence_items` có `self_consistency` và
   `crops`; phần còn lại vào `rejected`. Scene, preserve và subject theo đa số; bbox = trung vị từng
   tọa độ. Lần gọi lỗi bị bỏ; mọi lần lỗi → luật dự phòng (`vlm_fallback`).
   Ngân sách mỗi lần gọi: ảnh toàn khung + ≤ 6 crop + ≤ 2 overlay, cộng ảnh gốc từ vòng 2.
   `_encode_contents` mã hóa ảnh nhỏ trước, phần ngân sách inline còn lại dành cho ảnh toàn khung.
9. **Benchmark** (`scripts/benchmark_analysis.py`): `vlm_only` hiện chạy perceive v2. Thêm
   `vlm_only_n1` (N = 1) và `vlm_only_frame1024` (ảnh toàn khung 1024 px, crop vẫn 100 %).
   `results.json` ghi `confidence` và `rejected_defects`. `measure_only` đã được lead điền sẵn
   (chờ `calibration.py` của lane `calib`), không đổi. `fused` vẫn chờ lane `fusion`.

## Test (offline, không gọi Gemini)
- `tests/unit/test_agent_evidence.py` (12): crop phẳng nằm trong vùng phẳng, crop clip phủ khối cháy,
  crop cạnh theo mask chủ thể/nền, crop mặt đứng đầu và ở giữa mặt, crop nhiễu trúng tile σ cao (bản
  đồ synthetic), nhãn A…, 100 % (`array_equal` với lát cắt gốc), ảnh nhỏ không crop, ảnh xám, overlay.
- `tests/unit/test_agent_perception_v2.py` (25): gộp 3 phản hồi khác nhau; perceive gọi 3 lần với
  temperature 0.7; bỏ lần lỗi; mọi lần lỗi → luật; prompt chứa mọi nhãn crop và prior EXIF; ảnh
  toàn khung full-res hoặc thu nhỏ qua env; prompt Plan không có bbox; resolver nhận bbox từ
  perceive kèm `cache_key`; pipeline rule-based (không key) có `measurement`/`region_stats`, JSON
  không có `measurement_maps`; **cache trúng ở vòng 2 và ở executor** (spy `_resolve` thật, đường
  GrabCut); executor dùng/bỏ resolver chung; skin locus; ngân sách mã hóa; preload; `start_session`
  lưu metadata; route `create_session` đọc EXIF.
- `tests/unit/test_agent_perception.py`: chuyển sang patch `resolve_subject_mask`; test đo trên ảnh
  thu nhỏ đổi thành "mask full-res, RegionMetrics thu nhỏ, RegionStats có full/background".
- `tests/conftest.py`: autouse `PERCEIVE_SAMPLES = 1` và xóa cache mask chủ thể.
- `ruff check src/ tests/` sạch, `ruff format` đã chạy. `pytest tests/unit/ -q`: **1338 passed**
  (trước lane 1301).

## Benchmark real (32 ảnh, 30 chấm, 13 sạch/chủ ý)

Model `gemini-3.5-flash-lite`. Có DINO + MobileSAM (biến môi trường trỏ về repo chính), nên subject
có bbox đi đường `dino_sam`/`gemini_sam`. Kết quả nằm trong `data/real/benchmark_results/agent-v2*`
(gitignored).

| Config | Precision | Recall | FP rate sạch | MAE severity | Preserve recall | Dự đoán / mong đợi | Gọi Gemini | s/ảnh |
|---|---|---|---|---|---|---|---|---|
| v1 baseline (`baseline.md`) | 0.175 | 0.467 | 0.538 | 1.00 | 0.762 | 40 / 15 | 32 | 20.0 |
| v1 vlm_only (`baseline.md`) | 0.235 | 0.533 | 0.615 | 0.88 | 0.714 | 34 / 15 | 32 | 14.4 |
| v2 vlm_only, prompt rev 1 | 0.333 | 0.467 | 0.154 | 1.00 | 0.714 | 21 / 15 | 96 | 19.0 |
| v2 vlm_only_frame1024, rev 1 | 0.353 | 0.400 | 0.154 | 0.83 | 0.762 | 17 / 15 | 96 | 17.1 |
| v2 vlm_only_n1, rev 1 | 0.389 | 0.467 | 0.154 | 0.71 | 0.714 | 18 / 15 | 32 | 15.5 |
| **v2 vlm_only, prompt rev 2 (HEAD)** | **0.583** | **0.467** | **0.000** | 1.14 | 0.619 | 12 / 15 | 96 | 19.3 |

Theo loại (tp/fp/fn), HEAD: noise 4/0/0, underexposed 3/1/2, backlit_subject 0/3/1,
low_contrast 0/1/1, blur 0/0/1, color_cast_cool 0/0/2, color_cast_warm 0/0/1. So với v1 vlm_only:
blur FP 12 → 0, low_contrast FP 6 → 1, backlit_subject FP 2 → 3.

Nhận xét:
- FP trên ảnh sạch giảm từ 0.62 về 0.00 (0/13). Phần lớn nhờ tóm tắt số đo, crop 100 % và lời dặn
  "không có lỗi". Ở rev 1, crop/overlay clip làm VLM báo `overexposed@sky` 8 lần (v1 không báo lần
  nào). Rev 2 bỏ hết các FP này. **Lưu ý: rev 2 được chỉnh trên chính bộ 30 ảnh này, nên số của nó
  lạc quan.** Cần một bộ giữ riêng để xác nhận.
- Recall không đổi (0.47) và vẫn xa mục tiêu 0.85. Chỉ riêng ám màu đã bỏ sót 4/4: VLM không báo
  `color_cast_*` dù số đo có góc lệch nguồn sáng. Còn bỏ sót `underexposed` ở low_light_01,
  noisy_02 và `backlit_subject` ở backlit_02. Đây là phần lane `fusion` (đối chiếu số đo) nên xử lý.
- MAE severity xấu đi (0.88 → 1.14, chỉ trên 7 cặp khớp: mẫu rất nhỏ). Preserve recall giảm
  0.71 → 0.62 vì preserve cần đủ 2/3 đồng thuận.
- Self-consistency (N = 3 so với N = 1, cùng prompt rev 1): FP sạch bằng nhau, recall bằng nhau,
  precision 0.33 so với 0.39, MAE 1.00 so với 0.71. **Trên bộ này N = 3 chưa cho lợi ích đo được**,
  lại tốn 3 lần gọi. Bước gộp đã loại 17–19 giả thuyết (`rejected`), trong đó có ít nhất một lỗi thật
  (`underexposed@full` ở backlit_02, khi mong đợi là `backlit_subject@face`). Đề nghị lead xem lại D3
  khi có `fusion`: giữ N = 3 nhưng đưa `rejected` vào hợp nhất làm giả thuyết yếu, hoặc hạ về N = 1.
- Số lần gọi Gemini của lane: **326** (smoke 6, vlm_only rev 1: 78 + 18 chạy lại 6 ảnh thiếu,
  frame1024 96, n1 32, vlm_only rev 2 96). Không có fallback.

## Câu hỏi mở §7: có cần ảnh toàn khung full-res khi đã có crop?
Đo bằng `vlm_only` so với `vlm_only_frame1024` (cùng prompt rev 1, N = 3): FP sạch bằng nhau (0.154),
recall 0.47 so với 0.40 (khác một defect), MAE 1.00 so với 0.83, preserve 0.71 so với 0.76, thời gian
19.0 so với 17.1 s/ảnh. Chênh lệch nằm trong mức dao động của mẫu 15 defect có temperature 0.7.
Ảnh benchmark đã ≤ 1600 px, nên thu về 1024 chỉ giảm ~1.5 lần. Phép đo này **chưa trả lời được**
câu hỏi cho ảnh 12 MP.
**Đề nghị:** giữ full-res làm mặc định (`FULL_FRAME_MAX_SIDE = None`; crop và overlay được mã hóa trước nên ảnh toàn khung
vẫn giữ gần hết ngân sách inline). Khi có ảnh full-res, đo lại bằng
`PERCEIVE_FULL_FRAME_MAX_SIDE=1024` trên `data/real/benchmark_fullres/` (cần thêm cờ đọc
`fullres_file` vào script benchmark, chưa làm).

## Lệch khỏi brief / cần lead quyết định
- Sửa nhỏ ngoài phạm vi lane: `graph.py` (truyền metadata và bbox/cache key), `api/routes.py`,
  `api/main.py`, `frontend/app.py` (preload), `tests/conftest.py`. Chi tiết và chữ ký mở rộng ghi ở
  `contract-changes.md` mục 5, đang chờ duyệt.
- Executor bỏ qua `feather_radius` của action khi dùng resolver chung (mask luôn feather 15 px theo
  C2), vì mask cache dùng chung cho đo và xử lý.
- 6 ảnh `intentional_style_05..10` chưa có trong `data/real/benchmark` của worktree này (script khởi
  động copy từ repo chính, nơi chưa có các ảnh này). Lane đã copy từ worktree `bench`. Lead nên tải
  về repo chính bằng `data/download_real_benchmark.py`.
- Chưa đo thời gian Perceive v2 trên ảnh 12 MP. Ước lượng: đo C1 ~8.5 s, cộng DINO ~4.5 s/subject
  (chỉ vòng 1), cộng 3 lần gọi Gemini song song. Benchmark ≤ 1600 px: 19 s/ảnh.

## Commit
```
ca0d8e1 fix(agent): tell perceive that clipped backgrounds and lit faces behind the sun are not defects
afca40a docs(workflow): record agent lane edits outside its owned files
0da51b6 feat(workflow): add perceive v2 benchmark configs for self-consistency and full-frame size
04ab635 feat(agent): perceive v2 with subject boxes, region stats, evidence prompt and self-consistency
7ee3b44 feat(agent): select 100% evidence crops and overlays from local measurement maps
```
Status này là commit cuối. Không push, không mở PR, không merge.

LANE DONE
