# Kiểm tra mức sẵn sàng Module 2 trước khi triển khai controller

Ngày kiểm tra: 2026-09-21. Phạm vi: đọc implementation, caller, tests, scripts đánh giá, manifests và artifact; chạy lại test chọn lọc và smoke API. Không sửa production hoặc cài/tải model trong đợt audit.

## Kết luận

**Đủ nền tảng để bắt đầu xây controller và tích hợp các nhánh hình học. Chưa đủ để nghiệm thu end-to-end toàn bộ Module 2 với face/semantic inference thật.**

Sau đợt triển khai theo audit, controller Module 2, capability report và đường
đánh giá mock/real đã được bổ sung. Kết luận về real-model vẫn giữ nguyên:
controller có thể báo `assets_ready`, `inference_executed` và `quality_passed`
riêng rẽ, nhưng môi trường hiện tại chưa có checkpoint/dependency/dataset nên
chưa thể công bố chất lượng AI thật.

Bốn hàm nền chạy thật. Hai hàm AI có code adapter gọi thư viện/model thật, không phải production stub trả dữ liệu giả; tuy nhiên bằng chứng kiểm thử đang chủ yếu dùng fake backend, và môi trường hiện không có đủ dependency/checkpoint/dataset để chạy model.

## 1. Trạng thái từng function

| Function | Implementation hiện tại | Bằng chứng kiểm tra | Kết luận |
|---|---|---|---|
| create_soft_mask | NumPy validation/chuẩn hóa + OpenCV GaussianBlur thật | Test chạy lại pass; artifact cũ có 20/20 case trong atol=1e-6 | Sẵn sàng tích hợp theo contract hiện tại |
| blend_regions | Broadcasting float32, clip/cast uint8, bảo toàn M=0/1 và kênh giống nhau | Test chạy lại pass; có baseline độc lập và kiểm tra ownership | Sẵn sàng tích hợp; có sai số lượng tử hóa 1 mức tại ranh giới float32 đã được ghi nhận |
| create_bbox_mask | Validate shape/bbox, clip, slicing nửa kín, gọi soft-mask | Test chạy lại pass; artifact cũ 819/819 soft outputs, hard mismatch=0 | Sẵn sàng tích hợp |
| create_quadrant_mask | Validate 5 vùng, map bbox, xử lý kích thước lẻ/tiny | Test chạy lại pass; artifact cũ 312/312 soft outputs, hard mismatch=0 | Sẵn sàng tích hợp |
| detect_faces | Adapter MediaPipe Tasks CPU thật, lazy cache/lock, bbox expansion và mask | Unit/fake integration pass; 2 real-model tests fail vì asset thiếu | Hậu xử lý sẵn sàng; detector thật chưa nghiệm thu |
| segment_by_prompt | Exact commands chạy thật; semantic dùng adapter GroundingDINO + MobileSAM | Unit/fake integration pass; 1 real-model test fail vì asset thiếu | Hình học chạy được; chất lượng semantic chưa xác minh |

Các metric artifact là báo cáo đã có trên disk, không phải toàn bộ script đánh giá được chạy lại trong audit. Kết quả chạy mới ở mục 2.

Feather trong các API hiện là kích thước kernel lịch sử: số chẵn làm lẻ kế tiếp, sigma=k/3; không phải bán kính hình học. Controller phải giữ hoặc công bố rõ quy ước này.

## 2. Kết quả chạy mới

Interpreter: `C:/Users/ADMIN/scoop/apps/python313/current/python.exe`, Python 3.13.

Lượt không model:

```powershell
python -m pytest tests/unit/test_create_soft_mask.py tests/unit/test_soft_mask_fixtures.py tests/unit/test_blend_regions.py tests/unit/test_create_bbox_mask.py tests/unit/test_create_quadrant_mask.py tests/unit/test_detect_faces.py tests/unit/test_segment_by_prompt.py tests/unit/test_region_engine.py tests/integration/test_region_blending.py tests/integration/test_face_region_processing.py tests/integration/test_prompt_region_processing.py -q
```

Kết quả: **255 passed in 3.77s**. Đây là suite chọn lọc Module 2/caller, không phải toàn bộ dự án. Nhóm AI trong suite này có mock.

Lượt model thật, chủ động bỏ cấu hình loại marker mặc định:

```powershell
python -m pytest -o addopts= -m real_model tests/integration/test_face_detection_model.py tests/integration/test_prompt_segmentation_model.py -q --tb=short
```

Kết quả: **3 failed in 0.60s** vì thiếu model/ảnh fixture, chưa đến bước inference. Không được diễn giải thành model dự đoán sai hoặc pipeline đã pass.

Smoke gọi public API trực tiếp với ảnh zeros RGB `(64,96,3)`:

- `detect_faces(image)` → FaceDetectorUnavailableError: thiếu checkpoint full-range.
- `segment_by_prompt(image,"person")` → SegmentationUnavailableError: thiếu snapshot DINO.
- `segment_by_prompt(image,"center")` → ndarray `(64,96)`, float32.

Các path default thiếu gồm model MediaPipe, snapshot DINO, MobileSAM checkpoint, portrait/cat upstream và ảnh Penn-Fudan 00046. Metadata môi trường: chưa cài mediapipe, torchvision, mobile-sam; có torch 2.12.0 và transformers 5.9.0. Chưa import/khởi tạo neural backend thành công nên việc có package không đồng nghĩa tương thích runtime.

## 3. Mock nằm ở đâu?

Production default factories là `_MediaPipeTasksBackend` và `_GroundingDinoMobileSAMBackend`: có lời gọi inference thật. Không thấy fallback mock được chọn tự động trong public API.

Mock được inject tại:

- `tests/unit/test_detect_faces.py`: FakeBackend thay `_DETECTOR_FACTORY`.
- `tests/unit/test_segment_by_prompt.py`: FakeBackend thay `_SEGMENTATION_FACTORY`.
- `tests/integration/test_face_region_processing.py`: bbox giả để kiểm tra mask/blend và monkeypatch executor.
- `tests/integration/test_prompt_region_processing.py`: union mask tự tạo, fake segment/gamma để kiểm tra routing; không phải semantic end-to-end thật.
- `scripts/evaluate_face_detection.py`: `_FixtureBackend` cấp bbox cố định.
- `scripts/evaluate_prompt_segmentation.py`: `_FixtureBackend` cấp bbox/mask cố định.

Mock là phù hợp để kiểm thử logic, nhưng không chứng minh nhận diện/phân đoạn ảnh thật. Các plot single-face/multiple-face/person hiện có trong hai thư mục artifact AI minh họa dữ liệu tổng hợp/mock. Thời gian CPU của chúng không phải latency MediaPipe/GroundingDINO/MobileSAM.

## 4. Các thiếu sót cần xử lý trước nghiệm thu

### P1 — Evaluator chưa đánh giá model thật và trạng thái dễ gây hiểu nhầm — đã xử lý phần code

`scripts/evaluate_prompt_segmentation.py:200` gán `real_ready = mock_only`. Artifact hiện ghi đồng thời `mock_only=true`, `real_model_ready=true`, `passed=true`. Chạy không có `--mock-only` sẽ không chuyển sang model thật: toàn bộ đường đánh giá vẫn inject fixture, chỉ trạng thái cuối đổi thành fail. CLI `--manifest` và `--device` được parse nhưng không truyền vào evaluate.

`scripts/evaluate_face_detection.py:204` chỉ kiểm tra file tồn tại/import MediaPipe ở `_real_readiness`; không chạy model hoặc so expected bbox. Nếu có file và import được thì báo ready vẫn chưa chứng minh checkpoint hợp lệ hoặc inference đúng.

Đã tách `assets_ready`, `inference_executed`, `quality_passed`, `mode=mock|real`;
normal mode chạy đường real khi asset sẵn sàng, ghi failure rõ khi thiếu asset,
và restore/reset factory trong `finally`. Face real evaluator so expected bbox
IoU; semantic real evaluator so GT IoU/Dice. Chất lượng model thật vẫn chưa có
kết quả vì asset/runtime chưa sẵn sàng.

### P1 — Asset và dependency chưa hoàn chỉnh

Hai assets.json còn URL/hash null. `pyproject.toml` extra segmentation chưa bao gồm MobileSAM hoặc recipe pin commit đủ để tái tạo; các dependency chỉ lower-bound, chưa có bộ đã kiểm chứng. Môi trường chạy test có NumPy 2.2.6 trong khi dự án khai báo `<2.0.0`: test hiện tại không xác nhận môi trường cài đúng specification.

`prepare_prompt_segmentation_assets.py` chỉ tải từng file bằng URL/hash, trong khi entry GroundingDINO mô tả một directory snapshot. Script chưa có logic snapshot/extract/config/tokenizer nhiều file và không xử lý trường datasets. Điền URL/hash chưa đủ để tạo bộ asset dùng được.

### P1 — Real-model tests còn quá yếu ngay cả khi có asset

Face portrait test chỉ kiểm tra số mask và mask nonempty; `expected_bbox` trong manifest chưa được assert. Không có real test nhiều mặt/sát mép được triển khai trong file này.

Semantic real test chỉ kiểm tra dtype/shape/range/nonempty. Một mask toàn 1 có thể pass; test không đọc GT Penn-Fudan, không tính IoU/Dice, chưa có sky/negative/multi-instance evaluation thật. Cần bổ sung đúng baseline/gate trong docs 05/06.

`pyproject.toml:63` mặc định `-m 'not real_model'`: test mặc định xanh không có nghĩa model thật đã đạt. CI cần một job explicit real-model khác với unit/mock.

### P2 — Prompt length và lifecycle cần hoàn thiện

`detector.py:69` vẫn dùng bộ đếm conservative trước khi khởi tạo backend;
backend nay gọi tokenizer với `truncation=False` và đối chiếu số `input_ids`
thực tế với model limit 256, nên prompt bị truncate không được âm thầm chấp nhận.
Test ranh giới tokenizer với checkpoint thật vẫn cần chạy khi asset có mặt.

Cache semantic nay dùng factory id cùng DINO/SAM resolved paths. Nếu controller
đổi config khác ngoài hai path thì vẫn phải gọi reset/reinitialize; metadata
capability ghi rõ đây là cấu hình process-local.

## 5. Sẵn sàng cho controller đến đâu?

Những phần đã có:

- Sáu public functions và exception riêng cho hai backend AI.
- Mask float32, `(H,W)`, `[0,1]`; `blend_regions` còn nhận `(H,W,1)`/None theo contract riêng.
- Executor đã bỏ qua action khi không mặt/zero semantic mask, merge nhiều mặt bằng max, không tự fallback toàn ảnh khi lỗi detector.
- Module 3 `apply_region_op` xử lý ảnh rồi gọi `blend_regions`; không cần controller blend thêm lần nữa.

Các phần controller đã được bổ sung:

- `src/region_engine/controller.py` cung cấp `RegionRequest`, `RegionResult`,
  `resolve_region()` và `capabilities()` cho sáu loại vùng: full, bbox,
  spatial, face, semantic và binary mask.
- `RegionOperation` có `region_type`, bbox/quadrant, feather, expand ratio và
  merge policy; executor routing qua controller nhưng vẫn giữ seam test cho
  detector/segmenter.
- Result luôn có mask cụ thể; `empty` là zero mask, lỗi request/backend/inference
  dùng exception typed, và metadata ghi backend/count/prompt/config hình học.
- `capabilities()` tách readiness hình học, assets, dependency và
  `inference_verified`; controller không blend ảnh.
- Hai evaluator đã tách `mode=mock|real`, `assets_ready`,
  `inference_executed`, `quality_passed`; real mode chỉ gọi model khi assets
  sẵn sàng và semantic case có IoU/Dice với GT.

End-to-end thật từ request face/semantic → model → mask → Module 3 vẫn chưa
được nghiệm thu vì thiếu asset/runtime ở môi trường hiện tại.

Đề xuất controller:

```text
resolve_region(image, request) -> RegionResult

request:
  kind = full | bbox | spatial | face | semantic | binary_mask
  bbox / quadrant / prompt / binary_mask tương ứng
  feather_radius, expand_ratio và merge policy khi phù hợp

result:
  status = ok | empty
  mask: float32 (H,W)
  instance_masks tùy chọn
  metadata: backend, config/model version, bbox/count, timing

typed exceptions:
  invalid_request | backend_unavailable | inference_failed
```

Không dùng None cho empty vì Module 3 hiểu None là toàn ảnh. Full nên tạo ones tại ranh giới controller để kết quả luôn có mask cụ thể; empty là zeros + status. `blend_regions` vẫn là utility được Module 3 gọi một lần. `capabilities()` phân biệt geometric ready, assets present và backend warmup/inference verified.

Theo sơ đồ người dùng: Module 4 gọi controller Module 2 để lấy vùng, Module 3 nhận ảnh/mask/tham số để xử lý; quyết định SHIP/tái xử lý vẫn thuộc Module 4 + evaluator. Không đưa vòng lặp đánh giá chất lượng toàn hệ thống vào controller vùng.

## 6. Thứ tự công việc đề xuất

1. [x] Sửa status/evaluator để mock không thể bị hiểu là real-model pass; bổ sung đường real evaluation.
2. Hoàn thiện môi trường/asset chuẩn, downloader snapshot/dataset, pin version/hash và chạy smoke CPU.
3. Chạy MediaPipe baseline bbox + negative/multi/edge; chạy semantic GT IoU/Dice/negative, sky riêng. Xuất lại plot ghi đúng mock/real.
4. [x] Xây controller request/result/routing trên bốn hàm đã ổn; hai capability AI có thể unavailable rõ ràng trong khi hoàn tất bước 2–3.
5. Nghiệm thu controller từng nhánh và với Module 3: full/bbox/spatial/binary chạy thật; face/semantic bắt buộc model thật; empty no-op, error propagate, ảnh ngoài vùng giữ nguyên, không blend hai lần.

Không cần đợi model để bắt đầu code controller. Tuy nhiên chỉ được công bố Module 2 end-to-end hoàn chỉnh khi hai nhánh AI qua real-model evaluation và contract tích hợp được kiểm chứng.

## 7. Kết quả triển khai audit

- `src/region_engine/controller.py`: request/result contract, routing sáu loại
  vùng, merge face, empty/error semantics và `capabilities()`.
- `src/agent/executor.py` và `src/agent/state.py`: routing qua controller,
  schema runtime có region type/bbox/quadrant/feather/merge policy.
- Hai evaluator: mock/real mode tách biệt, readiness không thể báo mock là
  real, real evaluator có gate bbox hoặc IoU/Dice.
- `scripts/prepare_prompt_segmentation_assets.py`: hỗ trợ file,
  archive/snapshot và dataset archive với SHA-256, required files và kiểm tra
  archive path.
- Semantic backend kiểm tra token thực tế sau tokenizer và cache theo factory
  cùng model paths.

Kiểm chứng sau triển khai:

```text
270 passed — suite chọn lọc Module 2/controller/caller, không model thật
ruff check — All checks passed
mock face evaluator — 7/7
mock semantic evaluator — 3/3
real evaluator — fail rõ với assets_ready=false, inference_executed=false
```

Controller và các nhánh hình học đủ để tích hợp tiếp; hai nhánh AI chỉ được
chuyển sang trạng thái nghiệm thu sau khi chuẩn bị asset/dependency thật và
chạy lại real-model evaluator với quality gate đạt.
