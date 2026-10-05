# Kế hoạch triển khai detect_faces — Module 2

Ngày lập: 2026-09-21. Trạng thái: **đã triển khai adapter, mask hậu xử lý, lifecycle, caller, unit/mock integration và artifact; real-model chưa nghiệm thu vì môi trường chưa có MediaPipe/checkpoint/ảnh upstream**.

API mục tiêu: `detect_faces(image, feather_radius=20) -> list[np.ndarray]`.

Mục tiêu: phát hiện từng khuôn mặt trên ảnh RGB bằng MediaPipe chạy CPU, mở rộng bbox và tạo một soft-mask toàn ảnh cho mỗi mặt. Mask phải là float32, finite, shape `(H,W)`, miền `[0,1]`. Phân biệt rõ không có detection với lỗi dependency/model/inference.

## 1. Nguồn trình bày thuật toán

| Nguồn chính thức | Vai trò |
|---|---|
| [Google AI Edge — Face Detector Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python) | Nguồn triển khai chính: MediaPipe Tasks, IMAGE mode, SRGB và kết quả bbox pixel |
| [Google AI Edge — Face Detector overview/models](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector) | Lựa chọn checkpoint, các biến thể short/full-range và tùy chọn detector |
| [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://arxiv.org/abs/1907.05047) | Bài báo gốc về thiết kế detector nhẹ; không lấy tốc độ GPU trong bài làm cam kết CPU máy hiện tại |
| [MediaPipe — face_detector_test.py](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/test/vision/face_detector_test.py) | Test upstream để đối chiếu ảnh đầu vào và expected bbox |

Tra cứu ngày 2026-09-21. Chọn hướng MediaPipe Tasks FaceDetector thay vì tiếp tục phụ thuộc trực tiếp `mp.solutions.face_detection`. Việc API legacy có chạy được hay không phải kiểm tra trên môi trường thực; chưa kết luận từ khai báo `mediapipe>=0.10.11`.

Luồng thuật toán: ảnh → preprocessing/inference/hậu xử lý của MediaPipe → bbox và confidence → mở rộng bbox → clipping → raster hóa → Gaussian feathering. Không tự viết lại mạng neural, decode anchor hay NMS. Bài BlazeFace mô tả nền tảng; checkpoint full-range cụ thể phải theo model card của nó, không mặc định mọi biến thể cùng kiến trúc của short-range.

Mở rộng bbox và feathering là hậu xử lý do dự án quy định. Đây là mask chữ nhật quanh mặt, không phải face parsing, mask da hoặc đường biên khuôn mặt chính xác.

## 2. Triển khai thuật toán

### 2.1. Hiện trạng đã kiểm tra và đã cập nhật

- `src/region_engine/face_detector.py` dùng MediaPipe Tasks `FaceDetector` ở IMAGE mode, CPU, confidence `0.5`, suppression `0.3`, lazy cache và lock quanh init/inference/close.
- Model không tải ngầm; đường dẫn lấy từ `REGION_FACE_MODEL_PATH` hoặc mặc định `models/mediapipe/face_detection_full_range.tflite`. Script tải/verify riêng yêu cầu URL và SHA-256.
- Ảnh, feather, expand ratio và output backend được validate; bbox pixel mở rộng bằng floor/ceil, clip qua `create_bbox_mask`, bỏ bbox hoàn toàn ngoài canvas kèm warning.
- Lỗi dependency/checkpoint/init và inference/output được tách thành `FaceDetectorUnavailableError` và `FaceDetectionError`, giữ cause; không biến lỗi thành `[]`.
- `src/agent/executor.py` bỏ qua action face khi không có mặt và hợp nhất nhiều mặt bằng `np.maximum.reduce` trước khi gọi Module 3.
- `create_bbox_mask` được tái sử dụng cho phần raster/feather; detector public API vẫn trả từng mask độc lập.

### 2.2. Contract đề xuất

Giữ hai tham số positional hiện có, bổ sung tùy chọn keyword-only:

```python
def detect_faces(
    image: np.ndarray,
    feather_radius: int = 20,
    *,
    expand_ratio: float = 0.15,
) -> list[np.ndarray]:
    ...
```

Chọn 0.15 mặc định để giữ mức mở rộng của code hiện tại, sửa README 0.2 cho thống nhất. Tỷ lệ tính **trên mỗi phía**: r=0.15 làm chiều rộng/chiều cao tăng 30% trước clipping. Chuyển cách làm tròn sang floor/ceil ở mục 2.4 là thay đổi có chủ đích và cần test.

| Tình huống | Hành vi |
|---|---|
| Ảnh hợp lệ | ndarray RGB uint8, `(H,W,3)`, H,W dương |
| Input không ndarray hoặc sai dtype | TypeError |
| Shape sai, grayscale/RGBA, ảnh rỗng | ValueError |
| BGR | Caller phải chuyển sang RGB; không tự đoán thứ tự màu |
| View không contiguous hoặc read-only | Chấp nhận; tạo buffer contiguous khi cần, không sửa ảnh gốc |
| Feather | Python/NumPy integer không âm, loại bool; sai kiểu → TypeError, âm → ValueError |
| Expand ratio | Số thực hữu hạn trong `[0,1]`, nhận integer 0/1, loại bool; sai kiểu → TypeError, ngoài miền/NaN/Inf → ValueError |
| Inference thành công, không có mặt đủ confidence | `[]` |
| Có N bbox hợp lệ giao ảnh | N mảng float32 `(H,W)`, độc lập vùng nhớ; không merge trong hàm |
| Thiếu dependency/model, checkpoint hỏng, backend lỗi | Exception rõ ràng, không trả `[]` hoặc mask toàn ảnh |

Validate tất cả input trước khi lazy-load model, kể cả ảnh không có mặt. Sắp output theo bbox thô `(ymin,xmin,ymax,xmax)` rồi confidence giảm dần để có thứ tự ổn định; caller không được suy ra danh tính người từ chỉ số list.

### 2.3. Backend, dependency và lifecycle

Hướng mặc định: **Tasks IMAGE + full-range CPU**, phù hợp ý định full-range của code cũ. Kiểm chứng bằng checkpoint full-range của bộ test upstream; không coi `model_selection=1` và checkpoint Tasks khác tên là cùng model chỉ vì cùng nhãn full-range.

1. Kiểm tra Python/OS/architecture và import MediaPipe Tasks thật; xác minh wheel tương thích NumPy/OpenCV/protobuf hiện có.
2. Chọn phiên bản MediaPipe thực sự chạy được, pin phiên bản trong cấu hình dependency/lock của dự án; ghi đủ version để tái lập. Không chốt một số phiên bản chưa thử.
3. Tải model bằng script chuẩn bị riêng từ nguồn Google; lưu dưới `models/mediapipe/`, ghi URL đã resolve, tên file, SHA-256, model card và backend CPU. Không tải mạng ngầm trong `detect_faces`.
4. Cấu hình đường dẫn qua `REGION_FACE_MODEL_PATH`, có đường dẫn mặc định rõ trong dự án. File model phải có trước khi chạy real-model tests.
5. Tạo FaceDetector với IMAGE mode, confidence 0.5, suppression threshold 0.3. Cố định cấu hình cho cả test và evaluation; không tự thêm NMS lần hai.
6. Tách adapter nội bộ trả record bbox pixel + confidence để đánh giá detector trước hậu xử lý; public API vẫn trả list mask.
7. Lazy-init một instance mỗi process và cache theo cấu hình bất biến. Dùng lock bảo vệ cả khởi tạo lẫn inference/close; không giả định detector thread-safe. Đóng khi application shutdown, có hook reset/close phục vụ test, đăng ký cleanup khi cần.

Đề xuất `FaceDetectorUnavailableError` cho lỗi dependency/checkpoint/init và `FaceDetectionError` cho inference/output model bất thường; subclass RuntimeError, giữ nguyên nguyên nhân bằng `raise ... from exc`. Catch có ngữ cảnh tại adapter, không nuốt exception. Không cache một lần khởi tạo thất bại thành trạng thái “không mặt”.

Nếu full-range checkpoint không tương thích với phiên bản đã chọn, ghi rõ nguyên nhân và đánh giá checkpoint thay thế riêng; không âm thầm đổi sang short-range rồi dùng cùng baseline. Full-range/sparse/short-range có expected khác nhau.

### 2.4. Bbox → mask

Tasks trả bbox pixel `(origin_x,origin_y,width,height)` trên ảnh đầu vào. Không nhân thêm W/H như legacy normalized bbox. Truyền ảnh nguyên kích thước, để Tasks preprocessing; tránh tự resize khiến sai tọa độ. Loader fixture thống nhất decode RGB và hướng EXIF; hàm ndarray không tự xử lý EXIF.

Với `(x,y,bw,bh)` và ratio r:

```text
xmin = floor(x - r*bw)
ymin = floor(y - r*bh)
xmax = ceil(x + bw + r*bw)
ymax = ceil(y + bh + r*bh)
mask = create_bbox_mask((H,W), (xmin,ymin,xmax,ymax), feather_radius)
```

Kiểm tra bbox hữu hạn, bw/bh > 0, confidence hữu hạn trong `[0,1]`. Record sai cấu trúc/NaN/âm là lỗi backend, không giả dạng ảnh không mặt. Bbox hợp lệ nhưng hoàn toàn ngoài ảnh sau mở rộng: bỏ qua và ghi warning; bbox cắt mép: clip qua helper. Không trả một mask toàn 0 đại diện cho “mặt” ngoài canvas.

Feather kế thừa contract hiện có: r=0/1 không blur, số chẵn tăng lên lẻ kế tiếp, sigma=k/3, `BORDER_REFLECT_101`. Mặc định feather=20 → k=21, sigma=7, bán kính hỗ trợ 10 pixel. Tránh nhầm `expand_ratio` với feather và tránh blur hai lần.

### 2.5. Interface Module 3 và sửa caller liên quan

Trong executor, target face với `[]` phải **bỏ qua action hiện tại**, giữ ảnh và tiếp tục action sau; không truyền None vào Module 3. Lỗi detector phải truyền lên hoặc báo trạng thái lỗi theo cơ chế của ứng dụng, không đổi thành full-image fallback.

Với nhiều mặt, đề xuất target face áp dụng cho tất cả mặt: `mask = np.maximum.reduce(face_masks)` và xử lý một lần, tránh tăng hiệu ứng do vùng chồng nhau. Đây là quy ước caller; `detect_faces` vẫn giữ mỗi mặt một mask. Cần test cả no-face, một mặt, nhiều mặt và hai mask overlap.

## 3. Testcase trên mạng, baseline và metric

### 3.1. Test upstream có expected rõ ràng

Nguồn [MediaPipe FaceDetector tests](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/test/vision/face_detector_test.py) ghép `portrait.jpg` với expected `.pbtxt`, có case `cat.jpg` expected không detection và tolerance bbox 5 pixel. Đây là **regression baseline của model**, không phải nhãn con người độc lập.

| ID | Input/checkpoint | Baseline bbox thô công khai |
|---|---|---|
| MP-PORTRAIT-FULL | `portrait.jpg` + `face_detection_full_range.tflite` cùng bộ test | 1 detection: x=293, y=112, width=229, height=229 |
| MP-PORTRAIT-SHORT | Cùng ảnh + `face_detection_short_range.tflite` | 1 detection: x=283, y=115, width=234, height=234 |
| MP-NO-FACE | `cat.jpg` + short-range như upstream | 0 detection, expected `[]` |

Expected đã đọc trực tiếp tại [portrait_expected_full_range_detection.pbtxt](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/testdata/vision/portrait_expected_full_range_detection.pbtxt) và [portrait_expected_detection.pbtxt](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/testdata/vision/portrait_expected_detection.pbtxt). Asset được khai báo trong [testdata/vision/BUILD](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/testdata/vision/BUILD).

Với full-range baseline và expand=0.15, bbox mở rộng theo contract dự án trước clip là `(258,77,557,376)`. Với short-range là `(247,79,553,385)`. Hai bbox mở rộng là **giá trị suy ra của dự án**, không phải output upstream đã công bố. Soft expected được tạo độc lập từ bbox chuẩn; không yêu cầu model thực sinh mask khớp từng pixel nếu bbox được phép sai 5 pixel.

Khi triển khai phải lấy đúng bytes ảnh/model/expected cùng revision, ghi commit SHA thay cho link master, hash từng file và xác minh kích thước ảnh sau decode. Hiện đã xác minh test/expected, chưa tải hoặc hash asset. Không dùng ảnh portrait khác cùng tên. Case cat trên backend full-range là test chuyển thể cần chạy riêng, không nói upstream đã chứng minh cấu hình đó.

### 3.2. Nhiều mặt và sát mép: baseline chuyển thể xác định

Từ ảnh portrait đã khóa kích thước `(Hp,Wp)` và bbox chuẩn full-range B=(293,112,522,341):

- **TWO-FACES:** ghép hai bản ảnh nguyên kích thước theo chiều ngang, ngăn bằng 32 cột đen; output shape `(Hp,2*Wp+32,3)`. Baseline hình học B1=B, B2=B dịch x thêm `Wp+32`; expected count=2. Không lấy dự đoán trên ảnh ghép làm ground truth.
- **EDGE-FACE:** crop portrait từ x=293, y=0 đến hết ảnh; baseline bbox thô chuyển thành `(0,112,229,341)` trong tọa độ crop. Sau expand=0.15 và clip, x bắt đầu 0, y bắt đầu 77; expected count=1. Đây là case khó hơn vì crop làm mất context.

Hai case trên là ảnh biến đổi do dự án tạo từ nguồn công khai, không phải testcase model nguyên bản của upstream. Cố định công thức/hash output trước chạy detector. Nếu model bỏ sót, ghi fail ở bộ chuyển thể và giới hạn của detector; không sửa nhãn để khớp dự đoán. Hai bản cùng một người không thay thế tập ảnh nhiều người tự nhiên.

### 3.3. Benchmark nhãn người để đánh giá khả năng tổng quát

Nguồn mở rộng: [WIDER FACE — bài báo gốc](https://arxiv.org/abs/1511.06523) và [trang dự án/dữ liệu](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). Dữ liệu có nhãn bbox và nhiều điều kiện khó. Dùng validation với annotation công khai, không dùng test split thiếu nhãn.

Đây là hạng mục đánh giá mở rộng sau smoke suite, không thay thế testcase cụ thể ở trên. Trước chạy phải tạo manifest cố định chứa ID ảnh, hash, kích thước, mọi bbox GT và thuộc tính ignore/invalid; chọn subset theo metadata với seed cố định, không chọn lại ảnh sau khi nhìn dự đoán. Báo rõ số ảnh/mặt thực tế. Chỉ gọi kết quả là WIDER AP khi dùng đúng official evaluator và split/protocol; metric subset tự tính phải ghi “subset diagnostic”.

Không chuyển bbox GT thành mask da rồi gọi đó là segmentation ground truth. Không đặt mục tiêu 100% recall cho mặt nhỏ/che khuất của model full-range khi chưa có bằng chứng. Ngưỡng chất lượng sản phẩm tổng quát cần chốt từ yêu cầu sử dụng, riêng smoke/regression có tiêu chí cụ thể bên dưới.

### 3.4. Unit test hậu xử lý với input/output tính tay

Không phụ thuộc MediaPipe hoặc checkpoint: fake adapter trả bbox pixel xác định trên ảnh RGB zeros `(10,12,3)`, score=0.9, feather=0.

| ID | Detection input, expand | Expected |
|---|---|---|
| CENTER | `(x=4,y=3,bw=4,bh=4)`, ratio=0 | Một mask 1 tại y=3..6, x=4..7; còn lại 0 |
| EXPAND | Cùng bbox, ratio=0.25 | Một mask 1 tại y=2..7, x=3..8; tổng 36 |
| DEFAULT-EXPAND | Cùng bbox, ratio=0.15 | floor/ceil cũng tạo bbox `(3,2,9,8)`; tổng 36 |
| EDGE | `(0,0,4,4)`, ratio=0.25 | Clip bbox `(-1,-1,5,5)` thành `(0,0,5,5)`; tổng 25 |
| OUTSIDE | `(20,20,2,2)`, ratio=0 | `[]`, warning về bbox ngoài canvas |
| EMPTY | Không detections | `[]`; model đã chạy thành công |
| MULTI | CENTER và `(0,0,2,2)`, ratio=0 | Hai mask độc lập, mặt trên trái đứng trước |
| INVALID-BOX | NaN hoặc bw<=0/bh<=0 | FaceDetectionError |
| MODEL-ERROR | Adapter raise lỗi inference | FaceDetectionError giữ cause, không `[]` |

Thêm test ảnh/dtype/shape/feather/ratio không hợp lệ; input read-only/noncontiguous; output không alias; init một lần qua nhiều lần gọi; close/reset giải phóng; thiếu model/dependency không ảnh hưởng import các hàm mask thuần; input sai được phát hiện trước load model. Unit test RGB kiểm tra buffer đưa vào adapter giữ thứ tự kênh bằng ảnh nhỏ ba màu khác nhau.

Hard baseline tạo bằng bất đẳng thức trên lưới chỉ số hoặc literal, không gọi bbox production. Soft oracle SciPy float64 `gaussian_filter(hard, sigma=k/3, radius=k//2, mode="mirror")`, r=0/1 giữ hard. Kiểm tra r=0,1,3,20,21 và kernel lớn hơn ảnh. Golden generator không import hàm production để sinh expected. Tham chiếu [SciPy gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).

### 3.5. Metric và tiêu chí nghiệm thu

**Tách ba tầng:** model bbox, bbox→mask, và ghép ảnh. Không suy ngược bbox từ soft-mask threshold để đánh giá detector.

| Tầng | Metric | Ngưỡng đề xuất |
|---|---|---|
| Upstream portrait | Số detection; sai số tuyệt đối x/y/width/height | Đúng 1; mỗi trường <=5 pixel, cùng model/fixture |
| No-face | Số false positive | 0 trên cat/ảnh trống smoke suite |
| TWO/EDGE chuyển thể | Matching bbox thô, IoU>=0.5 | TWO: TP=2, FP=0, FN=0; EDGE: TP=1, FP=0, FN=0; fail phải báo rõ |
| Mask contract | Shape/dtype/finite/range | 100% mask hợp lệ |
| Mock hard-mask | Pixel mismatch; diện tích | 0 mismatch; đúng diện tích tính tay |
| Mock soft-mask | MaxAE/MAE so với oracle float64 | MaxAE<=1e-6, MAE<=1e-7; báo RMSE bổ sung |
| Integration | Empty → no-op; lỗi → exception; overlap → một lần xử lý | Khớp expected ảnh từng byte ở fixture hard-mask |
| Runtime CPU | Cold init, warm inference, mask generation | Báo p50/p95 và peak memory; chưa đặt SLA chưa đo |

Matching diagnostic: sort predictions theo confidence giảm dần, mỗi prediction match một GT chưa dùng có IoU lớn nhất nếu IoU>=0.5; prediction dư là FP, GT chưa match là FN. IoU dùng bbox **chưa expand** đã thống nhất hệ tọa độ. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R). Mẫu không GT báo FP/image; mẫu không prediction báo FN, không tạo chia 0/NaN âm thầm. Bbox mở rộng nhằm chỉnh ảnh nên không dùng để đo detection IoU.

AP cần score và sweep theo protocol; public list-mask không đủ, phải dùng adapter record. Chỉ số trên WIDER diagnostic là báo cáo khám phá, chưa phải gate chất lượng tổng quát. Golden model không được tự cập nhật khi test fail. Lưu seed, CPU/OS, Python, MediaPipe, NumPy/OpenCV, model hash, config và fixture hash cùng metrics.

## 4. Test Python để đánh giá

### 4.1. File cần tạo/sửa

| File | Công việc |
|---|---|
| `src/region_engine/face_detector.py` | Tasks adapter, lifecycle, validation, errors, bbox→mask |
| `src/region_engine/README.md` | Signature, expand/feather, model setup và error contract |
| `src/agent/executor.py` | Không xử lý toàn ảnh khi không mặt; hợp nhất các mặt bằng max |
| Dependency/lock tương ứng | Pin bộ phiên bản đã kiểm chứng, model manifest |
| `scripts/prepare_face_detection_assets.py` | Tải/verify model và fixture có manifest/hash; chạy riêng trước test |
| `scripts/generate_face_mask_baselines.py` | Literal/mock/SciPy expected độc lập |
| `tests/fixtures/face_detection/` | Manifest, upstream expected, mock detections và mask golden |
| `tests/unit/test_detect_faces.py` | Validation, fake backend, hình học và lifecycle |
| `tests/integration/test_face_detection_model.py` | Inference thật, upstream và case chuyển thể |
| `tests/integration/test_face_region_processing.py` | Detect/mask/Module 3/executor |
| `scripts/evaluate_face_detection.py` | Metrics, ảnh/plot và JSON/CSV; exit khác 0 khi gate fail |

### 4.2. Trình tự chạy dự kiến

1. Chuẩn bị môi trường và asset, khóa model/fixture revision/hash; kiểm tra import và một lần inference CPU.
2. Chạy unit thuần bằng fake backend; xác nhận không cần mạng hoặc MediaPipe thật.
3. Chạy integration với model thật; không skip rồi tuyên bố detector đã nghiệm thu.
4. Chạy evaluate/export plot, mở kiểm tra, chạy regression Module 2 và executor.

CLI dự kiến sau khi tạo script:

```powershell
python scripts/prepare_face_detection_assets.py --manifest tests/fixtures/face_detection/assets.json
python scripts/generate_face_mask_baselines.py
python -m pytest tests/unit/test_detect_faces.py -q
python -m pytest tests/integration/test_face_region_processing.py -q
python -m pytest tests/integration/test_face_detection_model.py -m real_model -q
python scripts/evaluate_face_detection.py --mock-only --output-dir artifacts/module-2/detect-faces
python scripts/evaluate_face_detection.py --manifest tests/fixtures/face_detection/assets.json --output-dir artifacts/module-2/detect-faces-real
python -m pytest tests/unit/test_create_bbox_mask.py tests/unit/test_create_soft_mask.py tests/unit/test_blend_regions.py tests/unit/test_region_engine.py tests/integration/test_region_blending.py -q
```

Đã tạo các script/test nêu trên. `--mock-only` chạy không cần mạng, MediaPipe hoặc
checkpoint và chỉ kết luận về bbox→mask; lệnh real-model và evaluation mặc định
fail rõ ràng khi thiếu asset/dependency, không skip im lặng. Test real-model dùng
marker `real_model` nhưng bản thân test sẽ fail với hướng dẫn chuẩn bị asset nếu
được gọi. Lưu JUnit XML, stdout/stderr, pass/fail/skip và lý do trong CI; giữ
asset lớn ngoài Git nếu cần, kèm downloader/hash/license provenance để tái lập.

Benchmark dùng một lượt cold init, ít nhất 5 warm-up và 30 lượt warm cho mỗi fixture; tách I/O/decode khỏi inference; ghi kích thước ảnh và số mặt vì chi phí mask tăng theo N*H*W. Không tải model lại giữa các lượt warm.

## 5. Save plot và ảnh trước/sau

Thư mục dự kiến `artifacts/module-2/detect-faces/`:

```text
detect-faces/
  README.md
  run_manifest.json
  metrics.json
  metrics.csv
  detections.json
  test-results.xml
  logs/
  arrays/                 # per-face hard/soft mask, reference .npy
  images/                 # original, processed, final PNG từng case
  plots/
    01_single_face_before_after.png
    02_multiple_faces_before_after.png
    03_edge_face_clipping.png
    04_no_face_unchanged.png
    05_bbox_baseline_vs_prediction.png
    06_mask_error_and_profile.png
    07_cpu_latency.png
```

Ảnh minh họa dùng đúng fixture RGB đã khóa, không lấy screenshot có bbox làm input. Tạo processed bằng `clip(int16(original)+40,0,255).astype(uint8)` để nhìn rõ vùng chỉnh sáng, hợp nhất mask bằng max rồi blend một lần. Case no-face giữ nguyên ảnh, lưu ảnh trước/sau bằng nhau.

Plot gồm original, bbox GT/thô dự đoán/mở rộng bằng ba màu có legend, từng soft-mask, hard-blend và soft-blend. Gắn ID mặt, confidence, shape, expand, feather/k/sigma và model hash rút gọn. Plot mock error dùng cùng bbox cho actual/reference; plot detector khác bbox chỉ là sai lệch end-to-end, không gán toàn bộ lỗi cho Gaussian.

Mask hiển thị vmin=0/vmax=1; heatmap có colorbar; profile qua một biên bbox; dữ liệu metric lấy từ `.npy`, không đo từ PNG. Dùng matplotlib Agg. Mở kiểm tra single/multiple/edge/no-face, xác nhận đúng RGB, không cắt nhãn và vùng ngoài mask giữ nguyên. README artifact dẫn tới từng ảnh và lệnh tái tạo.

## 6. Kết luận thuật toán, kết quả và điều kiện nghiệm thu

Phương án: dùng model MediaPipe Tasks trên CPU để định vị bbox, mở rộng có quy tắc, tái sử dụng bbox mask và Gaussian đã có. Trọng tâm hoàn thiện là khả năng tái lập model, validation, phân biệt lỗi với no-face, quản lý vòng đời và bảo đảm executor không biến yêu cầu chỉnh mặt thành chỉnh toàn ảnh.

Giới hạn: bbox feathered có thể phủ tóc/nền; khuôn mặt nhỏ, nghiêng hoặc che khuất có thể bị bỏ sót; nhiều mask toàn ảnh tốn bộ nhớ theo số mặt. Mask đúng về số học không chứng minh detector đúng trên mọi ảnh. Test fake backend không thay thế inference thật.

Kết quả hiện tại: đã triển khai adapter Tasks, validation, bbox→mask, lifecycle,
error contract và caller; đã chạy unit/mock integration và sinh artifact mock.
Real-model chưa được tuyên bố nghiệm thu: môi trường hiện tại không có
`mediapipe`, checkpoint hoặc ảnh upstream. Artifact mock chỉ chứng minh hậu xử
lý hình học, không chứng minh chất lượng detector.

| Kết quả phải cập nhật sau chạy | Hiện tại |
|---|---|
| MediaPipe/checkpoint/CPU đã chạy được, version và hash | Chưa xác minh: thiếu dependency/checkpoint |
| Upstream portrait và no-face | Chưa chạy; test real-model fail rõ khi thiếu asset |
| Nhiều mặt và sát mép | Mock MULTI/EDGE: đạt; real model chưa chạy |
| Unit bbox→mask, MaxAE/MAE/RMSE | 37 unit pass; 7/7 mock case pass; MaxAE mock ≤ `3.68e-8` |
| Executor no-face/multiple/error và Module 3 | 4 integration pass |
| Latency p50/p95, memory | Đã ghi mock CPU p50/p95 trong `artifacts/module-2/detect-faces/`; chưa phải SLA real |
| Plot đã mở kiểm tra | Đã sinh 7 plot mock; chưa xác minh plot real |
| Kết luận nghiệm thu | Hậu xử lý đạt gate mock; real detector còn blocked bởi asset/runtime |

Checklist thực hiện:

- [x] Khảo sát code, tìm nguồn thuật toán và baseline công khai có số liệu.
- [x] Lập contract, phương án backend, testcase, metric và quy trình xuất plot.
- [ ] Pin dependency/model/fixture và chạy smoke CPU thật.
- [x] Triển khai adapter, mask, lifecycle/error và sửa caller liên quan.
- [x] Chạy unit, integration mock và regression liên quan; real-model còn chờ asset.
- [x] Lưu artifact mock gồm ảnh trước/sau, plot, metrics và logs.
- [x] Điền kết luận thực nghiệm; nêu rõ real-model chưa chạy và mock không thay thế detector thật.
