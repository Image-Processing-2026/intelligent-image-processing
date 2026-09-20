# Module 2: Region Engine & Mask Synthesis

**Owner:** Person 2  
**Directory:** `src/region_engine/`

---

## 1. Responsibilities
This module acts as the "Eyes & Spatial Targeting" system:
1. **`mask_utils.py`:** Transforms binary masks into continuous soft masks (`float32 [0.0, 1.0]`) via Gaussian feathering and handles seamless alpha-blending to prevent edge artifacts.
2. **`face_detector.py`:** Runs lightweight MediaPipe Face Detection to localize faces on CPU without CUDA requirements.
3. **`detector.py`:** Runs text-prompted semantic segmentation (GroundingDINO + MobileSAM/SAM2) when the VLM requests a target like "sky" or "background".
4. **`spatial.py`:** Generates geometric masks (quadrants, bounding boxes, user crops).

---

## 2. Interface Contracts

### 2.1 Soft-Mask Generation & Blending (`mask_utils.py`)
```python
def create_soft_mask(binary_mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    """
    Làm mịn biên mặt nạ nhị phân thành mặt nạ mềm (soft mask) float32 [0.0, 1.0].
    """


def blend_regions(
    original_image: np.ndarray, processed_image: np.ndarray, soft_mask: np.ndarray | None
) -> np.ndarray:
    """
    Hòa trộn ảnh gốc và ảnh đã xử lý thông qua mặt nạ mềm:
    I_out = soft_mask * I_processed + (1.0 - soft_mask) * I_original
    soft_mask=None chọn toàn bộ processed_image và trả về một bản sao.
    """
```

### 2.2 Face Detection (`face_detector.py`)
```python
def detect_faces(
    image: np.ndarray,
    feather_radius: int = 20,
    *,
    expand_ratio: float = 0.15,
) -> list[np.ndarray]:
    """
    Phát hiện khuôn mặt bằng MediaPipe Tasks và trả về danh sách soft-masks.
    """
```

`image` phải là ảnh RGB `uint8` shape `(H, W, 3)` không rỗng; không tự đoán
hoặc đổi BGR/RGBA. View không contiguous và buffer read-only được chấp nhận,
nhưng adapter tạo bản sao contiguous trước khi đưa vào backend. `feather_radius`
dùng cùng quy ước Gaussian với các API mask khác. `expand_ratio` là số thực hữu
hạn trong `[0, 1]`, mở rộng trên từng phía của bbox thô; mặc định `0.15` làm
tăng mỗi kích thước khoảng 30% trước khi clip. Bbox dùng `floor` cho góc trên-
trái và `ceil` cho góc dưới-phải. Detections được sắp theo bbox thô
`(ymin, xmin, ymax, xmax)` rồi confidence giảm dần, chuyển sang
`create_bbox_mask`, trả một mask độc lập cho mỗi mặt và không merge trong hàm.
Bbox giao ảnh một phần được clip; bbox hoàn toàn ngoài ảnh bị bỏ qua kèm warning.

Backend dùng MediaPipe Tasks IMAGE mode, confidence `0.5`, suppression `0.3`
và được lazy-init/cached theo process dưới lock. Model không được tải ngầm;
đường dẫn lấy từ `REGION_FACE_MODEL_PATH`, mặc định là
`models/mediapipe/face_detection_full_range.tflite`. Cần chuẩn bị checkpoint
trước khi gọi inference thật.

Inference thành công nhưng không có detection trả `[]`. Thiếu MediaPipe/model
hoặc lỗi khởi tạo ném `FaceDetectorUnavailableError`; lỗi inference/output
backend ném `FaceDetectionError`. Các lỗi này giữ nguyên cause và không bị nuốt
thành mask rỗng hoặc mask toàn ảnh. Dùng `close_face_detector()` hoặc
`reset_face_detector()` trong lifecycle/test hook. Script
`scripts/prepare_face_detection_assets.py` là đường duy nhất để tải model và
phải kiểm tra SHA-256; `detect_faces` không tải mạng ngầm.

### 2.3 Bounding-Box Mask (`spatial.py`)

```python
def create_bbox_mask(
    image_shape: tuple[int, int] | list[int] | np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | np.ndarray,
    feather_radius: int = 15,
) -> np.ndarray:
    """
    Tạo soft mask float32 toàn ảnh từ bbox (xmin, ymin, xmax, ymax).
    """
```

`image_shape` phải là tuple/list hoặc NumPy array 1D đúng hai số nguyên dương
`(H, W)`. `bbox` phải có đúng bốn số nguyên theo quy ước OpenCV/NumPy: biên
trái và trên được tính, biên phải và dưới không được tính. Số âm hoặc tọa độ
vượt ảnh được phép và được clip vào canvas; bbox đảo (`xmin > xmax` hoặc
`ymin > ymax`) là lỗi `ValueError`, không tự động đổi đầu mút.

`feather_radius` là số nguyên không âm và không nhận bool/float. Tên tham số
được giữ tương thích với API hiện có nhưng giá trị này là kích thước Gaussian
kernel: số chẵn được làm tròn lên số lẻ kế tiếp, sigma bằng `kernel / 3`, và
biên dùng `BORDER_REFLECT_101`. Giá trị `0` và `1` không blur. Bbox rỗng hoặc
nằm hoàn toàn ngoài ảnh trả mask toàn 0; bbox phủ toàn ảnh trả mask toàn 1.

Hàm trả về một mảng mới `float32` shape `(H, W)`, hữu hạn và nằm trong
`[0.0, 1.0]`; không sửa `image_shape`, `bbox` hoặc input array. Caller phải
truyền `image.shape[:2]`, không truyền trực tiếp shape ba chiều.

### 2.4 Quadrant Mask (`spatial.py`)

```python
def create_quadrant_mask(
    image_shape: tuple[int, int] | list[int] | np.ndarray,
    quadrant: str,
    feather_radius: int = 25,
) -> np.ndarray:
    """
    Tạo soft mask float32 cho top/bottom/left/right/center.
    """
```

`image_shape` dùng cùng contract với `create_bbox_mask`: tuple/list hoặc
NumPy array 1D đúng hai số nguyên dương `(H, W)`. `quadrant` phải là chuỗi
khớp chính xác một trong `top`, `bottom`, `left`, `right`, `center`;
chuỗi sai, rỗng, viết hoa hoặc có khoảng trắng báo `ValueError`, không
fallback sang mask toàn ảnh. Quadrant không phải chuỗi báo `TypeError`.

Quy ước vùng dùng cận cuối không bao gồm: `top` chọn
`y in [0, H//2)`, `bottom` chọn `y in [H//2, H)`, `left` chọn
`x in [0, W//2)`, `right` chọn `x in [W//2, W)`. `center` chọn
`y in [H//4, (3*H)//4)` và `x in [W//4, (3*W)//4)`. Kích thước lẻ dành
hàng/cột dư cho `bottom`/`right`; vùng center có thể rỗng trên ảnh rất
nhỏ.

`feather_radius` giữ nguyên quy ước Gaussian của `create_soft_mask`:
`0`/`1` không blur, số chẵn dương được làm tròn lên số lẻ kế tiếp,
sigma `= kernel / 3`, biên `BORDER_REFLECT_101`. Kết quả là mảng mới
`float32` shape `(H, W)`, finite, `[0, 1]`; không sửa input.

### 2.5 Semantic Text Segmentation (`detector.py`)
```python
def segment_by_prompt(
    image: np.ndarray,
    text_prompt: str,
    feather_radius: int = 15,
) -> np.ndarray:
    """
    Chuyển đổi câu lệnh mô tả đối tượng (ví dụ 'sky', 'background') thành soft-mask.
    """
```

Ảnh phải là RGB `uint8` shape `(H, W, 3)` không rỗng; prompt phải là chuỗi
không rỗng sau NFC, gộp whitespace, strip và lowercase. Prompt dài hơn giới
hạn 256 token ước lượng của model báo `ValueError`; không cắt ngầm. Alias exact
`người`, `bầu trời`/`trời`, `mèo`, `chó` lần lượt gửi `person`, `sky`, `cat`,
`dog`; không dịch tự do các câu tiếng Việt khác.

Exact command `full`, `all`, `toàn`, `toàn bộ`, `full_image` trả mask toàn 1.
`top`, `bottom`, `left`, `right`, `center`, `giữa` dùng quadrant geometry mà
không load model. Các prompt còn lại đi qua GroundingDINO Tiny + MobileSAM
local trên CPU. DINO dùng box threshold `0.35`, text threshold `0.25`, NMS
class-agnostic IoU `>0.8`; các bbox được clip, mask MobileSAM được OR thành
một union rồi mới gọi `create_soft_mask` đúng một lần. Không có detection hợp lệ
trả mask toàn 0. Thiếu model/dependency/init ném `SegmentationUnavailableError`;
lỗi inference hoặc output ném `SegmentationInferenceError`, không fallback thành
mask toàn ảnh hay quadrant.

Backend lazy-load/cache/lock nằm ở `segmentation_backend.py`. Mặc định model
là thư mục local `models/segmentation/grounding-dino-tiny` và checkpoint
`models/segmentation/mobile_sam.pt`; có thể đổi bằng
`REGION_DINO_MODEL_PATH` và `REGION_MOBILE_SAM_CHECKPOINT`. Runtime không tải
model từ mạng. Dùng `close_segmentation_backend()` hoặc
`reset_segmentation_backend()` khi shutdown/test. Cần cài optional extra
segmentation và chuẩn bị asset bằng script riêng trước real-model inference.

### 2.6 Module 2 controller (`controller.py`)

`resolve_region(image, request)` là facade chuẩn hóa cho caller bên ngoài.
Request có `kind` là `full`, `bbox`, `spatial`, `face`, `semantic` hoặc
`binary_mask`, cùng các trường `bbox`, `quadrant`, `prompt`, `binary_mask`,
`feather_radius`, `expand_ratio` và `merge_policy`. Mapping runtime cũng được
chấp nhận; nếu bỏ `kind`, controller suy luận từ `target_prompt` theo các
exact command đã khóa.

Controller luôn trả `RegionResult` với mask `float32`, shape `(H,W)`, finite và
`[0,1]`; không dùng `None` cho empty. `status` là `ok` hoặc `empty`, trong đó
empty dùng mask toàn 0. Face masks được merge bằng `max` một lần và giữ
`instance_masks`; semantic result giữ provenance prompt/backend trong metadata.
Controller không gọi `blend_regions`; Module 3 vẫn là nơi blend duy nhất.

`InvalidRegionRequestError` dành cho request sai, `RegionBackendUnavailableError`
cho dependency/checkpoint chưa sẵn sàng và `RegionInferenceError` cho lỗi model
đã khởi tạo. `capabilities()` chỉ đọc đường dẫn/dependency và trả riêng trạng
thái hình học, asset, dependency và `inference_verified`; không tự warm-up hoặc
tải model.

---

## 3. Important Rules for Person 2 & AI Sessions
- All masks exported to Module 3 must have `dtype=np.float32`, values in `[0.0, 1.0]`, and dimensions matching the input image `(H, W)`.
- Always optimize for CPU execution (MobileSAM, MediaPipe) — avoid heavy CUDA requirements to ensure cross-platform reproducibility on AMD iGPUs.
- Write internal logic comments in **Vietnamese with proper diacritics**.
- Write logging in **English**.
