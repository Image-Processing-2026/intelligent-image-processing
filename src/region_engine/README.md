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
def detect_faces(image: np.ndarray, expand_ratio: float = 0.2) -> list[np.ndarray]:
    """
    Phát hiện các khuôn mặt trong ảnh bằng MediaPipe và trả về danh sách soft-masks.
    """
```

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
def segment_by_prompt(image: np.ndarray, text_prompt: str) -> np.ndarray:
    """
    Chuyển đổi câu lệnh mô tả đối tượng (ví dụ 'sky', 'background') thành soft-mask.
    """
```

---

## 3. Important Rules for Person 2 & AI Sessions
- All masks exported to Module 3 must have `dtype=np.float32`, values in `[0.0, 1.0]`, and dimensions matching the input image `(H, W)`.
- Always optimize for CPU execution (MobileSAM, MediaPipe) — avoid heavy CUDA requirements to ensure cross-platform reproducibility on AMD iGPUs.
- Write internal logic comments in **Vietnamese with proper diacritics**.
- Write logging in **English**.
