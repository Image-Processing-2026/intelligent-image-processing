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
    original_image: np.ndarray, 
    processed_image: np.ndarray, 
    soft_mask: np.ndarray
) -> np.ndarray:
    """
    Hòa trộn ảnh gốc và ảnh đã xử lý thông qua mặt nạ mềm:
    I_out = soft_mask * I_processed + (1.0 - soft_mask) * I_original
    """
```

### 2.2 Face Detection (`face_detector.py`)
```python
def detect_faces(image: np.ndarray, expand_ratio: float = 0.2) -> list[np.ndarray]:
    """
    Phát hiện các khuôn mặt trong ảnh bằng MediaPipe và trả về danh sách soft-masks.
    """
```

### 2.3 Semantic Text Segmentation (`detector.py`)
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
