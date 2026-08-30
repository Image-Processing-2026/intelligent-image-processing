# Module 3: Classical Image Processing Engine

**Owner:** Person 3  
**Directory:** `src/processing_engine/`

---

## 1. Responsibilities
This module is the **"Graded Substance & Hands"** of the project:
1. Implements classical, mathematical image processing algorithms using **OpenCV** and **scikit-image**.
2. **Every single operation** must support region-aware execution via the standardized wrapper:
   $$\text{operation}(image, mask, **parameters) \rightarrow \text{processed\_image}$$
3. Prevents using generative AI black-box shortcuts.

---

## 2. Interface Contracts

All operations take `image: np.ndarray` (RGB uint8), an optional `mask: np.ndarray | None` (float32 `[0.0, 1.0]`), and algorithm parameters, returning the blended `np.ndarray` (RGB uint8).

### 2.1 Base Blending Mechanism (`base.py`)
```python
def apply_region_op(
    image: np.ndarray, 
    mask: np.ndarray | None, 
    op_func: Callable[[np.ndarray, ...], np.ndarray], 
    **kwargs
) -> np.ndarray:
    """
    Áp dụng hàm op_func lên ảnh và tự động hòa trộn theo mặt nạ mềm (nếu có).
    """
```

### 2.2 Denoising (`denoise.py`)
- `apply_denoise(image, mask, method="bilateral", strength=1.0)`
  - Methods: `"gaussian"`, `"median"`, `"bilateral"`, `"nlm"`.

### 2.3 Exposure & Contrast (`exposure_contrast.py`)
- `apply_gamma(image, mask, gamma=1.2)`
- `apply_clahe(image, mask, clip_limit=2.0, tile_grid_size=(8, 8))`

### 2.4 Sharpening (`sharpen.py`)
- `apply_sharpen(image, mask, method="unsharp_mask", amount=1.0)`

### 2.5 Color Balance & Correction (`color.py`)
- `apply_color_balance(image, mask, saturation_scale=1.1, temperature_shift=0.0)`

---

## 3. Important Rules for Person 3 & AI Sessions
- Never modify the global image when a mask is provided; always apply the operation and blend using `src.region_engine.mask_utils.blend_regions`.
- All operations must be deterministic, mathematically transparent, and parameter-controlled.
- Code comments must be in **Vietnamese with proper diacritics**, explaining the mathematical principles (e.g. CLAHE tile clipping, bilateral filter spatial/range sigma).
- Logging in **English**.
