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
    *,
    crop_halo: int | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Áp dụng hàm op_func lên ảnh và tự động hòa trộn theo mặt nạ mềm (nếu có).
    """
```

**Crop-to-mask.** Outside `mask > 0` the blend always returns the original pixels, so an operation that declares `crop_halo` (its kernel's radius of influence) runs only on the mask's bounding box plus that halo. Output is byte-identical to running on the whole image, and 3–12× faster for masks covering 5–25% of a 1080p–12MP image (see [`docs/module3/benchmark/crop_to_mask.md`](../../docs/module3/benchmark/crop_to_mask.md)).

| Cropped (`crop_halo`) | Always whole image (`None`) |
|---|---|
| denoise `gaussian` (4σ_blur + 1), `median` (k/2), `nlm` (13) | denoise `bilateral` — OpenCV float SIMD rounds by pixel position (±1) |
| sharpen `unsharp_mask` (9) | sharpen `laplacian` — `filter2D` float, same ±1 issue |
| `gamma` (0), `color_balance` with only `temperature_shift` (0) | `color_balance` with `saturation_scale` — `HSV2RGB` ±1 |
| | `clahe`, `white_balance` — use whole-image / whole-region statistics |

Only declare `crop_halo` for an operation after adding it to `tests/unit/test_crop_to_mask.py`, which checks byte equality against the whole-image path.

### 2.2 Denoising (`denoise.py`)
- `apply_denoise(image, mask, method="bilateral", strength=1.0)`
  - Methods: `"gaussian"`, `"median"`, `"bilateral"`, `"nlm"`.

### 2.3 Exposure & Contrast (`exposure_contrast.py`)
- `apply_gamma(image, mask, gamma=1.2)`
- `apply_clahe(image, mask, clip_limit=2.0, tile_grid_size=(8, 8))`

### 2.4 Sharpening (`sharpen.py`)
- `apply_sharpen(image, mask, method="unsharp_mask", amount=1.0)`

### 2.5 Color Balance & Correction (`color.py`)
- `apply_color_balance(image, mask, saturation_scale=1.0, temperature_shift=0.0)`
  - `temperature_shift` is a warm/cool tone shift via channel gains (`gain_R = 1 + k·t`, `gain_B = 1 − k·t`, `k = 20/128`), not a Kelvin colour temperature.
- `apply_white_balance(image, mask, method="gray_world", percentile=99.0)`
  - Methods: `"gray_world"`, `"white_patch"`. The mask also weights the illuminant estimate.
  - Not yet wired into the Agent toolbox — see [`HANDOFF_PERSON4.md`](HANDOFF_PERSON4.md).

### 2.6 Input contract & errors (`base.py`)
- Images: `uint8`, shape `(H, W, 3)` RGB, `(H, W)` or `(H, W, 1)` grayscale. The output always has the input shape.
- Masks: `float32`, shape `(H, W)` or `(H, W, 1)`, finite values in `[0, 1]`.
- Anything else (RGBA, float images, unknown `method`, `gamma <= 0`, NaN, …) raises `ProcessingInputError` (a `ValueError`) before any filtering runs. Normalising images (dropping alpha, converting dtype) is the caller's job.

---

## 3. Important Rules for Person 3 & AI Sessions
- Never modify the global image when a mask is provided; always apply the operation and blend using `src.region_engine.mask_utils.blend_regions`.
- All operations must be deterministic, mathematically transparent, and parameter-controlled.
- Code comments must be in **Vietnamese with proper diacritics**, explaining the mathematical principles (e.g. CLAHE tile clipping, bilateral filter spatial/range sigma).
- Logging in **English**.
