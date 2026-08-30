# Module 1: Image Analyzer & Evaluator

**Owner:** Person 1  
**Directory:** `src/analyzer_evaluator/`

---

## 1. Responsibilities
This module acts as the "Sensory & Diagnostic Instrument" of the system:
1. **`analyzer.py`:** Extracts objective, numerical image quality metrics (brightness, contrast, noise, blur, sharpness, histogram, color distribution).
2. **`reference_eval.py`:** Calculates full-reference fidelity metrics (**PSNR**, **SSIM**, **MSE**) when ground-truth original images are available (specifically on the synthetic test set).
3. **`no_reference_eval.py`:** Calculates no-reference perceptual quality (**BRISQUE**, **NIQE**) and metric deltas across iterations for real user-uploaded photos.

---

## 2. Interface Contracts

### 2.1 Technical Metrics Extraction (`analyzer.py`)
```python
def analyze_image(image: np.ndarray) -> TechnicalMetrics:
    """
    Phân tích toàn diện ảnh đầu vào và trả về cấu trúc TechnicalMetrics.
    
    Args:
        image: Ảnh đầu vào định dạng RGB, np.uint8, shape (H, W, 3).
        
    Returns:
        TechnicalMetrics: Chứa các trường brightness, contrast, noise, blur, sharpness, histogram.
    """
```

### 2.2 Reference-Based Evaluation (`reference_eval.py`)
```python
def evaluate_reference(
    current_image: np.ndarray, ground_truth_image: np.ndarray
) -> dict[str, float]:
    """
    Đánh giá độ tương đồng pixel giữa ảnh hiện tại và ảnh gốc mẫu (ground truth).
    Chỉ sử dụng cho tập kiểm thử nhân tạo (synthetic dataset).

    Returns:
        {"psnr": float, "ssim": float, "mse": float}
    """
```

### 2.3 No-Reference Evaluation (`no_reference_eval.py`)
```python
def evaluate_no_reference(
    current_image: np.ndarray, previous_image: np.ndarray | None = None
) -> dict[str, float]:
    """
    Đánh giá chất lượng ảnh thực tế không có ảnh gốc mẫu.
    Tính điểm BRISQUE/NIQE và đo lường độ cải thiện so với vòng lặp trước.

    Returns:
        {"brisque": float, "niqe": float, "delta_contrast": float, "delta_sharpness": float}
    """
```

---

## 3. Important Rules for Person 1 & AI Sessions
- **Never calculate PSNR/SSIM on real user images without ground truth.** Use `no_reference_eval.py` for real photos.
- Ensure all metric calculations handle single-channel (grayscale) and 3-channel (RGB) images gracefully.
- Write internal algorithmic comments and docstrings in **Vietnamese with proper diacritics**.
- Write logging in **English**.
