"""
Trích xuất chỉ số kỹ thuật ảnh (Technical Metrics Analyzer).

Cung cấp engine phân tích toàn diện biến một bức ảnh số thành bản "bệnh án kỹ thuật"
gồm: độ sáng, độ tương phản, mức độ nhiễu, độ sắc nét/mờ, phân bố histogram
RGB/HSV và phát hiện ám màu qua không gian màu CIE L*a*b*.

Thiết kế theo ADR-001: Không dùng generative AI, chỉ dùng OpenCV/NumPy thuần túy.
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .measurement import measure_image
from .measurement.calibration import calibration_version, defect_severities, load_calibration
from .schemas import TechnicalMetrics

logger = logging.getLogger("img_doctor.analyzer_evaluator")

# ---------------------------------------------------------------------------
# Ngưỡng phân loại (Thresholds)
# ---------------------------------------------------------------------------
# Các mức *_level suy ra từ bảng hiệu chỉnh severity (measurement/calibration.yaml, quyết định D4
# của Analysis v2: một nguồn ngưỡng duy nhất), không còn ngưỡng tay ở đây.
#
# Hằng số độ sáng/clipping dưới đây CHỈ còn cho phép đo độ sáng theo vùng của
# src/agent/perception.py (thuộc lane agent/fusion), sẽ chuyển sang calibration ở P6.
_BRIGHTNESS_LOW = 70.0
_BRIGHTNESS_HIGH = 185.0
_SHADOW_CLIP_THRESH = 5
_HIGHLIGHT_CLIP_THRESH = 250
_SHADOW_CLIP_RATIO_THRESH = 0.25
_HIGHLIGHT_CLIP_RATIO_THRESH = 0.20

# Mức severity → nhãn level của TechnicalMetrics
_NOISE_LEVELS = ("clean", "low", "medium", "severe")
_COLOR_CAST_LEVELS = (
    ("color_cast_warm", "warm"),
    ("color_cast_cool", "cool"),
    ("color_cast_green", "greenish"),
)
# Các lỗi mà TechnicalMetrics.defect_severity ghi lại (cùng tập lỗi các *_level biểu diễn)
LEVEL_DEFECTS = (
    "underexposed",
    "overexposed",
    "low_contrast",
    "noise",
    "blur",
    "color_cast_warm",
    "color_cast_cool",
    "color_cast_green",
)

# Cache severity theo nội dung ảnh: pipeline gọi analyze_image 2–3 lần mỗi vòng trên cùng một
# ảnh (analyze, evaluate ảnh hiện tại và ảnh trước), còn đo cục bộ tốn ~1 s/2 MP.
_SEVERITY_CACHE_SIZE = 8
_severity_cache: "OrderedDict[Tuple[Any, ...], Dict[str, int]]" = OrderedDict()

# Kernel Immerkær Fast Noise Variance Estimator (float32 để giảm băng thông bộ nhớ;
# các giá trị nguyên -2..4 biểu diễn chính xác tuyệt đối trong float32)
_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)

# Trục bins dùng chung cho mọi thống kê moments từ histogram (tránh cấp phát lặp lại)
_GRAY_BINS = np.arange(256, dtype=np.float64)


# ---------------------------------------------------------------------------
# Hàm nội bộ (Internal Helpers)
# ---------------------------------------------------------------------------


def _ensure_rgb_and_gray(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuẩn hóa ảnh đầu vào về định dạng RGB uint8 và trả về ảnh xám tương ứng.

    Xử lý tất cả các trường hợp đầu vào:
    - Grayscale (H, W) -> Chuyển sang RGB 3 kênh
    - Grayscale có kênh giữ (H, W, 1) -> Ép về (H, W) rồi chuyển sang RGB
    - RGBA (H, W, 4)   -> Tách bỏ kênh Alpha
    - float32 [0,1]    -> Ép về uint8 [0, 255]
    - RGB (H, W, 3)    -> Giữ nguyên

    Args:
        image: Ảnh đầu vào với kiểu dữ liệu và kênh bất kỳ.

    Returns:
        Tuple (rgb_image, gray_image): Ảnh RGB và ảnh xám chuẩn hóa uint8.
    """
    # Bước 1: Chuyển float về uint8 nếu cần
    if image.dtype in (np.float32, np.float64):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        logger.debug("Input image converted from float to uint8.")

    # Bước 2: Xử lý theo số kênh màu
    if image.ndim == 2:
        # Grayscale (H, W) -> RGB (H, W, 3)
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        gray = image
        logger.debug("Grayscale input detected, converted to RGB.")
    elif image.ndim == 3 and image.shape[2] == 1:
        # Grayscale giữ kênh (H, W, 1) -> ép về (H, W) rồi chuyển sang RGB
        gray = np.ascontiguousarray(image[:, :, 0])
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        logger.debug("Single-channel (H, W, 1) input detected, squeezed to grayscale.")
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA (H, W, 4) -> RGB (H, W, 3)
        rgb = image[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        logger.debug("RGBA input detected, alpha channel stripped.")
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = image
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}. Expected (H,W), (H,W,1), (H,W,3), or (H,W,4)."
        )

    return rgb, gray


def _estimate_noise_immerkaer(gray: np.ndarray) -> float:
    """
    Ước lượng độ lệch chuẩn nhiễu bằng toán tử Immerkær (Fast Noise Variance Estimator).

    Phương pháp: Tích chập ảnh xám với kernel phát hiện nhiễu 3x3, sau đó
    tính tổng giá trị tuyệt đối nhân với hệ số chuẩn hóa.

    Args:
        gray: Ảnh xám uint8 đầu vào.

    Returns:
        float: Ước lượng độ lệch chuẩn nhiễu. Đơn vị tương đương độ xám [0..255].
    """
    h, w = gray.shape
    if h < 5 or w < 5:
        # Fallback: Median Difference cho ảnh quá nhỏ
        median_filtered = cv2.medianBlur(gray, 3)
        noise_val = float(np.mean(cv2.absdiff(gray, median_filtered)))
        logger.debug("Image too small for Immerkaer, using median-diff fallback: %.4f", noise_val)
        return noise_val

    gray_f = gray.astype(np.float32)
    conv = cv2.filter2D(gray_f, cv2.CV_32F, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REPLICATE)
    sigma = (np.pi / 2.0) * (1.0 / (6.0 * (w - 2) * (h - 2))) * np.sum(np.abs(conv))
    logger.debug("Immerkaer noise estimate: %.4f", sigma)
    return float(sigma)


def _compute_gray_hist_stats(gray: np.ndarray) -> Dict[str, Any]:
    """
    Tính mọi thống kê ảnh xám chỉ trong MỘT lần duyệt pixel (single-pass histogram).

    Dựng histogram 256 bins bằng np.bincount (vòng lặp C duy nhất trên toàn ảnh),
    rồi suy ra toàn bộ đặc trưng từ histogram thay vì duyệt ảnh nhiều lần:
    min/max (bin khác 0 đầu/cuối), mean/std/skewness (moments), clipping
    (cộng bins đuôi), P1/P99 (tìm trên hàm phân bố tích lũy cumsum).

    Độ chính xác: tổng histogram là số nguyên chính xác tuyệt đối (< 2^53),
    moments tính trên 256 bins bằng float64 nên sai số ~1e-12, không đổi sau
    khi làm tròn 2-4 chữ số thập phân ở đầu ra.

    Args:
        gray: Ảnh xám uint8 (H, W), không rỗng.

    Returns:
        Dict với các khóa thô (chưa làm tròn): mean, std, skewness, gray_min,
        gray_max, highlight_clip_ratio, shadow_clip_ratio, p1, p99, dynamic_range.

    Raises:
        ValueError: Khi ảnh rỗng (không có bin nào khác 0).
    """
    total_f = float(gray.size)
    hist = np.bincount(gray.ravel(), minlength=256)
    nonzero = np.flatnonzero(hist)
    if nonzero.size == 0:
        raise ValueError("Cannot compute histogram stats of an empty image.")

    # --- Moments từ histogram (định nghĩa trực tiếp, không triệt tiêu số học) ---
    hist_f = hist.astype(np.float64)
    mean = float(((_GRAY_BINS * hist_f).sum()) / total_f)
    deviations = _GRAY_BINS - mean
    std = float(np.sqrt(((deviations**2) * hist_f).sum() / total_f))

    # Skewness: E[(X-mu)^3] / sigma^3, ảnh phẳng (std=0) trả về 0 chống NaN
    if std < 1e-8:
        skewness = 0.0
    else:
        skewness = float((((deviations**3) * hist_f).sum() / total_f) / (std**3))

    # --- Clipping từ bins đuôi (tương đương so sánh toàn ảnh, không cấp phát mask) ---
    highlight_clip_ratio = float(hist[250:].sum() / total_f)  # bins 250..255
    shadow_clip_ratio = float(hist[:6].sum() / total_f)  # bins 0..5

    # --- P1/P99 từ phân bố tích lũy (tránh sắp xếp O(N log N) của percentile) ---
    cumsum = np.cumsum(hist)
    p1 = int(np.searchsorted(cumsum, 0.01 * total_f, side="left"))
    p99 = int(np.searchsorted(cumsum, 0.99 * total_f, side="left"))

    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "gray_min": int(nonzero[0]),
        "gray_max": int(nonzero[-1]),
        "highlight_clip_ratio": highlight_clip_ratio,
        "shadow_clip_ratio": shadow_clip_ratio,
        "p1": p1,
        "p99": p99,
        "dynamic_range": p99 - p1,
    }


def _levels_from_severity(
    severity: Dict[str, int], contrast_std: float, contrast_high_std: float
) -> Dict[str, str]:
    """Nhãn *_level của TechnicalMetrics từ severity đã hiệu chỉnh (D4).

    Phơi sáng lấy chiều có severity cao hơn; ám màu lấy hướng có severity cao nhất (hòa: ấm,
    lạnh, xanh lá). "high" contrast không phải lỗi nên giữ ngưỡng prior trên độ lệch chuẩn xám
    (đọc từ calibration.yaml, mục levels).
    """
    under, over = severity.get("underexposed", 0), severity.get("overexposed", 0)
    if under >= 1 and under >= over:
        brightness = "underexposed"
    elif over >= 1:
        brightness = "overexposed"
    else:
        brightness = "normal"

    if severity.get("low_contrast", 0) >= 1:
        contrast = "low"
    elif contrast_std > contrast_high_std:
        contrast = "high"
    else:
        contrast = "normal"

    blur = severity.get("blur", 0)
    cast, cast_severity = "none", 0
    for defect, label in _COLOR_CAST_LEVELS:
        if severity.get(defect, 0) > cast_severity:
            cast, cast_severity = label, severity[defect]
    return {
        "brightness_level": brightness,
        "contrast_level": contrast,
        "noise_level": _NOISE_LEVELS[min(severity.get("noise", 0), 3)],
        "blur_level": "sharp" if blur == 0 else "mild_blur" if blur == 1 else "severe_blur",
        "color_cast": cast,
    }


def clear_severity_cache() -> None:
    """Xóa cache severity theo nội dung ảnh (dùng trong test và khi nạp lại calibration)."""
    _severity_cache.clear()


def _calibrated_severity(rgb: np.ndarray) -> Dict[str, int]:
    """Severity hiệu chỉnh của các lỗi LEVEL_DEFECTS, cache theo băm nội dung ảnh."""
    key = (
        rgb.shape,
        hashlib.blake2b(np.ascontiguousarray(rgb).tobytes(), digest_size=16).hexdigest(),
        calibration_version(),
    )
    cached = _severity_cache.get(key)
    if cached is not None:
        _severity_cache.move_to_end(key)
        return dict(cached)
    report = measure_image(rgb).report
    severity = {
        defect: level
        for defect, (level, _) in defect_severities(report).items()
        if defect in LEVEL_DEFECTS
    }
    _severity_cache[key] = severity
    while len(_severity_cache) > _SEVERITY_CACHE_SIZE:
        _severity_cache.popitem(last=False)
    return dict(severity)


def _compute_histogram_stats(
    rgb: np.ndarray,
    gray: np.ndarray,
    bgr: Optional[np.ndarray] = None,
    gray_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tính toán bộ thống kê histogram toàn diện theo kế hoạch Stage 1.6.

    Bao gồm:
    - Thống kê ảnh xám: min, max, skewness, highlight/shadow clipping, dynamic range
    - Thống kê per-channel RGB: mean, std của kênh R, G, B
    - Dominant Hue từ không gian HSV

    Tối ưu: toàn bộ thống kê xám lấy từ single-pass histogram
    (`_compute_gray_hist_stats`); mean/std RGB lấy từ `cv2.meanStdDev`
    (một lần duyệt C, không sao chép ảnh sang float).

    Args:
        rgb:  Ảnh RGB uint8 (H, W, 3).
        gray: Ảnh xám uint8 (H, W).
        bgr:  Ảnh BGR uint8 đã tính trước (tùy chọn). Nếu None, sẽ tính lại từ rgb.
        gray_stats: Kết quả `_compute_gray_hist_stats(gray)` đã tính trước
            (tùy chọn). Nếu None, sẽ tự tính từ gray.

    Returns:
        Dict[str, Any]: Từ điển với các trường thống kê đã xác định.
    """
    if gray_stats is None:
        gray_stats = _compute_gray_hist_stats(gray)

    gray_min = gray_stats["gray_min"]
    gray_max = gray_stats["gray_max"]
    gray_skewness = gray_stats["skewness"]
    highlight_clip_ratio = gray_stats["highlight_clip_ratio"]
    shadow_clip_ratio = gray_stats["shadow_clip_ratio"]
    dynamic_range = gray_stats["dynamic_range"]

    # --- Thống kê per-channel RGB (một lần duyệt C, không sao chép float) ---
    ch_mean_mat, ch_std_mat = cv2.meanStdDev(rgb)
    ch_mean = ch_mean_mat.ravel()  # shape (3,)
    ch_std = ch_std_mat.ravel()  # shape (3,)

    # --- Dominant Hue từ HSV (tái sử dụng BGR nếu có) ---
    if bgr is None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_channel = hsv[:, :, 0]  # Kênh H: 0..180 trong OpenCV
    hist_hue = cv2.calcHist([hue_channel], [0], None, [180], [0, 180]).flatten()
    dominant_hue = float(np.argmax(hist_hue))

    return {
        # Ảnh xám
        "gray_min": gray_min,
        "gray_max": gray_max,
        "gray_skewness": round(gray_skewness, 4),
        "highlight_clip_ratio": round(highlight_clip_ratio, 4),
        "shadow_clip_ratio": round(shadow_clip_ratio, 4),
        "dynamic_range": dynamic_range,
        # Per-channel RGB
        "r_mean": round(float(ch_mean[0]), 2),
        "g_mean": round(float(ch_mean[1]), 2),
        "b_mean": round(float(ch_mean[2]), 2),
        "r_std": round(float(ch_std[0]), 2),
        "g_std": round(float(ch_std[1]), 2),
        "b_std": round(float(ch_std[2]), 2),
        # HSV dominant hue (góc màu 0..180)
        "dominant_hue": dominant_hue,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_image(image: np.ndarray) -> TechnicalMetrics:
    """
    Phân tích định lượng toàn diện các thuộc tính kỹ thuật của bức ảnh.

    Các bước:
    1. Chuẩn hóa đầu vào và chuyển đổi không gian màu
    2. Độ sáng, độ tương phản (thống kê histogram xám)
    3. Nhiễu Immerkær và độ nét Laplacian/Tenengrad (chỉ số thô, nhánh đánh giá vẫn dùng)
    4. Histogram RGB/HSV
    5. Các mức *_level và `defect_severity`: đo cục bộ (`measurement.measure_image`) rồi tra bảng
       hiệu chỉnh severity (`measurement/calibration.yaml`, quyết định D4: một nguồn ngưỡng duy
       nhất, `knowledge/engine._metric_defects` dùng lại cùng severity)

    Tối ưu hóa hiệu năng:
    - Mọi thống kê ảnh xám (mean, std, skewness, clipping, P1/P99) suy ra từ
      MỘT histogram duy nhất (`_compute_gray_hist_stats`).
    - Mean/std RGB lấy từ `cv2.meanStdDev` (một lần duyệt C, không sao chép).

    Args:
        image: Ảnh đầu vào. Hỗ trợ: uint8 hoặc float32, RGB/Grayscale/RGBA.

    Returns:
        TechnicalMetrics: Cấu trúc dữ liệu chứa toàn bộ chỉ số đã trích xuất.

    Raises:
        ValueError: Khi shape ảnh không được hỗ trợ.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty array.")

    logger.debug("analyze_image called: shape=%s dtype=%s", image.shape, image.dtype)

    # Bước 1: Chuẩn hóa đầu vào; mọi thống kê xám tái sử dụng một histogram
    rgb, gray = _ensure_rgb_and_gray(image)
    gray_stats = _compute_gray_hist_stats(gray)

    # Bước 2: Độ sáng và độ tương phản
    brightness_mean = gray_stats["mean"]
    contrast_std = gray_stats["std"]

    # Bước 3: Chỉ số thô nhiễu (Immerkær) và độ nét (Laplacian; ảnh phẳng cho đúng 0 tuyệt đối)
    noise_variance = _estimate_noise_immerkaer(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness_laplacian_var = float(laplacian.var())
    # Tenengrad chỉ dùng cho log debug, tính trên ảnh thu nhỏ 1/2 (ảnh < 4 px giữ nguyên)
    gh, gw = gray.shape
    if gh >= 4 and gw >= 4:
        small_gray = cv2.resize(gray, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray
    sobelx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobelx**2 + sobely**2))

    # Bước 4: Histogram RGB/HSV
    histogram_stats = _compute_histogram_stats(rgb, gray, gray_stats=gray_stats)

    # Bước 5: Severity đã hiệu chỉnh → *_level (D4)
    severity = _calibrated_severity(rgb)
    contrast_high_std = float(load_calibration().get("levels", {}).get("contrast_high_std", 80.0))
    levels = _levels_from_severity(severity, contrast_std, contrast_high_std)

    logger.debug(
        "Raw metrics: brightness=%.2f contrast_std=%.2f immerkaer=%.4f laplacian_var=%.2f "
        "tenengrad=%.2f",
        brightness_mean,
        contrast_std,
        noise_variance,
        sharpness_laplacian_var,
        tenengrad,
    )
    logger.info(
        "Image diagnosed: brightness=%s contrast=%s noise=%s blur=%s color_cast=%s",
        levels["brightness_level"],
        levels["contrast_level"],
        levels["noise_level"],
        levels["blur_level"],
        levels["color_cast"],
    )

    return TechnicalMetrics(
        brightness_mean=round(brightness_mean, 2),
        contrast_std=round(contrast_std, 2),
        noise_variance=round(noise_variance, 4),
        sharpness_laplacian_var=round(sharpness_laplacian_var, 2),
        histogram_stats=histogram_stats,
        defect_severity=severity,
        calibration_version=calibration_version(),
        **levels,
    )
