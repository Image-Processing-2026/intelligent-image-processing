"""
Trích xuất chỉ số kỹ thuật ảnh (Technical Metrics Analyzer).

Cung cấp engine phân tích toàn diện biến một bức ảnh số thành bản "bệnh án kỹ thuật"
gồm: độ sáng, độ tương phản, mức độ nhiễu, độ sắc nét/mờ, phân bố histogram
RGB/HSV và phát hiện ám màu qua không gian màu CIE L*a*b*.

Thiết kế theo ADR-001: Không dùng generative AI, chỉ dùng OpenCV/NumPy thuần túy.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .schemas import TechnicalMetrics

logger = logging.getLogger("img_doctor.analyzer_evaluator")

# ---------------------------------------------------------------------------
# Hằng số ngưỡng phân loại (Thresholds)
# ---------------------------------------------------------------------------
_BRIGHTNESS_LOW = 70.0
_BRIGHTNESS_HIGH = 185.0
_SHADOW_CLIP_THRESH = 5
_HIGHLIGHT_CLIP_THRESH = 250
_SHADOW_CLIP_RATIO_THRESH = 0.25
_HIGHLIGHT_CLIP_RATIO_THRESH = 0.20

_CONTRAST_LOW = 40.0
_CONTRAST_HIGH = 80.0

_NOISE_CLEAN = 3.0
_NOISE_LOW = 8.0
_NOISE_MEDIUM = 15.0

_BLUR_SEVERE = 100.0
_BLUR_MILD = 300.0

# Ngưỡng ám màu LAB và RGB
# Lưu ý: OpenCV LAB encode a*/b* trong [0, 255] với 128 = trung tính
# a* < 128 = xanh lá | a* > 128 = đỏ
# b* < 128 = xanh lam | b* > 128 = vàng/ấm
_LAB_GREEN_A_THRESH = 110.0  # a* < 110 → thiên xanh lá rõ rệt
_LAB_WARM_B_THRESH = 140.0  # b* > 140 → thiên vàng/ấm (chỉ khi a* bình thường)
_LAB_COOL_B_THRESH = 118.0  # b* < 118 → thiên xanh lam/lạnh
_RGB_CHANNEL_DIFF = 20.0
_RGB_GREEN_DIFF = 15.0

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


def _classify_noise(sigma: float) -> str:
    """Phân loại mức độ nhiễu dựa trên ngưỡng đã định nghĩa."""
    if sigma < _NOISE_CLEAN:
        return "clean"
    elif sigma < _NOISE_LOW:
        return "low"
    elif sigma < _NOISE_MEDIUM:
        return "medium"
    return "severe"


def _detect_color_cast(rgb: np.ndarray, bgr: Optional[np.ndarray] = None) -> str:
    """
    Phát hiện ám màu chủ đạo của ảnh qua không gian màu CIE L*a*b*.

    Chuyển ảnh RGB sang LAB, phân tích tâm trọng lực kênh a* và b* để xác định:
    - warm   : Ám vàng/đỏ ấm (b* cao hoặc R >> B)
    - cool   : Ám xanh lam lạnh (b* thấp hoặc B >> R)
    - greenish: Ám xanh lá (a* thấp và G >> R)
    - none   : Cân bằng màu tốt

    Args:
        rgb: Ảnh RGB uint8 (H, W, 3).
        bgr: Ảnh BGR uint8 đã tính trước (tùy chọn). Nếu None, sẽ tính lại từ rgb.

    Returns:
        str: Loại ám màu ('warm', 'cool', 'greenish', 'none').
    """
    # Tái sử dụng BGR nếu đã được tính trước để tránh chuyển đổi trùng lặp
    if bgr is None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Lấy mean trực tiếp trên uint8 (tích lũy float64 pairwise, không cần sao chép float)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    mu_a = float(np.mean(lab[:, :, 1]))  # Kênh a*: Trục Lục-Đỏ
    mu_b = float(np.mean(lab[:, :, 2]))  # Kênh b*: Trục Lam-Vàng

    # Tính mean RGB để hỗ trợ quyết định
    mu_r = float(np.mean(rgb[:, :, 0]))
    mu_g = float(np.mean(rgb[:, :, 1]))
    mu_b_rgb = float(np.mean(rgb[:, :, 2]))

    logger.debug(
        "Color cast analysis: LAB a*=%.2f b*=%.2f | RGB R=%.2f G=%.2f B=%.2f",
        mu_a,
        mu_b,
        mu_r,
        mu_g,
        mu_b_rgb,
    )

    # Thứ tự ưu tiên: greenish trước (a* là chỉ số rõ nhất),
    # sau đó warm/cool theo b*
    if mu_a < _LAB_GREEN_A_THRESH and (mu_g - mu_r) > _RGB_GREEN_DIFF:
        return "greenish"
    if mu_b > _LAB_WARM_B_THRESH or (mu_r - mu_b_rgb) > _RGB_CHANNEL_DIFF:
        return "warm"
    if mu_b < _LAB_COOL_B_THRESH or (mu_b_rgb - mu_r) > _RGB_CHANNEL_DIFF:
        return "cool"
    return "none"


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

    Thực hiện 6 phép đo theo kế hoạch Stage 1:
    1. Chuẩn hóa đầu vào và chuyển đổi không gian màu
    2. Đo độ sáng và phân tích phơi sáng (Brightness & Exposure)
    3. Phân tích độ tương phản và dải động (Contrast Dynamics)
    4. Ước lượng mức độ nhiễu bằng Immerkær estimator
    5. Đo độ sắc nét/mờ bằng Laplacian và Tenengrad
    6. Phân tích histogram RGB/HSV và phát hiện ám màu CIE LAB

    Tối ưu hóa hiệu năng:
    - Mọi thống kê ảnh xám (mean, std, skewness, clipping, P1/P99) suy ra từ
      MỘT histogram duy nhất (`_compute_gray_hist_stats`), loại bỏ sắp xếp
      O(N log N) của percentile và mọi bản sao float của ảnh xám.
    - Mean/std RGB lấy từ `cv2.meanStdDev` (một lần duyệt C, không sao chép).
    - Tính BGR một lần, chia sẻ cho cả _detect_color_cast và _compute_histogram_stats
      để loại bỏ chuyển đổi không gian màu trùng lặp.
    - Immerkær/Sobel/LAB dùng float32 (đủ chính xác so với biên ngưỡng phân loại).

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

    # -----------------------------------------------------------------------
    # Bước 1: Chuẩn hóa đầu vào
    # -----------------------------------------------------------------------
    rgb, gray = _ensure_rgb_and_gray(image)

    # -----------------------------------------------------------------------
    # Single-pass histogram: mọi thống kê xám bên dưới tái sử dụng kết quả này
    # -----------------------------------------------------------------------
    gray_stats = _compute_gray_hist_stats(gray)

    # -----------------------------------------------------------------------
    # Bước 2: Độ sáng và Phơi sáng (Brightness & Exposure Analysis)
    # -----------------------------------------------------------------------
    brightness_mean = gray_stats["mean"]
    highlight_ratio = gray_stats["highlight_clip_ratio"]
    shadow_ratio = gray_stats["shadow_clip_ratio"]

    if brightness_mean < _BRIGHTNESS_LOW or shadow_ratio > _SHADOW_CLIP_RATIO_THRESH:
        brightness_level = "underexposed"
    elif brightness_mean > _BRIGHTNESS_HIGH or highlight_ratio > _HIGHLIGHT_CLIP_RATIO_THRESH:
        brightness_level = "overexposed"
    else:
        brightness_level = "normal"

    logger.debug(
        "Brightness: mean=%.2f level=%s (highlight_ratio=%.3f shadow_ratio=%.3f)",
        brightness_mean,
        brightness_level,
        highlight_ratio,
        shadow_ratio,
    )

    # -----------------------------------------------------------------------
    # Bước 3: Độ tương phản và Dải động (Contrast Dynamics)
    # -----------------------------------------------------------------------
    contrast_std = gray_stats["std"]

    if contrast_std < _CONTRAST_LOW:
        contrast_level = "low"
    elif contrast_std > _CONTRAST_HIGH:
        contrast_level = "high"
    else:
        contrast_level = "normal"

    logger.debug("Contrast: std=%.2f level=%s", contrast_std, contrast_level)

    # -----------------------------------------------------------------------
    # Bước 4: Ước lượng Nhiễu (Noise Estimation via Immerkær)
    # -----------------------------------------------------------------------
    noise_variance = _estimate_noise_immerkaer(gray)
    noise_level = _classify_noise(noise_variance)

    logger.debug("Noise: variance=%.4f level=%s", noise_variance, noise_level)

    # -----------------------------------------------------------------------
    # Bước 5: Độ Sắc Nét / Mờ (Sharpness & Blur via Laplacian + Tenengrad)
    # -----------------------------------------------------------------------
    # Laplacian variance — phép đo chính (float32: ảnh phẳng cho đúng 0 tuyệt đối;
    # ngưỡng phân loại 100/300 cách xa sai số tương đối ~1% của float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness_laplacian_var = float(laplacian.var())

    # Tenengrad — bổ trợ phát hiện nhiễu giả sắc nét (chỉ dùng cho log debug,
    # nên tính trên ảnh thu nhỏ 1/2 để tiết kiệm 3/4 chi phí tích chập Sobel;
    # ảnh tí hon (< 4px) giữ nguyên để tránh kích thước 0 sau khi chia)
    gh, gw = gray.shape
    if gh >= 4 and gw >= 4:
        small_gray = cv2.resize(gray, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray
    sobelx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobelx**2 + sobely**2))

    if sharpness_laplacian_var < _BLUR_SEVERE:
        blur_level = "severe_blur"
    elif sharpness_laplacian_var < _BLUR_MILD:
        blur_level = "mild_blur"
    else:
        blur_level = "sharp"

    logger.debug(
        "Sharpness: laplacian_var=%.2f tenengrad=%.2f blur_level=%s",
        sharpness_laplacian_var,
        tenengrad,
        blur_level,
    )

    # -----------------------------------------------------------------------
    # Bước 6: Histogram RGB/HSV và Ám Màu CIE LAB (Color Cast Detection)
    # Tối ưu: tính BGR một lần, chia sẻ cho cả hai hàm con;
    # thống kê xám tái sử dụng gray_stats (không duyệt ảnh lại)
    # -----------------------------------------------------------------------
    bgr_shared = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color_cast = _detect_color_cast(rgb, bgr=bgr_shared)
    histogram_stats = _compute_histogram_stats(rgb, gray, bgr=bgr_shared, gray_stats=gray_stats)

    logger.info(
        "Image diagnosed: brightness=%s contrast=%s noise=%s blur=%s color_cast=%s",
        brightness_level,
        contrast_level,
        noise_level,
        blur_level,
        color_cast,
    )

    return TechnicalMetrics(
        brightness_mean=round(brightness_mean, 2),
        brightness_level=brightness_level,
        contrast_std=round(contrast_std, 2),
        contrast_level=contrast_level,
        noise_variance=round(noise_variance, 4),
        noise_level=noise_level,
        sharpness_laplacian_var=round(sharpness_laplacian_var, 2),
        blur_level=blur_level,
        color_cast=color_cast,
        histogram_stats=histogram_stats,
    )
