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

# Kernel Immerkær Fast Noise Variance Estimator
_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Hàm nội bộ (Internal Helpers)
# ---------------------------------------------------------------------------


def _ensure_rgb_and_gray(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuẩn hóa ảnh đầu vào về định dạng RGB uint8 và trả về ảnh xám tương ứng.

    Xử lý tất cả các trường hợp đầu vào:
    - Grayscale (H, W) -> Chuyển sang RGB 3 kênh
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
            f"Unsupported image shape: {image.shape}. Expected (H,W), (H,W,3), or (H,W,4)."
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

    gray_f = gray.astype(np.float64)
    conv = cv2.filter2D(gray_f, cv2.CV_64F, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REPLICATE)
    sigma = (np.pi / 2.0) * (1.0 / (6.0 * (w - 2) * (h - 2))) * np.sum(np.abs(conv))
    logger.debug("Immerkaer noise estimate: %.4f", sigma)
    return float(sigma)


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
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

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
    rgb: np.ndarray, gray: np.ndarray, bgr: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Tính toán bộ thống kê histogram toàn diện theo kế hoạch Stage 1.6.

    Bao gồm:
    - Thống kê ảnh xám: min, max, skewness, highlight/shadow clipping, dynamic range
    - Thống kê per-channel RGB: mean, std của kênh R, G, B
    - Dominant Hue từ không gian HSV

    Args:
        rgb:  Ảnh RGB uint8 (H, W, 3).
        gray: Ảnh xám uint8 (H, W).
        bgr:  Ảnh BGR uint8 đã tính trước (tùy chọn). Nếu None, sẽ tính lại từ rgb.

    Returns:
        Dict[str, Any]: Từ điển với các trường thống kê đã xác định.
    """
    total_pixels = float(gray.size)
    # Dùng float32 thay float64 để giảm băng thông bộ nhớ trên ảnh lớn
    gray_f = gray.astype(np.float32)

    # --- Thống kê ảnh xám ---
    gray_min = int(np.min(gray))
    gray_max = int(np.max(gray))
    gray_mean = float(np.mean(gray_f))
    gray_std = float(np.std(gray_f))

    # Skewness: E[(X-mu)^3] / sigma^3, phòng trường hợp ảnh phẳng sigma=0
    if gray_std < 1e-8:
        gray_skewness = 0.0
    else:
        gray_skewness = float(np.mean((gray_f - gray_mean) ** 3) / (gray_std**3))

    # Tỷ lệ clipping
    highlight_clip_ratio = float(np.sum(gray >= _HIGHLIGHT_CLIP_THRESH) / total_pixels)
    shadow_clip_ratio = float(np.sum(gray <= _SHADOW_CLIP_THRESH) / total_pixels)

    # Dynamic range thực tế: P99 - P1
    p1 = float(np.percentile(gray_f, 1))
    p99 = float(np.percentile(gray_f, 99))
    dynamic_range = int(round(p99 - p1))

    # --- Thống kê per-channel RGB (vectorized: tính tất cả kênh cùng lúc) ---
    rgb_f = rgb.astype(np.float32)
    ch_mean = rgb_f.mean(axis=(0, 1))  # shape (3,)
    ch_std = rgb_f.std(axis=(0, 1))  # shape (3,)

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
    - Tính BGR một lần, chia sẻ cho cả _detect_color_cast và _compute_histogram_stats
      để loại bỏ chuyển đổi không gian màu trùng lặp.
    - Dùng float32 thay float64 cho các thống kê per-channel RGB.

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
    total_pixels = float(gray.size)

    # -----------------------------------------------------------------------
    # Bước 2: Độ sáng và Phơi sáng (Brightness & Exposure Analysis)
    # -----------------------------------------------------------------------
    brightness_mean = float(np.mean(gray))
    highlight_ratio = float(np.sum(gray >= _HIGHLIGHT_CLIP_THRESH) / total_pixels)
    shadow_ratio = float(np.sum(gray <= _SHADOW_CLIP_THRESH) / total_pixels)

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
    contrast_std = float(np.std(gray.astype(np.float32)))

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
    # Laplacian variance — phép đo chính
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_laplacian_var = float(laplacian.var())

    # Tenengrad — bổ trợ phát hiện nhiễu giả sắc nét
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
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
    # Tối ưu: tính BGR một lần, chia sẻ cho cả hai hàm con
    # -----------------------------------------------------------------------
    bgr_shared = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color_cast = _detect_color_cast(rgb, bgr=bgr_shared)
    histogram_stats = _compute_histogram_stats(rgb, gray, bgr=bgr_shared)

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
