"""
Đánh giá chất lượng ảnh dựa trên ảnh chuẩn (Full-Reference Evaluation).

Chỉ áp dụng cho tập kiểm thử nhân tạo (Synthetic Benchmark) có ground-truth.
Tuân thủ ADR-002: Không bao giờ gọi hàm này trên ảnh thực tế tải lên từ người dùng.
"""

import logging

import cv2
import numpy as np

from .analyzer import _ensure_rgb_and_gray, analyze_image
from .schemas import EvaluationResult

logger = logging.getLogger("img_doctor.analyzer_evaluator")

try:
    from skimage.metrics import peak_signal_noise_ratio as _skimage_psnr
    from skimage.metrics import structural_similarity as _skimage_ssim

    SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover - chỉ xảy ra khi gỡ scikit-image
    SKIMAGE_AVAILABLE = False
    logger.warning(
        "scikit-image not available. PSNR/SSIM will use built-in fallback implementations."
    )


# ---------------------------------------------------------------------------
# Hàm nội bộ (Internal Helpers)
# ---------------------------------------------------------------------------


def _compute_mse(ground_truth: np.ndarray, current: np.ndarray) -> float:
    """
    Tính toán Mean Squared Error (MSE) giữa hai ảnh.

    Ép kiểu sang float64 trước khi trừ để tránh underflow/overflow của uint8
    (ví dụ: 50 - 100 sẽ vòng lại thành 206 trong modulo C).

    Args:
        ground_truth: Ảnh gốc sạch ground-truth (RGB, uint8).
        current: Ảnh đã qua xử lý (RGB, uint8).

    Returns:
        float: Giá trị MSE trung bình trên tất cả kênh màu.
    """
    gt_f = ground_truth.astype(np.float64)
    curr_f = current.astype(np.float64)
    mse = float(np.mean((gt_f - curr_f) ** 2))
    logger.debug("MSE computed: %.6f", mse)
    return mse


def _compute_psnr(ground_truth: np.ndarray, current: np.ndarray, mse: float) -> float:
    """
    Tính toán Peak Signal-to-Noise Ratio (PSNR) tính bằng decibel (dB).

    Công thức: PSNR = 20 * log10(255 / sqrt(MSE))
    Nếu MSE = 0 (hai ảnh hoàn toàn trùng khớp) → trả về 100.0 dB (giá trị trần quy ước).

    Args:
        ground_truth: Ảnh gốc sạch ground-truth (RGB, uint8).
        current: Ảnh đã qua xử lý (RGB, uint8).
        mse: Giá trị MSE đã tính từ _compute_mse().

    Returns:
        float: Giá trị PSNR tính bằng dB. Trả về 100.0 nếu MSE < 1e-10.
    """
    if mse < 1e-10:
        logger.debug("MSE is effectively zero, returning perfect PSNR=100.0 dB.")
        return 100.0

    if SKIMAGE_AVAILABLE:
        psnr = float(_skimage_psnr(ground_truth, current, data_range=255))
    else:
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        logger.debug("scikit-image unavailable, using manual PSNR formula.")

    logger.debug("PSNR computed: %.4f dB", psnr)
    return psnr


def _compute_ssim(ground_truth: np.ndarray, current: np.ndarray) -> float:
    """
    Tính toán Structural Similarity Index Measure (SSIM).

    Sử dụng multichannel SSIM (channel_axis=2) cho ảnh màu RGB.
    Tự động co giãn win_size nếu ảnh nhỏ hơn cửa sổ mặc định.

    Fallback thủ công nếu scikit-image không có:
        SSIM = [(2*mu1*mu2 + C1)(2*sigma12 + C2)] / [(mu1² + mu2² + C1)(sigma1² + sigma2² + C2)]

    Args:
        ground_truth: Ảnh gốc sạch ground-truth (RGB, uint8).
        current: Ảnh đã qua xử lý (RGB, uint8).

    Returns:
        float: Giá trị SSIM trong khoảng [-1, 1]. 1.0 là trùng khớp hoàn hảo.
    """
    min_dim = min(ground_truth.shape[0], ground_truth.shape[1])
    if SKIMAGE_AVAILABLE and min_dim >= 3:
        # Cửa sổ mặc định 7x7 — thu nhỏ nếu ảnh quá nhỏ (scikit-image yêu cầu win_size >= 3)
        win_size = min(7, min_dim)
        # win_size phải lẻ
        if win_size % 2 == 0:
            win_size = max(3, win_size - 1)

        channel_axis = 2 if ground_truth.ndim == 3 else None
        ssim = float(
            _skimage_ssim(
                ground_truth,
                current,
                data_range=255,
                channel_axis=channel_axis,
                win_size=win_size,
            )
        )
        logger.debug("SSIM computed (scikit-image, win_size=%d): %.6f", win_size, ssim)
    else:
        # Fallback: SSIM đơn giản toàn cục (không sliding window) hoặc cho ảnh nhỏ < 3x3
        logger.debug(
            "Using global SSIM fallback (SKIMAGE_AVAILABLE=%s, min_dim=%d).",
            SKIMAGE_AVAILABLE,
            min_dim,
        )
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        img1 = ground_truth.astype(np.float64)
        img2 = current.astype(np.float64)
        mu1 = float(np.mean(img1))
        mu2 = float(np.mean(img2))
        sigma1_sq = float(np.var(img1))
        sigma2_sq = float(np.var(img2))
        sigma12 = float(np.mean((img1 - mu1) * (img2 - mu2)))
        ssim = float(
            ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2))
            / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        )

    return ssim


def _ensure_shape_match(
    ground_truth: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Đảm bảo hai ảnh có cùng không gian màu RGB và cùng kích thước trước khi tính metrics.

    Chuẩn hóa số kênh qua _ensure_rgb_and_gray để tránh lỗi broadcast khi một ảnh là grayscale/RGBA.
    Nếu kích thước (H, W) lệch nhau, tự động resize ground_truth về khớp current bằng INTER_AREA.

    Args:
        ground_truth: Ảnh gốc sạch ground-truth.
        current: Ảnh đã qua xử lý (kích thước chuẩn tham chiếu).

    Returns:
        Tuple (ground_truth_resized, current): Cặp ảnh cùng kích thước và cùng 3 kênh RGB.
    """
    ground_truth, _ = _ensure_rgb_and_gray(ground_truth)
    current, _ = _ensure_rgb_and_gray(current)

    if ground_truth.shape != current.shape:
        logger.warning(
            "Shape mismatch: ground_truth=%s vs current=%s. "
            "Resizing ground_truth to match current image shape.",
            ground_truth.shape,
            current.shape,
        )
        ground_truth = cv2.resize(
            ground_truth,
            (current.shape[1], current.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return ground_truth, current


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_reference(
    current_image: np.ndarray,
    ground_truth_image: np.ndarray,
    iteration: int = 1,
    previous_image: np.ndarray | None = None,
) -> EvaluationResult:
    """
    Đánh giá toàn diện chất lượng ảnh bằng Full-Reference Metrics (PSNR, SSIM, MSE).

    Dùng cho tập synthetic benchmark có ground-truth. Tuân thủ ADR-002:
    Không bao giờ gọi hàm này trên ảnh thực tế (is_synthetic=False).

    Quy trình:
    1. Kiểm tra và đồng bộ kích thước hai ảnh (Shape Guard).
    2. Tính MSE (Mean Squared Error) với float64 anti-underflow.
    3. Tính PSNR (Peak Signal-to-Noise Ratio) với xử lý biên MSE=0.
    4. Tính Multichannel SSIM với win_size tự động co giãn.
    5. Gọi analyze_image() để lấy TechnicalMetrics của ảnh kết quả.
    6. Tính delta_metrics nếu có previous_image.
    7. Xác định quality_improved dựa trên PSNR/SSIM delta.

    Args:
        current_image: Ảnh đã qua xử lý (RGB, uint8).
        ground_truth_image: Ảnh gốc sạch ground-truth (RGB, uint8).
        iteration: Số thứ tự vòng lặp hiện tại (1..3).
        previous_image: Ảnh vòng lặp trước (hoặc ảnh đầu vào ban đầu) để tính delta.

    Returns:
        EvaluationResult: Kết quả kiểm định đầy đủ với PSNR, SSIM, MSE,
            TechnicalMetrics và delta_metrics.
    """
    if current_image is None or current_image.size == 0:
        raise ValueError("current_image is None or empty.")
    if ground_truth_image is None or ground_truth_image.size == 0:
        raise ValueError("ground_truth_image is None or empty.")

    logger.debug(
        "evaluate_reference called: current=%s gt=%s iteration=%d",
        current_image.shape,
        ground_truth_image.shape,
        iteration,
    )

    # -----------------------------------------------------------------------
    # Bước 1: Shape Guard — đồng bộ kích thước
    # -----------------------------------------------------------------------
    ground_truth_image, current_image = _ensure_shape_match(ground_truth_image, current_image)

    # -----------------------------------------------------------------------
    # Bước 2-4: Tính MSE, PSNR, SSIM
    # -----------------------------------------------------------------------
    mse_val = _compute_mse(ground_truth_image, current_image)
    psnr_val = _compute_psnr(ground_truth_image, current_image, mse_val)
    ssim_val = _compute_ssim(ground_truth_image, current_image)

    logger.info(
        "Full-reference evaluation (iter=%d): MSE=%.4f PSNR=%.2f dB SSIM=%.4f",
        iteration,
        mse_val,
        psnr_val,
        ssim_val,
    )

    # -----------------------------------------------------------------------
    # Bước 5: TechnicalMetrics của ảnh đã xử lý
    # -----------------------------------------------------------------------
    tech_metrics = analyze_image(current_image)

    # -----------------------------------------------------------------------
    # Bước 6: Delta Metrics so với vòng lặp trước
    # -----------------------------------------------------------------------
    delta_metrics: dict[str, float] = {}
    quality_improved = True

    if previous_image is not None:
        prev_mse = _compute_mse(ground_truth_image, previous_image)
        prev_psnr = _compute_psnr(ground_truth_image, previous_image, prev_mse)
        prev_ssim = _compute_ssim(ground_truth_image, previous_image)

        delta_psnr = psnr_val - prev_psnr
        delta_ssim = ssim_val - prev_ssim
        delta_mse = mse_val - prev_mse

        delta_metrics = {
            "delta_psnr": round(delta_psnr, 4),
            "delta_ssim": round(delta_ssim, 6),
            "delta_mse": round(delta_mse, 4),
        }

        # Chất lượng cải thiện khi PSNR tăng và/hoặc SSIM tăng
        quality_improved = delta_psnr >= 0.0 or delta_ssim >= 0.0

        logger.debug(
            "Delta metrics: ΔPSNR=%.4f ΔSSIM=%.6f ΔMSE=%.4f | quality_improved=%s",
            delta_psnr,
            delta_ssim,
            delta_mse,
            quality_improved,
        )

    return EvaluationResult(
        iteration=iteration,
        is_reference_eval=True,
        psnr=round(psnr_val, 4),
        ssim=round(ssim_val, 6),
        mse=round(mse_val, 4),
        technical_metrics=tech_metrics,
        delta_metrics=delta_metrics,
        quality_improved=quality_improved,
    )
