"""
Đánh giá chất lượng ảnh không cần ảnh gốc (No-Reference Image Quality Assessment).

Áp dụng cho ảnh thực tế tải lên từ người dùng — tuân thủ ADR-002:
Không bao giờ dùng Full-Reference metrics (PSNR/SSIM) trên ảnh thực tế.

Chiến lược đa tầng (Multi-Tier Strategy):
  Tier 1: pyiqa BRISQUE + NIQE (nếu cài đặt và model sẵn sàng).
  Tier 2: Heuristic Natural Scene Composite Index — luôn hoạt động 100%
           không phụ thuộc bất kỳ file model bên ngoài.
"""

import logging
from typing import Optional

import numpy as np

from .analyzer import analyze_image
from .schemas import EvaluationResult, TechnicalMetrics

logger = logging.getLogger("img_doctor.analyzer_evaluator")

# ---------------------------------------------------------------------------
# Tier 1: pyiqa (BRISQUE / NIQE) — import mềm, không crash nếu thiếu
# ---------------------------------------------------------------------------
try:
    import pyiqa  # type: ignore[import-untyped]

    _brisque_metric = pyiqa.create_metric("brisque", device="cpu")
    _niqe_metric = pyiqa.create_metric("niqe", device="cpu")
    PYIQA_AVAILABLE = True
    logger.debug("pyiqa loaded successfully: BRISQUE + NIQE metrics ready.")
except Exception:  # ImportError hoặc RuntimeError khi không tìm thấy model
    PYIQA_AVAILABLE = False
    _brisque_metric = None
    _niqe_metric = None
    logger.warning(
        "pyiqa not available (or model download failed). Falling back to heuristic NR-IQA (Tier 2)."
    )

# ---------------------------------------------------------------------------
# Hằng số Heuristic NR-IQA (Tier 2)
# ---------------------------------------------------------------------------
# Trọng số phạt trong công thức Composite Score
_W_NOISE = 0.35
_W_BLUR = 0.30
_W_EXPOSURE = 0.20
_W_CONTRAST = 0.15

# Ngưỡng tham chiếu
_IDEAL_BRIGHTNESS = 128.0  # Mức xám cân bằng lý tưởng
_IDEAL_CONTRAST_MIN = 45.0  # Dải tương phản chuẩn tối thiểu
_IDEAL_CONTRAST_MAX = 75.0  # Dải tương phản chuẩn tối đa
_NOISE_MAX_PENALTY = 30.0  # Mức nhiễu tối đa tương đương 100% phạt nhiễu
_BLUR_SHARP_THRESHOLD = 300.0  # Laplacian var coi là nét tốt
_EXPOSURE_MAX_DEVIATION = 128.0  # Độ lệch tối đa khỏi mức cân bằng

# Ngưỡng quyết định quality_improved
_DELTA_NOISE_DENOISE_THRESHOLD = -1.5  # Cải thiện rõ khi nhiễu giảm > 1.5
_DELTA_SHARPNESS_SHARPEN_THRESHOLD = 20.0  # Cải thiện khi laplacian var tăng > 20
_DELTA_NOISE_SHARPEN_MAX = 3.0  # Khi tăng nét, nhiễu không được tăng quá 3
_DELTA_NOISE_DEGRADE_THRESHOLD = 10.0  # Thoái hóa rõ khi nhiễu tăng > 10
_SHARPNESS_DROP_MAX_RATIO = 0.30  # Khi khử nhiễu, không được mất > 30% nét


# ---------------------------------------------------------------------------
# Hàm nội bộ — Tier 2: Heuristic Natural Scene Composite Score
# ---------------------------------------------------------------------------


def _compute_heuristic_score(metrics: TechnicalMetrics) -> float:
    """
    Tính toán chỉ số chất lượng cảm nhận tổng hợp (Heuristic Composite Score).

    Công thức:
        Score = 100 - (w_noise * N_hat + w_blur * B_hat + w_exp * E_hat + w_contrast * C_hat)

    Mỗi thành phần phạt được chuẩn hóa về [0, 100]:
    - N_hat: Phạt nhiễu — noise_variance / NOISE_MAX_PENALTY * 100
    - B_hat: Phạt mờ  — (1 - min(laplacian_var, BLUR_SHARP_THRESHOLD) / BLUR_SHARP_THRESHOLD) * 100
    - E_hat: Phạt lệch sáng — |brightness_mean - 128| / 128 * 100
    - C_hat: Phạt tương phản kém — max(0, khoảng cách đến dải [45,75]) / 45 * 100

    Args:
        metrics: TechnicalMetrics của ảnh hiện tại.

    Returns:
        float: Điểm chất lượng cảm nhận trong [0, 100]. Càng cao càng tốt.
    """
    # Phạt nhiễu: chuẩn hóa về [0, 100]
    noise_penalty = min(metrics.noise_variance / _NOISE_MAX_PENALTY, 1.0) * 100.0

    # Phạt mờ: 0 khi nét hoàn hảo, 100 khi Laplacian var = 0
    sharpness_ratio = min(metrics.sharpness_laplacian_var / _BLUR_SHARP_THRESHOLD, 1.0)
    blur_penalty = (1.0 - sharpness_ratio) * 100.0

    # Phạt lệch sáng: khoảng cách từ mức cân bằng 128
    exposure_deviation = abs(metrics.brightness_mean - _IDEAL_BRIGHTNESS)
    exposure_penalty = min(exposure_deviation / _EXPOSURE_MAX_DEVIATION, 1.0) * 100.0

    # Phạt tương phản kém: 0 nếu trong dải [45, 75], tăng dần ra ngoài
    contrast = metrics.contrast_std
    if _IDEAL_CONTRAST_MIN <= contrast <= _IDEAL_CONTRAST_MAX:
        contrast_distance = 0.0
    elif contrast < _IDEAL_CONTRAST_MIN:
        contrast_distance = _IDEAL_CONTRAST_MIN - contrast
    else:
        contrast_distance = contrast - _IDEAL_CONTRAST_MAX
    contrast_penalty = min(contrast_distance / _IDEAL_CONTRAST_MIN, 1.0) * 100.0

    # Tổng hợp có trọng số
    total_penalty = (
        _W_NOISE * noise_penalty
        + _W_BLUR * blur_penalty
        + _W_EXPOSURE * exposure_penalty
        + _W_CONTRAST * contrast_penalty
    )

    score = max(0.0, 100.0 - total_penalty)

    logger.debug(
        "Heuristic NR-IQA score=%.2f | penalties: noise=%.2f blur=%.2f exposure=%.2f contrast=%.2f",
        score,
        noise_penalty,
        blur_penalty,
        exposure_penalty,
        contrast_penalty,
    )
    return round(score, 2)


# ---------------------------------------------------------------------------
# Hàm nội bộ — Tier 1: pyiqa wrapper
# ---------------------------------------------------------------------------


def _compute_pyiqa_scores(rgb_image: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """
    Tính điểm BRISQUE và NIQE qua pyiqa nếu thư viện sẵn sàng.

    Chuyển đổi ảnh np.ndarray uint8 RGB sang tensor PyTorch [1, 3, H, W] float32 [0, 1].

    Args:
        rgb_image: Ảnh RGB uint8 (H, W, 3).

    Returns:
        Tuple (brisque_score, niqe_score): Cả hai là Optional[float].
        Trả về (None, None) nếu pyiqa không sẵn sàng hoặc gặp lỗi.
    """
    if not PYIQA_AVAILABLE:
        return None, None

    try:
        import torch  # type: ignore[import-untyped]

        # Chuyển numpy uint8 RGB → tensor float32 [0, 1] shape [1, 3, H, W]
        tensor = (
            torch.from_numpy(rgb_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        )

        with torch.no_grad():
            brisque = float(_brisque_metric(tensor).item())
            niqe = float(_niqe_metric(tensor).item())

        logger.debug("pyiqa scores: BRISQUE=%.4f NIQE=%.4f", brisque, niqe)
        return round(brisque, 4), round(niqe, 4)

    except Exception as exc:
        logger.warning("pyiqa scoring failed (%s). Returning None scores.", exc)
        return None, None


# ---------------------------------------------------------------------------
# Hàm nội bộ — Delta Metrics Engine
# ---------------------------------------------------------------------------


def _compute_delta_metrics(curr: TechnicalMetrics, prev: TechnicalMetrics) -> dict[str, float]:
    """
    Tính toán biến thiên các chỉ số kỹ thuật giữa vòng lặp hiện tại và vòng trước.

    Công thức:
        Δ_contrast   = σ_curr - σ_prev
        Δ_sharpness  = Var(∇²_curr) - Var(∇²_prev)
        Δ_noise      = Var(Noise_curr) - Var(Noise_prev)
        Δ_brightness = |μ_prev - 128| - |μ_curr - 128|   (dương = tiến gần cân bằng)

    Args:
        curr: TechnicalMetrics của ảnh vòng hiện tại.
        prev: TechnicalMetrics của ảnh vòng trước.

    Returns:
        dict[str, float]: Từ điển chứa 4 giá trị delta đã làm tròn 4 chữ số.
    """
    delta_contrast = curr.contrast_std - prev.contrast_std
    delta_sharpness = curr.sharpness_laplacian_var - prev.sharpness_laplacian_var
    delta_noise = curr.noise_variance - prev.noise_variance
    delta_brightness = abs(prev.brightness_mean - _IDEAL_BRIGHTNESS) - abs(
        curr.brightness_mean - _IDEAL_BRIGHTNESS
    )

    logger.debug(
        "Delta metrics: Δcontrast=%.4f Δsharpness=%.4f Δnoise=%.4f Δbrightness=%.4f",
        delta_contrast,
        delta_sharpness,
        delta_noise,
        delta_brightness,
    )

    return {
        "delta_contrast": round(delta_contrast, 4),
        "delta_sharpness": round(delta_sharpness, 4),
        "delta_noise": round(delta_noise, 4),
        "delta_brightness": round(delta_brightness, 4),
    }


def _determine_quality_improved(
    curr: TechnicalMetrics,
    prev: TechnicalMetrics,
    delta: dict[str, float],
) -> bool:
    """
    Xác định liệu chất lượng ảnh có thực sự cải thiện so với vòng lặp trước.

    Logic phân tích đa mục tiêu theo kế hoạch Stage 3.3:
    1. Phát hiện thoái hóa nghiêm trọng (ưu tiên cao nhất):
       - Nhiễu tăng đột biến > 10.0 → Thoái hóa.
       - Xuất hiện highlight clipping mới (cháy sáng) → Thoái hóa.
    2. Mục tiêu khử nhiễu đạt khi:
       - Δ_noise < -1.5 VÀ độ sụt sharpness < 30%.
    3. Mục tiêu tăng nét đạt khi:
       - Δ_sharpness > 20 VÀ Δ_noise < 3.0.
    4. Mục tiêu tăng sáng đạt khi:
       - Ảnh tiến gần mức cân bằng (Δ_brightness > 0).
    5. Nếu ít nhất 1 mục tiêu đạt và không có thoái hóa → Cải thiện.

    Args:
        curr: TechnicalMetrics vòng hiện tại.
        prev: TechnicalMetrics vòng trước.
        delta: Từ điển delta metrics đã tính.

    Returns:
        bool: True nếu chất lượng được đánh giá là đã cải thiện.
    """
    delta_noise = delta["delta_noise"]
    delta_sharpness = delta["delta_sharpness"]
    delta_brightness = delta["delta_brightness"]

    # --- Phát hiện thoái hóa nghiêm trọng ---
    if delta_noise > _DELTA_NOISE_DEGRADE_THRESHOLD:
        logger.debug(
            "quality_improved=False: severe noise degradation (Δnoise=%.4f > %.1f)",
            delta_noise,
            _DELTA_NOISE_DEGRADE_THRESHOLD,
        )
        return False

    # Cháy sáng xuất hiện mới (highlight clipping ratio tăng đáng kể)
    curr_clip = curr.histogram_stats.get("highlight_clip_ratio", 0.0)
    prev_clip = prev.histogram_stats.get("highlight_clip_ratio", 0.0)
    if curr_clip > 0.15 and curr_clip > prev_clip * 1.5:
        logger.debug(
            "quality_improved=False: new highlight clipping appeared (curr=%.3f prev=%.3f)",
            curr_clip,
            prev_clip,
        )
        return False

    # --- Kiểm tra từng mục tiêu cải thiện ---
    # Mục tiêu 1: Khử nhiễu thành công
    sharpness_drop_ratio = (prev.sharpness_laplacian_var - curr.sharpness_laplacian_var) / max(
        prev.sharpness_laplacian_var, 1.0
    )
    denoise_improved = (
        delta_noise < _DELTA_NOISE_DENOISE_THRESHOLD
        and sharpness_drop_ratio < _SHARPNESS_DROP_MAX_RATIO
    )

    # Mục tiêu 2: Tăng nét thành công
    sharpen_improved = (
        delta_sharpness > _DELTA_SHARPNESS_SHARPEN_THRESHOLD
        and delta_noise < _DELTA_NOISE_SHARPEN_MAX
    )

    # Mục tiêu 3: Cải thiện độ sáng (tiến về cân bằng)
    brightness_improved = delta_brightness > 0.0

    quality_improved = denoise_improved or sharpen_improved or brightness_improved

    logger.debug(
        "quality_improved=%s (denoise=%s sharpen=%s brightness=%s)",
        quality_improved,
        denoise_improved,
        sharpen_improved,
        brightness_improved,
    )
    return quality_improved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_no_reference(
    current_image: np.ndarray,
    previous_image: Optional[np.ndarray] = None,
    iteration: int = 1,
) -> EvaluationResult:
    """
    Đánh giá chất lượng ảnh không cần ảnh gốc (No-Reference IQA).

    Tuân thủ ADR-002: Không dùng PSNR/SSIM. Chỉ áp dụng cho ảnh thực tế
    tải lên từ người dùng (data/real/).

    Quy trình:
    1. Trích xuất TechnicalMetrics của ảnh hiện tại qua analyze_image().
    2. [Tier 1] Tính BRISQUE và NIQE qua pyiqa nếu sẵn sàng.
    3. [Tier 2] Tính Heuristic Natural Scene Composite Score (luôn hoạt động).
    4. Tính Delta Metrics nếu có previous_image (ảnh vòng trước).
    5. Xác định quality_improved dựa trên quy tắc đa mục tiêu.

    Args:
        current_image: Ảnh ở vòng lặp hiện tại (RGB, uint8).
        previous_image: Ảnh vòng lặp trước (hoặc ảnh đầu vào ban đầu).
            None nếu là vòng lặp đầu tiên.
        iteration: Số thứ tự vòng lặp hiện tại (1..3).

    Returns:
        EvaluationResult: Kết quả đánh giá đầy đủ gồm brisque_score, niqe_score,
            heuristic_score, TechnicalMetrics, delta_metrics và quality_improved.

    Raises:
        ValueError: Khi current_image là None hoặc rỗng.
    """
    if current_image is None or current_image.size == 0:
        raise ValueError("current_image is None or empty.")

    logger.debug(
        "evaluate_no_reference called: shape=%s iteration=%d has_prev=%s",
        current_image.shape,
        iteration,
        previous_image is not None,
    )

    # -----------------------------------------------------------------------
    # Bước 1: Trích xuất TechnicalMetrics
    # -----------------------------------------------------------------------
    curr_metrics = analyze_image(current_image)

    # -----------------------------------------------------------------------
    # Bước 2: Tier 1 — pyiqa BRISQUE + NIQE
    # -----------------------------------------------------------------------
    brisque_score, niqe_score = _compute_pyiqa_scores(current_image)

    # -----------------------------------------------------------------------
    # Bước 3: Tier 2 — Heuristic Composite Score (luôn chạy)
    # -----------------------------------------------------------------------
    heuristic_score = _compute_heuristic_score(curr_metrics)

    # -----------------------------------------------------------------------
    # Bước 4 & 5: Delta Metrics + quality_improved
    # -----------------------------------------------------------------------
    delta_metrics: dict[str, float] = {}
    quality_improved = True  # Mặc định True cho vòng đầu tiên (chưa có baseline)

    if previous_image is not None:
        prev_metrics = analyze_image(previous_image)
        delta_metrics = _compute_delta_metrics(curr_metrics, prev_metrics)
        quality_improved = _determine_quality_improved(curr_metrics, prev_metrics, delta_metrics)

    logger.info(
        "NR-IQA evaluation (iter=%d): heuristic=%.2f brisque=%s niqe=%s quality_improved=%s",
        iteration,
        heuristic_score,
        f"{brisque_score:.4f}" if brisque_score is not None else "N/A",
        f"{niqe_score:.4f}" if niqe_score is not None else "N/A",
        quality_improved,
    )

    return EvaluationResult(
        iteration=iteration,
        is_reference_eval=False,
        brisque_score=brisque_score,
        niqe_score=niqe_score,
        technical_metrics=curr_metrics,
        delta_metrics=delta_metrics,
        quality_improved=quality_improved,
        # Lưu heuristic score vào vlm_feedback để Module 4 có thể đọc
        vlm_feedback=f"Heuristic perceptual score: {heuristic_score:.2f}/100",
    )
