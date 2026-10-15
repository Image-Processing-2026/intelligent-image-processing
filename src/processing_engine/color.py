"""
Hiệu chỉnh màu sắc và cân bằng trắng (Color Correction & White Balance).

Bao gồm:
    - apply_color_balance: chỉnh độ bão hòa (HSV) và tông ấm/lạnh bằng gain kênh R/B.
    - apply_white_balance: cân bằng trắng tự động Gray World / White Patch.
"""

import logging
from typing import Literal, Optional

import cv2
import numpy as np

from .base import ProcessingInputError, apply_region_op, require_choice, require_number

logger = logging.getLogger(__name__)

# Hệ số gain tông ấm/lạnh trên mỗi đơn vị temperature_shift: gain_R = 1 + k·t, gain_B = 1 − k·t.
# k = 20/128 để điểm ảnh xám trung bình (128) dịch đúng 20 mức như cách cộng ±20 trước đây,
# nhờ vậy các giá trị temperature_shift đã hiệu chỉnh trong playbook của Agent vẫn giữ ý nghĩa.
TEMPERATURE_GAIN_PER_UNIT = 20.0 / 128.0

WHITE_BALANCE_METHODS = ("gray_world", "white_patch")
# Điểm ảnh có kênh >= ngưỡng này coi là đã cháy sáng (clipped): không phản ánh màu nguồn sáng
_CLIPPED_LEVEL = 250
# Giới hạn gain mỗi kênh để tránh khuếch đại cực đoan khi một kênh gần như bằng 0
_MAX_CHANNEL_GAIN = 4.0


def _raw_color_balance(
    image: np.ndarray, saturation_scale: float = 1.0, temperature_shift: float = 0.0
) -> np.ndarray:
    """Hiệu chỉnh độ bão hòa và tông ấm/lạnh."""
    # Ảnh grayscale không có thông tin màu -> saturation/temperature vô nghĩa, trả về nguyên trạng
    if image.ndim == 2:
        return image

    # 1. Điều chỉnh độ bão hòa màu trong không gian HSV
    if abs(saturation_scale - 1.0) > 1e-3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 2. Điều chỉnh tông ấm/lạnh bằng gain kênh (mô hình nhân hệ số kênh kiểu von Kries).
    # Đây KHÔNG phải nhiệt độ màu Kelvin: chỉ dịch trục đỏ–lam, kênh G giữ nguyên.
    # Nhân hệ số giữ điểm đen ở 0 (cách cộng hằng số cũ làm vùng tối bị ám đỏ/lam).
    if abs(temperature_shift) > 1e-3:
        k = TEMPERATURE_GAIN_PER_UNIT * temperature_shift
        gains = np.array([1.0 + k, 1.0, 1.0 - k], dtype=np.float32)
        image = np.clip(np.rint(image.astype(np.float32) * gains), 0, 255).astype(np.uint8)

    return image


def apply_color_balance(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    saturation_scale: float = 1.0,
    temperature_shift: float = 0.0,
) -> np.ndarray:
    """
    Cân chỉnh màu sắc cho vùng ảnh xác định theo mặt nạ mềm.

    Args:
        saturation_scale: Hệ số nhân kênh S trong HSV (1.0 = giữ nguyên, > 1 rực hơn).
        temperature_shift: Tông ấm/lạnh không thứ nguyên; gain_R = 1 + k·t, gain_B = 1 − k·t
            với k = 20/128 ≈ 0.156 (t > 0 ấm hơn, t < 0 lạnh hơn).

    Raises:
        ProcessingInputError: saturation_scale âm, gain kênh R/B không dương (|t| ≥ 1/k),
            tham số không phải số hữu hạn, hoặc ảnh/mặt nạ vi phạm hợp đồng.
    """
    saturation_scale = require_number(
        saturation_scale, "saturation_scale", "apply_color_balance", ge=0.0
    )
    temperature_shift = require_number(
        temperature_shift, "temperature_shift", "apply_color_balance"
    )
    if TEMPERATURE_GAIN_PER_UNIT * abs(temperature_shift) >= 1.0:
        raise ProcessingInputError(
            "apply_color_balance: |temperature_shift| must be < "
            f"{1.0 / TEMPERATURE_GAIN_PER_UNIT:.2f} so channel gains stay positive, "
            f"got {temperature_shift}"
        )
    # Gain kênh là phép toán từng điểm ảnh → cắt theo mask không cần halo. Khi đổi saturation thì
    # không cắt: HSV2RGB của OpenCV làm tròn khác nhau theo vị trí điểm ảnh (lệch 1 mức xám).
    crop_halo = 0 if abs(saturation_scale - 1.0) <= 1e-3 else None
    return apply_region_op(
        image,
        mask,
        _raw_color_balance,
        crop_halo=crop_halo,
        saturation_scale=saturation_scale,
        temperature_shift=temperature_shift,
    )


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Phân vị có trọng số: giá trị nhỏ nhất mà tổng trọng số tích lũy đạt q% tổng trọng số."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    idx = int(np.searchsorted(cumulative, cumulative[-1] * q / 100.0, side="left"))
    return float(values[order][min(idx, len(values) - 1)])


def _raw_white_balance(
    image: np.ndarray,
    method: Literal["gray_world", "white_patch"] = "gray_world",
    percentile: float = 99.0,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Cân bằng trắng tự động bằng gain từng kênh: I'_c = g_c · I_c.

    Thống kê màu nguồn sáng được lấy có trọng số theo mặt nạ (weights), để vùng chọn
    (VD: khuôn mặt dưới đèn vàng) được cân bằng theo chính ánh sáng của nó.
    """
    # Ảnh xám không có thông tin màu -> cân bằng trắng vô nghĩa, trả về nguyên trạng
    if image.ndim == 2:
        return image

    pixels = image.reshape(-1, 3).astype(np.float64)
    w = (
        np.ones(len(pixels), dtype=np.float64)
        if weights is None
        else weights.reshape(-1).astype(np.float64)
    )

    # Loại điểm ảnh cháy sáng: kênh bị cắt ở 255 không còn phản ánh màu của nguồn sáng
    unclipped = pixels.max(axis=1) < _CLIPPED_LEVEL
    if np.sum(w[unclipped]) > 0:
        pixels, w = pixels[unclipped], w[unclipped]
    if np.sum(w) <= 0:
        logger.warning("White balance skipped: no pixels carry weight in the selected region")
        return image

    if method == "gray_world":
        # Giả thuyết Gray World: trung bình phản xạ của cảnh là xám trung tính,
        # nên trung bình 3 kênh phải bằng nhau. Đích = trung bình của 3 kênh → giữ độ sáng chung.
        stats = (pixels * w[:, None]).sum(axis=0) / w.sum()
        target = stats.mean()
    else:  # method == "white_patch" (đã được kiểm tra ở apply_white_balance)
        # Giả thuyết White Patch (Max-RGB / Retinex): vùng sáng nhất là vật trắng, nên các kênh
        # tại đó phải bằng nhau. Dùng phân vị (mặc định 99) thay cho max để bền với nhiễu/điểm nóng.
        # Đích = kênh sáng nhất → chỉ sửa màu, không tăng phơi sáng (việc đó thuộc gamma/CLAHE).
        stats = np.array([_weighted_percentile(pixels[:, c], w, percentile) for c in range(3)])
        target = stats.max()

    if target <= 0:
        logger.warning("White balance skipped: region is black, no illuminant estimate")
        return image

    gains = target / np.maximum(stats, 1e-6)
    clipped_gains = np.clip(gains, 1.0 / _MAX_CHANNEL_GAIN, _MAX_CHANNEL_GAIN)
    if not np.allclose(gains, clipped_gains):
        logger.warning(
            "White balance gains %s clamped to %s (limit x%.1f)",
            np.round(gains, 3).tolist(),
            np.round(clipped_gains, 3).tolist(),
            _MAX_CHANNEL_GAIN,
        )
    logger.debug("White balance (%s) channel gains: %s", method, np.round(clipped_gains, 4))

    balanced = image.astype(np.float32) * clipped_gains.astype(np.float32)
    return np.clip(np.rint(balanced), 0, 255).astype(np.uint8)


def apply_white_balance(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: Literal["gray_world", "white_patch"] = "gray_world",
    percentile: float = 99.0,
) -> np.ndarray:
    """
    Cân bằng trắng tự động theo vùng (khử ám màu do nguồn sáng).

    Args:
        image: Ảnh RGB uint8 (ảnh xám được trả về nguyên trạng).
        mask: Mặt nạ mềm float32 [0, 1]. Ngoài vai trò hòa trộn, mask còn là TRỌNG SỐ khi
            ước lượng màu nguồn sáng — chỉ vùng được chọn quyết định gain.
        method:
            - "gray_world": g_c = mean(μ_R, μ_G, μ_B) / μ_c (μ_c: trung bình kênh c).
            - "white_patch": g_c = max(p_R, p_G, p_B) / p_c (p_c: phân vị `percentile` của kênh c).
        percentile: Phân vị dùng cho white_patch, trong (50, 100]. Bỏ qua với gray_world.

    Returns:
        np.ndarray: Ảnh uint8 cùng shape. Gain mỗi kênh bị giới hạn trong [1/4, 4];
        điểm ảnh cháy sáng (kênh >= 250) không tham gia ước lượng.

    Raises:
        ProcessingInputError: method/percentile không hợp lệ hoặc ảnh/mặt nạ vi phạm hợp đồng.
    """
    require_choice(method, WHITE_BALANCE_METHODS, "method", "apply_white_balance")
    percentile = require_number(percentile, "percentile", "apply_white_balance", gt=50.0, le=100.0)
    return apply_region_op(
        image,
        mask,
        _raw_white_balance,
        method=method,
        percentile=percentile,
        weights=mask,
    )
