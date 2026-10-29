"""Tiện ích dùng chung cho các module đo: đóng gói kết quả, chuyển đổi màu, kẹp giá trị."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .schemas import Estimate


@dataclass
class Section:
    """Kết quả của một module đo: ước lượng (vào report), bản đồ tile và cảnh báo."""

    estimates: Dict[str, Estimate] = field(default_factory=dict)
    maps: Dict[str, np.ndarray] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def clamp01(value: float) -> float:
    """Kẹp về [0, 1]; NaN coi như 0."""
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def make_estimate(value: float, confidence: float, n_samples: int = 0) -> Optional[Estimate]:
    """Tạo Estimate. Trả None khi value không hữu hạn: khi đó không có key trong report."""
    if not math.isfinite(value):
        return None
    return Estimate(value=float(value), confidence=clamp01(confidence), n_samples=int(n_samples))


def put_estimate(
    section: Section, key: str, value: float, confidence: float, n_samples: int = 0
) -> None:
    """Ghi ước lượng vào section nếu hữu hạn."""
    estimate = make_estimate(value, confidence, n_samples)
    if estimate is not None:
        section.estimates[key] = estimate


def srgb_to_linear(rgb01: np.ndarray) -> np.ndarray:
    """Giải mã gamma sRGB (đầu vào [0, 1]) sang tuyến tính (IEC 61966-2-1)."""
    linear = np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)
