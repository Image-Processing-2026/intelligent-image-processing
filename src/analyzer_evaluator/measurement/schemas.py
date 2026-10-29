"""Schema của lõi đo cục bộ (hợp đồng C1 trong docs/analysis_v2/contracts.md).

`MeasurementReport` và `RegionStats` là Pydantic (JSON được, đi vào DiagnosisReport và API).
`MeasurementResult` là dataclass giữ bản đồ tile trong bộ nhớ, KHÔNG serialize.
"""

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from pydantic import BaseModel, field_validator


class Estimate(BaseModel):
    """Một ước lượng đo được kèm độ tin cậy. Không bao giờ chứa NaN."""

    value: float
    confidence: float
    n_samples: int = 0

    @field_validator("value")
    @classmethod
    def _value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Estimate.value must be finite; omit the key instead")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_in_unit_range(cls, confidence: float) -> float:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Estimate.confidence must be within [0, 1]")
        return confidence


class MeasurementReport(BaseModel):
    """Tóm tắt toàn ảnh, JSON được (đi vào DiagnosisReport/API)."""

    width: int
    height: int
    tile_size: int
    exposure: Dict[str, Estimate] = {}
    noise: Dict[str, Estimate] = {}
    sharpness: Dict[str, Estimate] = {}
    color: Dict[str, Estimate] = {}
    compression: Dict[str, Estimate] = {}
    haze: Dict[str, Estimate] = {}
    notes: List[str] = []  # cảnh báo đo, ví dụ "few flat patches"


class RegionStats(BaseModel):
    """Thống kê của một vùng (mask mềm), cùng key với MAP_KEYS."""

    region: str
    area_fraction: float  # trung bình mask
    values: Dict[str, Estimate] = {}
    mask_source: str = "unknown"
    mask_confidence: float = 1.0


@dataclass
class MeasurementResult:
    """Kết quả đo đầy đủ. Không serialize ra API/HistoryItem."""

    report: MeasurementReport
    maps: Dict[str, np.ndarray]  # lưới tile float32, (ceil(H/T), ceil(W/T)); NaN = không đo được
    tile_size: int
