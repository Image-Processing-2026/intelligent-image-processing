"""Đo nén: blockiness theo lưới 8×8 (kiểu Wang, Bovik, Evans 2002) theo hai chiều.

Với mỗi chiều: tỉ lệ trung bình |sai phân| tại biên khối 8×8 so với bên trong khối. Ảnh không nén
có tỉ lệ ≈ 1. JPEG chất lượng thấp có tỉ lệ rõ rệt > 1 do biên khối bị lệch. Hai chiều được lấy
trung bình hình học.
"""

import math

import cv2
import numpy as np

from ._util import Section, put_estimate

BLOCK = 8
EPSILON = 1e-3
FULL_CONFIDENCE_SIDE = 64


def _boundary_ratio(diff: np.ndarray, axis: int) -> float:
    """Tỉ lệ |sai phân| trung bình tại biên khối so với bên trong, dọc theo `axis` của diff.

    diff[:, x] (axis=1) là |I[:, x+1] − I[:, x]|: biên khối nằm sau cột x khi (x+1) chia hết cho 8.
    """
    length = diff.shape[axis]
    boundary = ((np.arange(length) + 1) % BLOCK) == 0
    if not boundary.any() or boundary.all():
        return math.nan
    if axis == 1:
        at_boundary = diff[:, boundary]
        inside = diff[:, ~boundary]
    else:
        at_boundary = diff[boundary]
        inside = diff[~boundary]
    return (float(at_boundary.mean()) + EPSILON) / (float(inside.mean()) + EPSILON)


def measure_compression(rgb: np.ndarray) -> Section:
    """Đo blockiness trên luma. Ảnh quá nhỏ (không đủ hai biên khối) thì bỏ qua và ghi chú."""
    section = Section()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    height, width = gray.shape
    ratios = [
        _boundary_ratio(np.abs(np.diff(gray, axis=1)), axis=1),
        _boundary_ratio(np.abs(np.diff(gray, axis=0)), axis=0),
    ]
    ratios = [r for r in ratios if math.isfinite(r)]
    if not ratios:
        section.notes.append("image too small for blockiness measurement")
        return section
    blockiness = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    confidence = min(1.0, min(height, width) / FULL_CONFIDENCE_SIDE)
    put_estimate(section, "blockiness", blockiness, confidence, height * width)
    return section
