"""
Module Image Analyzer & Evaluator.

Cung cấp bộ công cụ phân tích kỹ thuật và đánh giá chất lượng ảnh.

Import Graph (không circular):
    schemas.py          ← (Không import ai, chỉ được import)
        ↑               ↑               ↑
    analyzer.py   reference_eval.py   no_reference_eval.py
        ↑               ↑               ↑
                __init__.py (re-export)
                        ↑
               synthetic_loader.py  ← (import reference_eval)
"""

import logging
from typing import Optional

import numpy as np

from .analyzer import analyze_image
from .no_reference_eval import evaluate_no_reference
from .reference_eval import evaluate_reference
from .schemas import EvaluationResult, TechnicalMetrics
from .synthetic_loader import load_synthetic_pairs, run_benchmark_suite

logger = logging.getLogger("img_doctor.analyzer_evaluator")

__all__ = [
    # Schemas
    "TechnicalMetrics",
    "EvaluationResult",
    # Core analysis
    "analyze_image",
    # Evaluation pipelines
    "evaluate_reference",
    "evaluate_no_reference",
    # Unified entry point (dùng cho Module 4)
    "evaluate_quality",
    # Synthetic benchmark tools
    "load_synthetic_pairs",
    "run_benchmark_suite",
]


def evaluate_quality(
    current_image: np.ndarray,
    original_image: Optional[np.ndarray] = None,
    is_synthetic: bool = False,
    iteration: int = 1,
) -> EvaluationResult:
    """
    Điểm đầu vào thống nhất duy nhất cho Module 4 gọi đánh giá chất lượng ảnh.

    Tự động phân nhánh theo loại ảnh:
    - is_synthetic=True VÀ có original_image → evaluate_reference() (Full-Reference).
    - Tất cả trường hợp còn lại → evaluate_no_reference() (No-Reference, tuân ADR-002).

    Đảm bảo ADR-002: Không bao giờ gọi evaluate_reference() khi is_synthetic=False,
    kể cả khi original_image được truyền vào (ví dụ: người dùng thử truyền ảnh gốc
    nhưng đây là ảnh thực tế).

    Args:
        current_image: Ảnh hiện tại sau khi xử lý (RGB, uint8).
        original_image: Ảnh gốc ground-truth (chỉ dùng khi is_synthetic=True),
            hoặc ảnh đầu vào ban đầu (dùng làm previous_image trong No-Reference).
        is_synthetic: True nếu đây là tập dữ liệu nhân tạo có ground-truth
            (data/synthetic/). False cho ảnh thực tế người dùng tải lên.
        iteration: Số thứ tự vòng lặp xử lý hiện tại (1..3).

    Returns:
        EvaluationResult: Kết quả đánh giá đầy đủ.

    Raises:
        ValueError: Khi current_image là None hoặc rỗng.
    """
    if is_synthetic and original_image is not None:
        logger.debug(
            "evaluate_quality: routing to evaluate_reference() (is_synthetic=True, iteration=%d)",
            iteration,
        )
        return evaluate_reference(
            current_image=current_image,
            ground_truth_image=original_image,
            iteration=iteration,
        )

    if is_synthetic and original_image is None:
        logger.warning(
            "evaluate_quality called with is_synthetic=True but no original_image. "
            "Falling back to no-reference evaluation."
        )

    logger.debug(
        "evaluate_quality: routing to evaluate_no_reference() (is_synthetic=%s, iteration=%d)",
        is_synthetic,
        iteration,
    )
    return evaluate_no_reference(
        current_image=current_image,
        previous_image=original_image,
        iteration=iteration,
    )
