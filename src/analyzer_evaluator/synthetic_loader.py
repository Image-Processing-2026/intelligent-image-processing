"""
Synthetic Dataset Loader cho Module 1 — Image Analyzer & Evaluator.

Cung cấp hàm tải cặp ảnh (clean, degraded) từ thư mục benchmark nhân tạo
và runner chạy toàn bộ bộ đánh giá Full-Reference trên tập synthetic.

Quy ước đặt tên (Naming Convention):
    data/synthetic/clean/photo.png
    data/synthetic/degraded/photo_gaussian_noise.png
    data/synthetic/degraded/photo_motion_blur.png

Tất cả ảnh suy giảm phải có stem bắt đầu bằng stem của ảnh sạch tương ứng
(ví dụ: "photo" → "photo_gaussian_noise", "photo_underexposed").
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .reference_eval import evaluate_reference
from .schemas import EvaluationResult

logger = logging.getLogger("img_doctor.analyzer_evaluator")

# Định dạng ảnh được hỗ trợ
_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def load_synthetic_pairs(
    clean_dir: str = "data/synthetic/clean",
    degraded_dir: str = "data/synthetic/degraded",
) -> list[tuple[Path, Path]]:
    """
    Tải danh sách các cặp ảnh (ảnh sạch, ảnh suy giảm) từ thư mục benchmark.

    Quy ước: Mỗi ảnh suy giảm có tên bắt đầu bằng stem của ảnh sạch tương ứng
    (phân tách bằng dấu gạch dưới). Ví dụ:
        clean/photo.png → degraded/photo_gaussian_noise.png
        clean/portrait.jpg → degraded/portrait_motion_blur.jpg

    Args:
        clean_dir: Đường dẫn tới thư mục chứa ảnh sạch ground-truth.
        degraded_dir: Đường dẫn tới thư mục chứa ảnh suy giảm cần đánh giá.

    Returns:
        list[tuple[Path, Path]]: Danh sách các cặp (clean_path, degraded_path).
            Trả về list rỗng nếu thư mục không tồn tại hoặc không tìm thấy cặp.
    """
    clean_path = Path(clean_dir)
    degraded_path = Path(degraded_dir)

    if not clean_path.exists():
        logger.warning("Clean directory does not exist: %s", clean_path.resolve())
        return []

    if not degraded_path.exists():
        logger.warning("Degraded directory does not exist: %s", degraded_path.resolve())
        return []

    # Lập chỉ mục ảnh sạch
    clean_images = sorted(
        p for p in clean_path.iterdir() if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not clean_images:
        logger.warning("No supported images found in clean directory: %s", clean_path)
        return []

    pairs: list[tuple[Path, Path]] = []

    for clean_img in clean_images:
        # Tìm tất cả ảnh suy giảm có stem bắt đầu bằng stem của ảnh sạch
        matching_degraded = sorted(
            p
            for p in degraded_path.iterdir()
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS and p.stem.startswith(clean_img.stem + "_")
        )

        if not matching_degraded:
            logger.debug("No degraded variants found for clean image: %s", clean_img.name)
            continue

        for degraded_img in matching_degraded:
            pairs.append((clean_img, degraded_img))
            logger.debug("Paired: %s → %s", clean_img.name, degraded_img.name)

    logger.info(
        "Loaded %d synthetic pairs from '%s' (clean) and '%s' (degraded).",
        len(pairs),
        clean_dir,
        degraded_dir,
    )
    return pairs


def _load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """
    Đọc ảnh từ đĩa và chuyển về định dạng RGB uint8.

    Args:
        path: Đường dẫn tới file ảnh.

    Returns:
        np.ndarray RGB uint8 hoặc None nếu đọc thất bại.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        logger.error("Failed to read image: %s", path)
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run_benchmark_suite(
    clean_dir: str = "data/synthetic/clean",
    degraded_dir: str = "data/synthetic/degraded",
) -> dict[str, dict[str, float]]:
    """
    Chạy toàn bộ đánh giá Full-Reference trên tập synthetic benchmark.

    Với mỗi cặp ảnh (clean, degraded), gọi evaluate_reference() và thu thập
    kết quả PSNR, SSIM, MSE vào bảng báo cáo tổng hợp.

    Args:
        clean_dir: Đường dẫn thư mục ảnh sạch.
        degraded_dir: Đường dẫn thư mục ảnh suy giảm.

    Returns:
        dict[str, dict[str, float]]: Khoá là tên file ảnh suy giảm,
        giá trị là dict chứa {"psnr": ..., "ssim": ..., "mse": ...}.
        Trả về dict rỗng nếu không tìm thấy cặp ảnh nào.
    """
    pairs = load_synthetic_pairs(clean_dir, degraded_dir)

    if not pairs:
        logger.warning("No synthetic pairs found. Benchmark suite returns empty result.")
        return {}

    results: dict[str, dict[str, float]] = {}

    for clean_path, degraded_path in pairs:
        clean_img = _load_image_rgb(clean_path)
        degraded_img = _load_image_rgb(degraded_path)

        if clean_img is None or degraded_img is None:
            logger.warning(
                "Skipping pair (%s, %s) due to read failure.",
                clean_path.name,
                degraded_path.name,
            )
            continue

        try:
            eval_result: EvaluationResult = evaluate_reference(
                current_image=degraded_img,
                ground_truth_image=clean_img,
            )

            results[degraded_path.name] = {
                "psnr": eval_result.psnr if eval_result.psnr is not None else float("nan"),
                "ssim": eval_result.ssim if eval_result.ssim is not None else float("nan"),
                "mse": eval_result.mse if eval_result.mse is not None else float("nan"),
            }

            logger.info(
                "Benchmark [%s]: PSNR=%.2f dB  SSIM=%.4f  MSE=%.2f",
                degraded_path.name,
                eval_result.psnr or 0.0,
                eval_result.ssim or 0.0,
                eval_result.mse or 0.0,
            )

        except Exception as exc:
            logger.error(
                "Error evaluating pair (%s, %s): %s",
                clean_path.name,
                degraded_path.name,
                exc,
            )

    logger.info("Benchmark suite complete: %d pairs evaluated.", len(results))
    return results
