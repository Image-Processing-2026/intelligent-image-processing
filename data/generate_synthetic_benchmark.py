"""
Script sinh tập dữ liệu nhân tạo có suy giảm kiểm soát chính xác
(Synthetic Degraded Benchmark Dataset Generator).

Đọc ảnh sạch từ data/synthetic/clean/ và sinh ra các biến thể suy giảm
lưu vào data/synthetic/degraded/ theo quy ước đặt tên chuẩn của
synthetic_loader.load_synthetic_pairs():

    clean/photo.png → degraded/photo_underexposed.png
    clean/photo.png → degraded/photo_overexposed.png
    clean/photo.png → degraded/photo_gaussian_noise.png
    clean/photo.png → degraded/photo_salt_pepper.png
    clean/photo.png → degraded/photo_motion_blur.png
    clean/photo.png → degraded/photo_defocus_blur.png
    clean/photo.png → degraded/photo_low_contrast.png

Cách chạy:
    python data/generate_synthetic_benchmark.py
    python data/generate_synthetic_benchmark.py --clean-dir data/synthetic/clean --degraded-dir data/synthetic/degraded
    python data/generate_synthetic_benchmark.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("img_doctor.benchmark_generator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Degradation parameters (as specified in Section 4.1 of the plan)
# Quy ước gamma theo kế hoạch: I_out = (I_in / 255)^(1/gamma) * 255.
# - gamma < 1 (0.4) → số mũ 1/gamma > 1 → ảnh tối đi (underexposed).
# - gamma > 1 (2.2) → số mũ 1/gamma < 1 → ảnh sáng lên (overexposed).
# ---------------------------------------------------------------------------
_GAMMA_UNDEREXPOSE = 0.4  # Section 4.1: gamma=0.4 làm tối ảnh
_GAMMA_OVEREXPOSE = 2.2  # Section 4.1: gamma=2.2 làm sáng/cháy ảnh
_GAUSSIAN_NOISE_SIGMA = 25  # Section 4.1: sigma=25 for Gaussian noise
_SALT_PEPPER_DENSITY = 0.05  # Section 4.1: 5% impulse noise density
_MOTION_BLUR_KERNEL = 15  # Section 4.1: 15x15 motion blur kernel
_MOTION_BLUR_ANGLE = 45  # Section 4.1: 45-degree motion direction
_DEFOCUS_RADIUS = 5  # Section 4.1: disk radius r=5 for defocus blur
_LOW_CONTRAST_MIN = 80  # Section 4.1: compress dynamic range to [80, 140]
_LOW_CONTRAST_MAX = 140


# ---------------------------------------------------------------------------
# Degradation functions (all return uint8 RGB)
# ---------------------------------------------------------------------------


def apply_underexposure(image: np.ndarray, gamma: float = _GAMMA_UNDEREXPOSE) -> np.ndarray:
    """
    Giả lập ảnh thiếu sáng bằng hiệu chỉnh gamma theo quy ước kế hoạch.

    Công thức: I_out = (I_in / 255)^(1/gamma) * 255, với gamma < 1 (mặc định 0.4)
    cho số mũ 1/gamma = 2.5 > 1 nên ảnh tối đi rõ rệt.

    Args:
        image: Ảnh RGB uint8 đầu vào.
        gamma: Hệ số gamma theo quy ước kế hoạch (gamma < 1 làm tối ảnh).

    Returns:
        Ảnh thiếu sáng uint8 RGB.
    """
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_overexposure(image: np.ndarray, gamma: float = _GAMMA_OVEREXPOSE) -> np.ndarray:
    """
    Giả lập ảnh cháy sáng bằng hiệu chỉnh gamma theo quy ước kế hoạch.

    Công thức: I_out = (I_in / 255)^(1/gamma) * 255, với gamma > 1 (mặc định 2.2)
    cho số mũ 1/gamma ≈ 0.45 < 1 nên ảnh sáng lên, gây clipping vùng highlight.

    Args:
        image: Ảnh RGB uint8 đầu vào.
        gamma: Hệ số gamma theo quy ước kế hoạch (gamma > 1 làm sáng ảnh).

    Returns:
        Ảnh cháy sáng uint8 RGB.
    """
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float = _GAUSSIAN_NOISE_SIGMA,
    seed: int = 42,
) -> np.ndarray:
    """
    Thêm nhiễu Gauss cộng tính (Additive White Gaussian Noise) với độ lệch chuẩn sigma.

    Args:
        image: Ảnh RGB uint8 đầu vào.
        sigma: Độ lệch chuẩn của phân phối nhiễu Gauss.
        seed: Seed ngẫu nhiên đảm bảo tái tạo được.

    Returns:
        Ảnh nhiễu uint8 RGB.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_salt_pepper_noise(
    image: np.ndarray,
    density: float = _SALT_PEPPER_DENSITY,
    seed: int = 42,
) -> np.ndarray:
    """
    Thêm nhiễu muối tiêu (Salt & Pepper / Impulse Noise).

    Mỗi kênh màu có xác suất `density/2` bị đặt thành 255 (muối)
    hoặc 0 (tiêu) một cách độc lập.

    Args:
        image: Ảnh RGB uint8 đầu vào.
        density: Tỷ lệ điểm ảnh bị nhiễu [0, 1]. Mặc định 5%.
        seed: Seed ngẫu nhiên.

    Returns:
        Ảnh nhiễu muối tiêu uint8 RGB.
    """
    rng = np.random.default_rng(seed)
    output = image.copy()
    mask = rng.random(image.shape)
    output[mask < density / 2.0] = 0  # Tiêu (Pepper)
    output[mask > 1.0 - density / 2.0] = 255  # Muối (Salt)
    return output


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = _MOTION_BLUR_KERNEL,
    angle_deg: float = _MOTION_BLUR_ANGLE,
) -> np.ndarray:
    """
    Làm mờ chuyển động tuyến tính (Linear Motion Blur).

    Tạo kernel đường thẳng theo góc angle_deg, sau đó tích chập với ảnh.

    Args:
        image: Ảnh RGB uint8 đầu vào.
        kernel_size: Độ dài của đường chuyển động (lẻ để có tâm chính giữa).
        angle_deg: Góc hướng chuyển động tính theo độ (0° = ngang, 45° = chéo).

    Returns:
        Ảnh mờ chuyển động uint8 RGB.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    # Tạo kernel đường thẳng nằm ngang rồi xoay
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k

    # Xoay kernel để tạo hướng chuyển động
    center = (k // 2, k // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel_rotated = cv2.warpAffine(kernel, rotation_matrix, (k, k), flags=cv2.INTER_LINEAR)
    # Chuẩn hóa để tổng kernel = 1
    kernel_rotated /= kernel_rotated.sum() + 1e-10

    blurred = cv2.filter2D(image, -1, kernel_rotated, borderType=cv2.BORDER_REPLICATE)
    return blurred.astype(np.uint8)


def apply_defocus_blur(
    image: np.ndarray,
    radius: int = _DEFOCUS_RADIUS,
) -> np.ndarray:
    """
    Làm mờ lệch tiêu (Defocus Blur / Disk Blur).

    Tạo kernel hình đĩa tròn (disk aperture) mô phỏng lỗi lấy nét camera.

    Args:
        image: Ảnh RGB uint8 đầu vào.
        radius: Bán kính disk aperture (pixels). Càng lớn = càng mờ.

    Returns:
        Ảnh mờ lệch tiêu uint8 RGB.
    """
    # Tạo kernel hình đĩa tròn
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk_mask = (x**2 + y**2 <= radius**2).astype(np.float32)
    disk_kernel = disk_mask / (disk_mask.sum() + 1e-10)

    blurred = cv2.filter2D(image, -1, disk_kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred.astype(np.uint8)


def apply_low_contrast(
    image: np.ndarray,
    range_min: int = _LOW_CONTRAST_MIN,
    range_max: int = _LOW_CONTRAST_MAX,
) -> np.ndarray:
    """
    Nén dải động (Dynamic Range Compression) về khoảng [range_min, range_max].

    Công thức affine:
        I_out = range_min + (I_in / 255) * (range_max - range_min)

    Args:
        image: Ảnh RGB uint8 đầu vào.
        range_min: Giá trị pixel tối thiểu của ảnh đầu ra.
        range_max: Giá trị pixel tối đa của ảnh đầu ra.

    Returns:
        Ảnh tương phản thấp uint8 RGB.
    """
    normalized = image.astype(np.float32) / 255.0
    compressed = range_min + normalized * (range_max - range_min)
    return np.clip(compressed, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Degradation registry
# ---------------------------------------------------------------------------

DEGRADATION_PIPELINE: list[tuple[str, callable]] = [
    ("underexposed", apply_underexposure),
    ("overexposed", apply_overexposure),
    ("gaussian_noise", apply_gaussian_noise),
    ("salt_pepper", apply_salt_pepper_noise),
    ("motion_blur", apply_motion_blur),
    ("defocus_blur", apply_defocus_blur),
    ("low_contrast", apply_low_contrast),
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_benchmark(
    clean_dir: str = "data/synthetic/clean",
    degraded_dir: str = "data/synthetic/degraded",
    dry_run: bool = False,
) -> dict[str, list[Path]]:
    """
    Sinh tập dữ liệu benchmark nhân tạo từ ảnh sạch.

    Với mỗi ảnh sạch trong clean_dir, tạo 7 biến thể suy giảm lưu vào degraded_dir.
    Tên file tuân theo naming convention của synthetic_loader (stem + "_" + type + ext).

    Args:
        clean_dir: Thư mục chứa ảnh sạch ground-truth.
        degraded_dir: Thư mục đầu ra ảnh suy giảm.
        dry_run: Nếu True, chỉ log mà không ghi file.

    Returns:
        dict[str, list[Path]]: Mapping từ tên ảnh sạch → danh sách path ảnh đã sinh.
    """
    clean_path = Path(clean_dir)
    degraded_path = Path(degraded_dir)

    if not clean_path.exists():
        logger.error("Clean directory not found: %s", clean_path.resolve())
        return {}

    if not dry_run:
        degraded_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", degraded_path.resolve())

    clean_images = sorted(
        p for p in clean_path.iterdir() if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not clean_images:
        logger.warning("No supported images found in: %s", clean_path)
        return {}

    logger.info(
        "Found %d clean images. Generating %d variants each...",
        len(clean_images),
        len(DEGRADATION_PIPELINE),
    )

    results: dict[str, list[Path]] = {}

    for clean_img_path in clean_images:
        # Read image as RGB (OpenCV reads BGR → convert)
        img_bgr = cv2.imread(str(clean_img_path))
        if img_bgr is None:
            logger.error("Failed to read: %s — skipping.", clean_img_path.name)
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        stem = clean_img_path.stem
        # Luôn xuất .png để giữ chất lượng lossless bất kể định dạng đầu vào
        out_ext = ".png"

        generated: list[Path] = []

        for variant_name, fn in DEGRADATION_PIPELINE:
            out_name = f"{stem}_{variant_name}{out_ext}"
            out_path = degraded_path / out_name

            if dry_run:
                logger.info("[DRY-RUN] Would write: %s", out_path)
                generated.append(out_path)
                continue

            try:
                degraded_rgb = fn(img_rgb)
                # Convert back to BGR for OpenCV imwrite
                degraded_bgr = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), degraded_bgr)
                logger.info("Written: %s", out_path.name)
                generated.append(out_path)
            except Exception as exc:
                logger.error("Failed to generate %s: %s", out_name, exc)

        results[clean_img_path.name] = generated
        logger.info(
            "Processed '%s': %d/%d variants generated.",
            clean_img_path.name,
            len(generated),
            len(DEGRADATION_PIPELINE),
        )

    total = sum(len(v) for v in results.values())
    logger.info(
        "Benchmark generation complete: %d images → %d degraded variants.",
        len(results),
        total,
    )
    return results


def create_sample_clean_images(clean_dir: str = "data/synthetic/clean") -> None:
    """
    Tạo các ảnh sạch mẫu nếu thư mục clean_dir trống.

    Hữu ích cho việc demo và test nhanh mà không cần ảnh thực tế.
    Sinh 3 ảnh mẫu đại diện: ảnh phong cảnh gradient, ảnh chân dung,
    ảnh bàn cờ có độ tương phản cao.

    Args:
        clean_dir: Thư mục đầu ra cho ảnh mẫu.
    """
    clean_path = Path(clean_dir)
    clean_path.mkdir(parents=True, exist_ok=True)

    h, w = 256, 256

    # Sample 1: Natural gradient (mô phỏng ảnh phong cảnh)
    sample1 = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient: top=light blue, bottom=darker
    for y in range(h):
        r = int(135 + (y / h) * 80)
        g = int(180 + (y / h) * 40)
        b = int(235 - (y / h) * 60)
        sample1[y, :] = [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]
    # Add some "ground" texture
    sample1[h // 2 :, :, 0] = np.clip(sample1[h // 2 :, :, 0].astype(int) - 60, 0, 255)
    sample1[h // 2 :, :, 2] = np.clip(sample1[h // 2 :, :, 2].astype(int) - 80, 0, 255)
    cv2.imwrite(str(clean_path / "landscape.png"), cv2.cvtColor(sample1, cv2.COLOR_RGB2BGR))

    # Sample 2: Neutral gray patch (mô phỏng ảnh kiểm tra trung tính)
    sample2 = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add some texture via low-amplitude sinusoidal pattern
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    xx, yy = np.meshgrid(x_coords, y_coords)
    texture = (10 * np.sin(xx / 10.0) * np.cos(yy / 10.0)).astype(np.int16)
    for c in range(3):
        sample2[:, :, c] = np.clip(sample2[:, :, c].astype(np.int16) + texture, 0, 255)
    cv2.imwrite(str(clean_path / "neutral_patch.png"), cv2.cvtColor(sample2, cv2.COLOR_RGB2BGR))

    # Sample 3: High-contrast pattern (mô phỏng ảnh tài liệu/bàn cờ)
    sample3 = np.zeros((h, w, 3), dtype=np.uint8)
    block = 32
    for y in range(0, h, block):
        for x in range(0, w, block):
            val = 240 if ((y // block) + (x // block)) % 2 == 0 else 20
            sample3[y : y + block, x : x + block] = val
    cv2.imwrite(str(clean_path / "checkerboard.png"), cv2.cvtColor(sample3, cv2.COLOR_RGB2BGR))

    logger.info(
        "Created 3 sample clean images in '%s': landscape.png, neutral_patch.png, checkerboard.png",
        clean_dir,
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Điểm vào chính cho CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic degraded benchmark images for Module 1 evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clean-dir",
        default="data/synthetic/clean",
        help="Path to directory containing clean ground-truth images (default: data/synthetic/clean).",
    )
    parser.add_argument(
        "--degraded-dir",
        default="data/synthetic/degraded",
        help="Path to output directory for degraded variants (default: data/synthetic/degraded).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing any files.",
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample clean images if clean-dir is empty (useful for demo).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG level logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    clean_path = Path(args.clean_dir)
    if (
        args.create_samples
        or not clean_path.exists()
        or not any(
            p.suffix.lower() in _SUPPORTED_EXTENSIONS
            for p in clean_path.iterdir()
            if clean_path.exists()
        )
    ):
        logger.info("No clean images found in '%s'. Creating sample images...", args.clean_dir)
        create_sample_clean_images(args.clean_dir)

    results = generate_benchmark(
        clean_dir=args.clean_dir,
        degraded_dir=args.degraded_dir,
        dry_run=args.dry_run,
    )

    if not results:
        logger.error("No benchmark images generated. Check the clean directory.")
        sys.exit(1)

    total_variants = sum(len(v) for v in results.values())
    print(
        f"\n[OK] Benchmark generation complete: "
        f"{len(results)} source images x {len(DEGRADATION_PIPELINE)} variants "
        f"= {total_variants} degraded images."
    )
    if args.dry_run:
        print("  (DRY-RUN: no files written)")


if __name__ == "__main__":
    main()
