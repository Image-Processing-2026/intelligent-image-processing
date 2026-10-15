"""
Benchmark tốc độ cắt theo mask (crop-to-mask) của Module 3 ở độ phân giải thật.

So sánh hai đường chạy cho từng thuật toán được phép cắt:
    full  Chạy thuật toán trên toàn ảnh rồi hòa trộn (hành vi trước khi có crop-to-mask).
    crop  apply_* hiện tại: chỉ chạy trên hộp bao của mask > 0 cộng halo.
và kiểm tra hai kết quả GIỐNG HỆT TỪNG BYTE (script dừng với lỗi nếu không).

Ảnh: `astronaut` của scikit-image phóng lên 1920×1080 và 4000×3000, cộng nhiễu Gauss σ = 10.
Mask: elip nhị phân làm mềm bằng create_soft_mask của Module 2 (feather 15), chiếm ~5% và ~25% ảnh.
Thời gian: nhỏ nhất trong `--repeats` lần chạy (NLM ở 12MP chỉ chạy 1 lần vì quá chậm).

Cách chạy:
    python scripts/benchmark_crop_to_mask.py --out-dir docs/module3/benchmark
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing_engine import (  # noqa: E402
    apply_color_balance,
    apply_denoise,
    apply_gamma,
    apply_sharpen,
)
from src.processing_engine.base import apply_region_op  # noqa: E402
from src.processing_engine.color import _raw_color_balance  # noqa: E402
from src.processing_engine.denoise import _raw_denoise  # noqa: E402
from src.processing_engine.exposure_contrast import _raw_gamma  # noqa: E402
from src.processing_engine.sharpen import _raw_sharpen  # noqa: E402
from src.region_engine.mask_utils import create_soft_mask  # noqa: E402

logger = logging.getLogger("module3.benchmark.crop")

SEED = 2026
RESOLUTIONS = ((1920, 1080), (4000, 3000))
MASK_AREAS = (0.05, 0.25)

# (tên, hàm public, hàm thuần túy, tham số)
OPS = (
    ("denoise nlm s=0.8", apply_denoise, _raw_denoise, {"method": "nlm", "strength": 0.8}),
    (
        "denoise gaussian s=2.0",
        apply_denoise,
        _raw_denoise,
        {"method": "gaussian", "strength": 2.0},
    ),
    ("denoise median s=2.0", apply_denoise, _raw_denoise, {"method": "median", "strength": 2.0}),
    (
        "sharpen unsharp a=1.0",
        apply_sharpen,
        _raw_sharpen,
        {"method": "unsharp_mask", "amount": 1.0},
    ),
    ("gamma 2.0", apply_gamma, _raw_gamma, {"gamma": 2.0}),
    (
        "temperature_shift -0.3",
        apply_color_balance,
        _raw_color_balance,
        {"temperature_shift": -0.3},
    ),
)


def make_image(width: int, height: int) -> np.ndarray:
    img = cv2.resize(skdata.astronaut(), (width, height), interpolation=cv2.INTER_CUBIC)
    noise = np.random.default_rng(SEED).normal(0.0, 10.0, img.shape)
    return np.clip(np.rint(img + noise), 0, 255).astype(np.uint8)


def make_mask(width: int, height: int, area: float) -> np.ndarray:
    """Elip (tỉ lệ trục 1.44, kiểu khuôn mặt) đặt lệch trái, diện tích ≈ area · ảnh."""
    binary = np.zeros((height, width), np.uint8)
    semi_major = int(np.sqrt(area * width * height * 1.44 / np.pi))
    axes = (int(semi_major / 1.44), semi_major)
    cv2.ellipse(binary, (width // 3, height // 2), axes, 0, 0, 360, 1, -1)
    return create_soft_mask(binary, feather_radius=15)


def best_time(fn, repeats: int) -> tuple[float, np.ndarray]:
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run(repeats: int) -> list[dict]:
    rows = []
    for width, height in RESOLUTIONS:
        image = make_image(width, height)
        for area in MASK_AREAS:
            mask = make_mask(width, height, area)
            for name, public_op, raw_op, kwargs in OPS:
                n = 1 if "nlm" in name and width * height > 5e6 else repeats
                t_full, full = best_time(
                    lambda: apply_region_op(image, mask, raw_op, crop_halo=None, **kwargs), n
                )
                t_crop, crop = best_time(lambda: public_op(image, mask=mask, **kwargs), n)
                if not np.array_equal(full, crop):
                    diff = int(np.abs(full.astype(int) - crop.astype(int)).max())
                    raise AssertionError(f"{name} {width}x{height}: crop differs (max {diff})")
                rows.append(
                    {
                        "resolution": f"{width}x{height}",
                        "mask_area": round(float(np.count_nonzero(mask)) / mask.size, 4),
                        "op": name,
                        "full_ms": round(t_full * 1000, 1),
                        "crop_ms": round(t_crop * 1000, 1),
                        "speedup": round(t_full / t_crop, 2),
                        "identical": True,
                    }
                )
                logger.info("%s", rows[-1])
    return rows


def summarize(rows: list[dict]) -> str:
    lines = [
        "| độ phân giải | mask > 0 | thuật toán | toàn ảnh (ms) | cắt theo mask (ms) | tăng tốc |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['resolution']} | {r['mask_area']:.1%} | {r['op']} | {r['full_ms']:.1f} | "
            f"{r['crop_ms']:.1f} | ×{r['speedup']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", type=Path, default=Path("docs/module3/benchmark"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rows = run(args.repeats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "crop_to_mask.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "crop_to_mask.md").write_text(
        f"# Module 3 benchmark — crop_to_mask\n\nSeed {SEED}, ảnh `astronaut` phóng to + nhiễu σ = 10, "
        f"mask create_soft_mask (feather 15). Thời gian nhỏ nhất trong {args.repeats} lần "
        "(NLM 12MP: 1 lần). Mọi dòng đã kiểm tra kết quả giống hệt từng byte.\n\n"
        + summarize(rows)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
