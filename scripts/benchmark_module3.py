"""
Benchmark Module 3 (Processing Engine) trên ảnh có ground-truth với suy giảm có kiểm soát.

Ảnh sạch: ảnh mẫu đóng gói sẵn trong scikit-image (chạy offline, không tải mạng).
Suy giảm: seed cố định → kết quả tái lập được giữa các lần chạy và các máy.
Chỉ số: PSNR/SSIM cùng cấu hình với Module 1 (data_range=255, channel_axis=2);
màu sắc đo bằng ΔE CIEDE2000 trung bình (thấp = gần ảnh gốc hơn).

Các phần (--sections):
    calibrate Quét hệ số từng method khi biết σ nhiễu → chọn hằng số trong denoise.py.
    denoise   Quét strength cho 4 method ở nhiều mức σ nhiễu → kiểm tra quy ước σ = 12.5 · strength.
    exposure  Gamma khôi phục ảnh thiếu sáng; CLAHE trên ảnh tương phản thấp.
    sharpen   Unsharp/Laplacian trên ảnh mờ defocus.
    color     white_balance (gray_world / white_patch) và temperature_shift trên ảnh ám màu.

Cách chạy:
    python scripts/benchmark_module3.py
    python scripts/benchmark_module3.py --sections denoise --out-dir docs/module3/benchmark
"""

import argparse
import csv
import itertools
import logging
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing_engine import (  # noqa: E402
    apply_clahe,
    apply_color_balance,
    apply_denoise,
    apply_gamma,
    apply_sharpen,
    apply_white_balance,
)

logger = logging.getLogger("module3.benchmark")

# Ảnh RGB đóng gói sẵn trong scikit-image (không cần pooch/mạng)
IMAGE_NAMES = ("astronaut", "coffee", "chelsea", "rocket")
MAX_SIDE = 384  # thu nhỏ để benchmark chạy nhanh trên CPU; giữ tỉ lệ khung hình
SEED = 2026

# Quy ước của Module 3: strength = σ_target / 12.5 (σ theo thang 0–255)
SIGMA_PER_STRENGTH = 12.5
DENOISE_SIGMAS = (5, 10, 15, 20, 25, 40)  # σ = 40 vượt mức mục tiêu (strength tối đa 2.0 ↔ σ 25)
DENOISE_METHODS = ("gaussian", "median", "bilateral", "nlm")
STRENGTH_GRID = (0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


# ---------------------------------------------------------------------------
# Dữ liệu & chỉ số
# ---------------------------------------------------------------------------


def load_clean_images() -> dict[str, np.ndarray]:
    """Nạp ảnh sạch RGB uint8, thu nhỏ cạnh dài về MAX_SIDE bằng INTER_AREA (chống răng cưa)."""
    images = {}
    for name in IMAGE_NAMES:
        img = getattr(skdata, name)()
        scale = MAX_SIDE / max(img.shape[:2])
        if scale < 1.0:
            size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        images[name] = np.ascontiguousarray(img, dtype=np.uint8)
    return images


def metrics(result: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    return {
        "psnr": float(peak_signal_noise_ratio(clean, result, data_range=255)),
        "ssim": float(structural_similarity(clean, result, data_range=255, channel_axis=2)),
        "delta_e": float(deltaE_ciede2000(rgb2lab(clean), rgb2lab(result)).mean()),
    }


def _rng(*key: object) -> np.random.Generator:
    """Generator riêng cho mỗi (ảnh, suy giảm): kết quả không phụ thuộc thứ tự chạy."""
    # zlib.crc32 ổn định giữa các tiến trình (hash() của str bị ngẫu nhiên hóa mỗi lần chạy)
    return np.random.default_rng([SEED, *(zlib.crc32(str(k).encode()) for k in key)])


def add_gaussian_noise(img: np.ndarray, sigma: float, key: str) -> np.ndarray:
    noise = _rng(key, "noise", sigma).normal(0.0, sigma, img.shape)
    return np.clip(np.rint(img.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def gamma_degrade(img: np.ndarray, exponent: float) -> np.ndarray:
    """I_out = 255 · (I/255)^exponent; exponent > 1 làm tối (thiếu sáng)."""
    return np.clip(np.rint(255.0 * (img / 255.0) ** exponent), 0, 255).astype(np.uint8)


def low_contrast(img: np.ndarray, lo: int = 80, hi: int = 160) -> np.ndarray:
    return np.rint(lo + img.astype(np.float64) * (hi - lo) / 255.0).astype(np.uint8)


def defocus_blur(img: np.ndarray, radius: int = 2) -> np.ndarray:
    k = 2 * radius + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float32)
    return cv2.filter2D(img, -1, disk / disk.sum(), borderType=cv2.BORDER_REFLECT)


def color_cast(img: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(np.rint(img.astype(np.float64) * np.array(gains)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Các phần benchmark — mỗi hàm trả về list[dict] (1 dòng / ảnh / cấu hình)
# ---------------------------------------------------------------------------


def bench_denoise(images: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for (name, clean), sigma in itertools.product(images.items(), DENOISE_SIGMAS):
        noisy = add_gaussian_noise(clean, sigma, name)
        rows.append(_row("denoise", name, f"sigma={sigma}", "input", None, noisy, clean, 0.0))
        for method, strength in itertools.product(DENOISE_METHODS, STRENGTH_GRID):
            t0 = time.perf_counter()
            out = apply_denoise(noisy, method=method, strength=strength)
            rows.append(
                _row(
                    "denoise",
                    name,
                    f"sigma={sigma}",
                    method,
                    strength,
                    out,
                    clean,
                    time.perf_counter() - t0,
                )
            )
    return rows


# Lưới ứng viên cho hệ số của từng method (đơn vị: theo σ_target, xem src/processing_engine/denoise.py)
CALIBRATION_GRID = {
    "gaussian": ("GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA", (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)),
    "median": ("MEDIAN_SIGMA_PER_KERNEL_STEP", (10.0, 12.5, 15.0, 17.5, 20.0, 30.0)),
    "bilateral": ("BILATERAL_SIGMA_PER_NOISE_SIGMA", (2.0, 3.0, 4.0, 5.0, 6.0)),
    "nlm": ("NLM_H_PER_NOISE_SIGMA", (0.5, 0.6, 0.7, 0.8, 0.9)),
}
CALIBRATION_SIGMAS = (5, 10, 15, 20, 25)  # dải mục tiêu của pipeline (strength ≤ 2.0)


def bench_calibrate(images: dict[str, np.ndarray]) -> list[dict]:
    """
    Hiệu chỉnh hệ số khử nhiễu: biết trước σ nhiễu, đặt strength = σ / 12.5 rồi quét hệ số.
    Thay hằng số trên chính module denoise để đo đúng code chạy thật, khôi phục sau mỗi method.
    """
    from src.processing_engine import denoise as denoise_module

    rows = []
    for method, (constant, grid) in CALIBRATION_GRID.items():
        original = getattr(denoise_module, constant)
        try:
            for coef in grid:
                setattr(denoise_module, constant, coef)
                for (name, clean), sigma in itertools.product(images.items(), CALIBRATION_SIGMAS):
                    noisy = add_gaussian_noise(clean, sigma, name)
                    strength = sigma / denoise_module.NOISE_SIGMA_PER_STRENGTH
                    t0 = time.perf_counter()
                    out = apply_denoise(noisy, method=method, strength=strength)
                    seconds = time.perf_counter() - t0
                    rows.append(
                        _row("calibrate", name, f"{constant}", method, coef, out, clean, seconds)
                    )
        finally:
            setattr(denoise_module, constant, original)
    return rows


def bench_exposure(images: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, clean in images.items():
        for exponent in (1.5, 2.0, 2.5):
            dark = gamma_degrade(clean, exponent)
            case = f"underexposed_exp={exponent}"
            rows.append(_row("gamma", name, case, "input", None, dark, clean, 0.0))
            for gamma in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
                out = apply_gamma(dark, gamma=gamma)
                rows.append(_row("gamma", name, case, "gamma", gamma, out, clean, 0.0))
        flat = low_contrast(clean)
        rows.append(_row("clahe", name, "low_contrast", "input", None, flat, clean, 0.0))
        for clip in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            out = apply_clahe(flat, clip_limit=clip)
            rows.append(_row("clahe", name, "low_contrast", "clahe", clip, out, clean, 0.0))
    return rows


def bench_sharpen(images: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, clean in images.items():
        blurred = defocus_blur(clean, radius=2)
        rows.append(_row("sharpen", name, "defocus_r=2", "input", None, blurred, clean, 0.0))
        for method, amount in itertools.product(
            ("unsharp_mask", "laplacian"), (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
        ):
            out = apply_sharpen(blurred, method=method, amount=amount)
            rows.append(_row("sharpen", name, "defocus_r=2", method, amount, out, clean, 0.0))
    return rows


COLOR_CASTS = {
    "warm": (1.20, 1.0, 0.80),
    "cool": (0.85, 1.0, 1.18),
    "green": (0.90, 1.15, 0.90),
}


def bench_color(images: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for (name, clean), (cast, gains) in itertools.product(images.items(), COLOR_CASTS.items()):
        casted = color_cast(clean, gains)
        case = f"cast={cast}"
        rows.append(_row("color", name, case, "input", None, casted, clean, 0.0))
        for method in ("gray_world", "white_patch"):
            out = apply_white_balance(casted, method=method)
            # Bất biến nguồn sáng: WB(ảnh ám) phải ≈ WB(ảnh sạch). Chỉ số này công bằng hơn ΔE so với
            # ground-truth, vì ảnh gốc tự nó không trung tính (VD: coffee có trung bình R=159, B=52).
            reference = apply_white_balance(clean, method=method)
            rows.append(
                _row(
                    "color", name, case, f"wb_{method}", None, out, clean, 0.0, reference=reference
                )
            )
        for shift in (-1.0, -0.5, -0.35, -0.2, 0.2, 0.35, 0.5, 1.0):
            out = apply_color_balance(casted, temperature_shift=shift)
            rows.append(_row("color", name, case, "temperature_shift", shift, out, clean, 0.0))
    return rows


def _row(task, image, case, method, param, out, clean, seconds, reference=None) -> dict:
    invariance = np.nan
    if reference is not None:
        invariance = float(deltaE_ciede2000(rgb2lab(reference), rgb2lab(out)).mean())
    return {
        "task": task,
        "image": image,
        "case": case,
        "method": method,
        "param": "" if param is None else param,
        **{k: round(v, 4) for k, v in metrics(out, clean).items()},
        "invariance_de": round(invariance, 4),
        "seconds": round(seconds, 4),
    }


SECTIONS = {
    "calibrate": bench_calibrate,
    "denoise": bench_denoise,
    "exposure": bench_exposure,
    "sharpen": bench_sharpen,
    "color": bench_color,
}


# ---------------------------------------------------------------------------
# Tổng hợp
# ---------------------------------------------------------------------------


def _mean_by(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Trung bình PSNR/SSIM/ΔE/thời gian trên các ảnh, nhóm theo `keys`."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, items in groups.items():
        agg = dict(zip(keys, key))
        for m in ("psnr", "ssim", "delta_e", "invariance_de", "seconds"):
            values = [it[m] for it in items]
            agg[m] = float("nan") if np.all(np.isnan(values)) else float(np.nanmean(values))
        out.append(agg)
    return out


def summarize(rows: list[dict]) -> str:
    """Bảng Markdown: cấu hình tốt nhất (theo PSNR) cho mỗi tác vụ/ca suy giảm/method."""
    lines = []
    summary = _mean_by(rows, ("task", "case", "method", "param"))
    for task in dict.fromkeys(r["task"] for r in summary):
        lines += [
            f"\n### {task}\n",
            "| case | method | best param | PSNR | SSIM | ΔE vs GT | ΔE invariance | s/img |",
            "|---|---|---|---|---|---|---|---|",
        ]
        task_rows = [r for r in summary if r["task"] == task]
        for case, method in dict.fromkeys((r["case"], r["method"]) for r in task_rows):
            cands = [r for r in task_rows if r["case"] == case and r["method"] == method]
            best = max(cands, key=lambda r: r["psnr"])
            lines.append(
                f"| {case} | {method} | {best['param']} | {best['psnr']:.2f} | "
                f"{best['ssim']:.4f} | {best['delta_e']:.2f} | "
                f"{_fmt(best['invariance_de'])} | {best['seconds']:.3f} |"
            )
        if task == "denoise":
            lines += ["", _strength_calibration_table(task_rows)]
        if task == "calibrate":
            lines += ["", _coefficient_table(task_rows)]
    return "\n".join(lines)


# Ứng viên kém PSNR tốt nhất không quá ngưỡng này (dB) được xét lại theo SSIM
PSNR_TIE_DB = 0.15


def select_coefficient(cands: list[dict]) -> dict:
    """PSNR cao nhất; trong các ứng viên kém ≤ PSNR_TIE_DB, chọn SSIM cao nhất."""
    best_psnr = max(r["psnr"] for r in cands)
    return max((r for r in cands if r["psnr"] >= best_psnr - PSNR_TIE_DB), key=lambda r: r["ssim"])


def _coefficient_table(rows: list[dict]) -> str:
    lines = [
        f"Toàn bộ ứng viên (trung bình ảnh × σ ∈ {CALIBRATION_SIGMAS}); "
        f"chọn = PSNR cao nhất, hòa trong {PSNR_TIE_DB} dB thì xét SSIM:\n",
        "| method | hằng số | hệ số | PSNR | SSIM | s/img | chọn |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in dict.fromkeys(r["method"] for r in rows):
        cands = [r for r in rows if r["method"] == method]
        chosen = select_coefficient(cands)
        for r in sorted(cands, key=lambda r: r["param"]):
            lines.append(
                f"| {method} | {r['case']} | {r['param']} | {r['psnr']:.3f} | "
                f"{r['ssim']:.4f} | {r['seconds']:.3f} | {'✅' if r is chosen else ''} |"
            )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return "—" if np.isnan(value) else f"{value:.2f}"


def _strength_calibration_table(rows: list[dict]) -> str:
    """So sánh strength tối ưu đo được với quy ước strength = σ / 12.5."""
    lines = [
        "Strength tối ưu (PSNR) so với quy ước σ/12.5:\n",
        "| σ | target strength | " + " | ".join(DENOISE_METHODS) + " |",
        "|---|---|" + "---|" * len(DENOISE_METHODS),
    ]
    for sigma in DENOISE_SIGMAS:
        case = f"sigma={sigma}"
        best = []
        for method in DENOISE_METHODS:
            cands = [r for r in rows if r["case"] == case and r["method"] == method]
            best.append(str(max(cands, key=lambda r: r["psnr"])["param"]))
        lines.append(f"| {sigma} | {sigma / SIGMA_PER_STRENGTH:.2f} | " + " | ".join(best) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sections", nargs="+", choices=list(SECTIONS), default=list(SECTIONS))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/module3/benchmark"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    images = load_clean_images()
    logger.info("Loaded %d clean images: %s", len(images), {k: v.shape for k, v in images.items()})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for section in args.sections:
        t0 = time.perf_counter()
        rows = SECTIONS[section](images)
        csv_path = args.out_dir / f"{section}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        md_path = args.out_dir / f"{section}.md"
        md_path.write_text(
            f"# Module 3 benchmark — {section}\n\nSeed {SEED}, ảnh {', '.join(IMAGE_NAMES)} "
            f"(cạnh dài {MAX_SIDE}px). Trung bình trên các ảnh.\n" + summarize(rows) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Section %s: %d rows in %.1fs -> %s",
            section,
            len(rows),
            time.perf_counter() - t0,
            md_path,
        )


if __name__ == "__main__":
    main()
