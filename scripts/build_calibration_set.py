"""
Sinh bộ hiệu chỉnh synthetic: mỗi ảnh sạch thành nhiều biến thể với tham số biết trước.

Mỗi biến thể gồm `<tên>.png` (RGB uint8) và `<tên>.params.json` cạnh ảnh, đúng format C5 của
docs/analysis_v2/contracts.md: nguồn, seed, danh sách degradation và đáp án defect mong đợi.
Bộ này dùng để hiệu chỉnh severity (P2) và kiểm tra sai số ước lượng của phép đo (P1).

Ảnh sạch lấy từ data/real/benchmark/already_good/ và các nguồn trong data/calibration_sources.json
(Wikimedia Commons, có giấy phép). Ảnh được thu về cạnh dài tối đa `--max-side` px.

Quy ước:
- `out = 255 * (in / 255) ** gamma`: gamma > 1 làm tối (đáp án underexposed), gamma < 1 làm sáng
  (đáp án overexposed).
- Nhiễu Poisson-Gaussian: độ lệch chuẩn theo mức sáng I (0–255) là sqrt(a·I + b²).
- Haze: I = J·t + A(1−t), với J là ảnh sạch, t là độ truyền, A = 220 (ánh sáng khí quyển).
- Severity trong đáp án là bảng ngưỡng ban đầu theo độ lớn tham số (SEVERITY_RULES); P2 sẽ
  hiệu chỉnh lại. JPEG không thuộc từ vựng lỗi của toolbox nên có đáp án rỗng.

Chạy từ thư mục gốc repo (--fetch-sources tải ảnh sạch bổ sung từ Commons trước khi sinh):
    python scripts/build_calibration_set.py --fetch-sources --out data/synthetic/calibration/
    python scripts/build_calibration_set.py --out data/synthetic/calibration/ --limit 1
"""

import argparse
import json
import logging
import math
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("build_calibration_set")

ROOT = Path(__file__).resolve().parent.parent
ALREADY_GOOD = ROOT / "data" / "real" / "benchmark" / "already_good"
SOURCES_MANIFEST = ROOT / "data" / "calibration_sources.json"
DEFAULT_OUT = ROOT / "data" / "synthetic" / "calibration"
DEFAULT_SEED = 7
DEFAULT_MAX_SIDE = 1024
# Commons yêu cầu User-Agent mô tả rõ client
USER_AGENT = "AIImageDoctor-benchmark/0.1 (university course project; python urllib)"
# Ánh sáng khí quyển cho mô hình haze
HAZE_AIRLIGHT = 220.0
# Hệ số nhân kênh (R, G, B) cho các nhiệt màu, xấp xỉ quanh D65 (6500K)
CCT_GAINS: Dict[int, Tuple[float, float, float]] = {
    3000: (1.20, 1.00, 0.75),
    4500: (1.08, 1.00, 0.90),
    8000: (0.92, 1.00, 1.10),
}
TINT_GREEN_GAIN = 1.12
SUPPORTED_TYPES = (
    "gaussian_noise",
    "poisson_gaussian_noise",
    "chroma_noise",
    "gaussian_blur",
    "motion_blur",
    "gamma",
    "contrast_compress",
    "illuminant_shift",
    "tint_green",
    "saturation_scale",
    "jpeg",
    "haze",
)


def _sev(value: float, thresholds: Tuple[float, float]) -> int:
    """Severity 1/2/3 theo hai ngưỡng: dưới t0 → 1, dưới t1 → 2, còn lại → 3."""
    if value < thresholds[0]:
        return 1
    if value < thresholds[1]:
        return 2
    return 3


def _expected_defects(deg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Đáp án defect mong đợi của một degradation (region luôn là 'full' vì tác động toàn ảnh)."""

    def defect(defect_type: str, severity: int) -> Dict[str, Any]:
        return {"type": defect_type, "region": "full", "severity": severity}

    kind = deg["type"]
    if kind == "gaussian_noise":
        return [defect("noise", _sev(deg["sigma"], (4.0, 12.0)))]
    if kind == "poisson_gaussian_noise":
        return [defect("noise", _sev(math.sqrt(deg["b"] ** 2 + deg["a"] * 128), (4.0, 8.0)))]
    if kind == "chroma_noise":
        return [defect("noise", _sev(deg["sigma"], (4.0, 12.0)))]
    if kind == "gaussian_blur":
        return [defect("blur", _sev(deg["sigma"], (1.2, 2.5)))]
    if kind == "motion_blur":
        return [defect("blur", _sev(deg["length"], (8.0, 12.0)))]
    if kind == "gamma":
        gamma = deg["gamma"]
        kind_defect = "underexposed" if gamma > 1 else "overexposed"
        return [defect(kind_defect, _sev(abs(math.log(gamma)), (0.4, 0.6)))]
    if kind == "contrast_compress":
        return [defect("low_contrast", 2)]
    if kind == "illuminant_shift":
        r, _, b = CCT_GAINS[deg["cct"]]
        warm = deg["cct"] < 6500
        return [
            defect(
                "color_cast_warm" if warm else "color_cast_cool",
                _sev(abs(math.log(r / b)), (0.25, 0.4)),
            )
        ]
    if kind == "tint_green":
        return [defect("color_cast_green", _sev(math.log(TINT_GREEN_GAIN), (0.25, 0.4)))]
    if kind == "saturation_scale":
        scale = deg["scale"]
        if scale < 1:
            return [defect("undersaturated", 2 if scale < 0.6 else 1)]
        return [defect("oversaturated", 1 if scale < 1.8 else 2)]
    if kind == "jpeg":
        # Nén JPEG không có trong từ vựng lỗi của toolbox: đáp án rỗng, không chấm
        return []
    if kind == "haze":
        return [defect("low_contrast", 2 if deg["transmission"] < 0.7 else 1)]
    raise ValueError(f"Unknown degradation type: {kind}")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Cắt về [0, 255] và đưa về uint8."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def apply_degradation(
    image: np.ndarray, deg: Dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Áp một degradation lên ảnh RGB uint8 (H, W, 3); trả về ảnh RGB uint8 cùng shape."""
    kind = deg["type"]
    work = image.astype(np.float32)
    if kind == "gaussian_noise":
        out = work + rng.normal(0.0, deg["sigma"], work.shape)
    elif kind == "poisson_gaussian_noise":
        std = np.sqrt(deg["a"] * work + deg["b"] ** 2)
        out = work + rng.normal(0.0, 1.0, work.shape) * std
    elif kind == "chroma_noise":
        ycc = cv2.cvtColor(_to_uint8(work), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        ycc[..., 1:] += rng.normal(0.0, deg["sigma"], ycc[..., 1:].shape)
        out = cv2.cvtColor(_to_uint8(ycc), cv2.COLOR_YCrCb2RGB).astype(np.float32)
    elif kind == "gaussian_blur":
        out = cv2.GaussianBlur(work, (0, 0), sigmaX=deg["sigma"])
    elif kind == "motion_blur":
        out = _motion_blur(work, int(deg["length"]), float(deg["angle"]))
    elif kind == "gamma":
        out = 255.0 * np.power(work / 255.0, deg["gamma"])
    elif kind == "contrast_compress":
        low, high = deg["range"]
        out = low + work * (high - low) / 255.0
    elif kind == "illuminant_shift":
        out = work * np.array(CCT_GAINS[deg["cct"]], dtype=np.float32)
    elif kind == "tint_green":
        out = work * np.array((1.0, TINT_GREEN_GAIN, 1.0), dtype=np.float32)
    elif kind == "saturation_scale":
        hsv = cv2.cvtColor(_to_uint8(work), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= deg["scale"]
        out = cv2.cvtColor(_to_uint8(np.clip(hsv, 0, 255)), cv2.COLOR_HSV2RGB).astype(np.float32)
    elif kind == "jpeg":
        buffer = BytesIO()
        Image.fromarray(_to_uint8(work)).save(buffer, format="JPEG", quality=deg["quality"])
        buffer.seek(0)
        out = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
    elif kind == "haze":
        t = float(deg["transmission"])
        out = work * t + HAZE_AIRLIGHT * (1.0 - t)
    else:
        raise ValueError(f"Unknown degradation type: {kind}")
    return _to_uint8(out)


def _motion_blur(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    """Mờ chuyển động: kernel một đường thẳng dài `length` px, góc `angle` độ."""
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    radians = math.radians(angle)
    dx, dy = math.cos(radians) * center, math.sin(radians) * center
    cv2.line(
        kernel,
        (round(center - dx), round(center - dy)),
        (round(center + dx), round(center + dy)),
        1.0,
        1,
    )
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def variant_plan(seed: int = DEFAULT_SEED) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Danh sách biến thể cố định (tên, degradation). Góc mờ chuyển động lấy từ seed để tất định."""
    rng = np.random.default_rng(seed)
    motion_angles = [float(rng.uniform(0.0, 180.0)) for _ in range(2)]
    plan: List[Tuple[str, List[Dict[str, Any]]]] = [("clean", [])]
    for sigma in (2.0, 5.0, 10.0, 15.0, 25.0):
        plan.append((f"noise_s{sigma:g}", [{"type": "gaussian_noise", "sigma": sigma}]))
    plan.append(
        (
            "noise_poisson",
            [{"type": "poisson_gaussian_noise", "a": 0.25, "b": 2.0}],
        )
    )
    plan.append(("noise_chroma", [{"type": "chroma_noise", "sigma": 8.0}]))
    for sigma in (0.8, 1.5, 3.0):
        plan.append((f"blur_g{sigma:g}", [{"type": "gaussian_blur", "sigma": sigma}]))
    for length, angle in zip((5, 15), motion_angles):
        plan.append(
            (
                f"blur_m{length}",
                [{"type": "motion_blur", "length": length, "angle": round(angle, 2)}],
            )
        )
    for gamma in (0.5, 0.7, 1.5, 2.0):
        plan.append((f"gamma_{gamma:g}", [{"type": "gamma", "gamma": gamma}]))
    plan.append(("contrast_compress", [{"type": "contrast_compress", "range": [64, 192]}]))
    for cct in CCT_GAINS:
        plan.append((f"cct_{cct}", [{"type": "illuminant_shift", "cct": cct}]))
    plan.append(("tint_green", [{"type": "tint_green"}]))
    for scale in (0.5, 1.6):
        plan.append((f"sat_{scale:g}", [{"type": "saturation_scale", "scale": scale}]))
    for quality in (20, 40, 70):
        plan.append((f"jpeg_q{quality}", [{"type": "jpeg", "quality": quality}]))
    plan.append(("haze_t060", [{"type": "haze", "transmission": 0.6}]))
    plan.append(
        (
            "combo_noise_dark",
            [{"type": "gaussian_noise", "sigma": 10.0}, {"type": "gamma", "gamma": 1.5}],
        )
    )
    plan.append(
        (
            "combo_blur_noise",
            [{"type": "gaussian_blur", "sigma": 1.5}, {"type": "gaussian_noise", "sigma": 5.0}],
        )
    )
    return plan


def build_variants(
    image: np.ndarray, source_name: str, seed: int = DEFAULT_SEED
) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """Sinh mọi biến thể của một ảnh sạch: (tên biến thể, ảnh, nội dung params.json)."""
    out = []
    for name, degradations in variant_plan(seed):
        rng = np.random.default_rng(seed)
        result = image
        for deg in degradations:
            result = apply_degradation(result, deg, rng)
        expected = [d for deg in degradations for d in _expected_defects(deg)]
        params = {
            "source": source_name,
            "seed": seed,
            "degradations": degradations,
            "expected": {"defects": expected},
        }
        out.append((name, result, params))
    return out


def _load_rgb(path: Path, max_side: int) -> np.ndarray:
    """Đọc ảnh RGB uint8, thu về cạnh dài tối đa max_side (không phóng to)."""
    with Image.open(path) as pil:
        pil = pil.convert("RGB")
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
        return np.asarray(pil, dtype=np.uint8).copy()


def clean_sources(max_side: int) -> List[Tuple[str, Path]]:
    """Ảnh sạch: already_good và các nguồn trong data/calibration_sources.json (đã tải sẵn)."""
    sources = [(path.stem, path) for path in sorted(ALREADY_GOOD.glob("*.jpg"))]
    if SOURCES_MANIFEST.exists():
        data = json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8-sig"))
        for entry in data["images"]:
            path = ROOT / entry["file"]
            if path.exists():
                sources.append((entry["id"], path))
            else:
                logger.warning("Clean source missing, skipped: %s", entry["file"])
    return sources


def fetch_sources(max_side: int = DEFAULT_MAX_SIDE) -> int:
    """Tải các ảnh sạch trong data/calibration_sources.json còn thiếu; trả về số ảnh tải mới.

    Ảnh được thu về cạnh dài max_side và lưu JPEG q95. Commons yêu cầu User-Agent rõ ràng."""
    data = json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8-sig"))
    downloaded = 0
    for entry in data["images"]:
        path = ROOT / entry["file"]
        if path.exists():
            continue
        request = urllib.request.Request(entry["original_url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        with Image.open(BytesIO(raw)) as pil:
            pil = pil.convert("RGB")
            pil.thumbnail((max_side, max_side), Image.LANCZOS)
            path.parent.mkdir(parents=True, exist_ok=True)
            pil.save(path, "JPEG", quality=95)
        downloaded += 1
        logger.info("Downloaded %s (%s)", entry["id"], entry["license"])
        time.sleep(1.5)
    return downloaded


def write_variant(
    out_dir: Path, source_id: str, name: str, image: np.ndarray, params: Dict
) -> None:
    """Ghi ảnh PNG (RGB → BGR khi lưu) và params.json cạnh nhau."""
    stem = f"{source_id}__{name}"
    cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (out_dir / f"{stem}.params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build(
    out_dir: Path,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    seed: int = DEFAULT_SEED,
    limit: Optional[int] = None,
) -> int:
    """Sinh toàn bộ bộ hiệu chỉnh; trả về số ảnh đã ghi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = clean_sources(max_side)
    if limit is not None:
        sources = sources[:limit]
    written = 0
    for source_id, path in sources:
        image = _load_rgb(path, max_side)
        for name, variant, params in build_variants(image, path.name, seed):
            write_variant(out_dir, source_id, name, variant, params)
            written += 1
        logger.info("%s: %d variants", source_id, len(variant_plan(seed)))
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Thư mục đầu ra")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, help="Chỉ lấy N ảnh sạch đầu tiên")
    parser.add_argument(
        "--fetch-sources", action="store_true", help="Tải ảnh sạch bổ sung trước khi sinh"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.fetch_sources:
        logger.info("Fetched %d clean source(s)", fetch_sources(args.max_side))
    written = build(args.out, max_side=args.max_side, seed=args.seed, limit=args.limit)
    logger.info("Wrote %d images to %s", written, args.out)


if __name__ == "__main__":
    sys.exit(main())
