"""
Sai số ước lượng của lõi đo (P1) trên bộ synthetic (C5, data/synthetic/calibration/).

Mỗi ảnh sạch có một biến thể đối chứng `<nguồn>__clean`. Giá trị "thật" của một biến thể được
suy ra từ tham số degradation trong params.json kết hợp với số đo trên ảnh đối chứng (ảnh sạch
vốn đã có chút nhiễu, mờ và ám màu):

- nhiễu Gaussian σ (thêm độc lập trên từng kênh RGB): σ luma thật = sqrt(σ0² + (k·σ)²), với
  k = ||(0.299, 0.587, 0.114)|| ≈ 0.669 vì luma là tổ hợp tuyến tính của ba kênh;
- mờ Gaussian σ: σ defocus thật = sqrt(σ0² + σ²); độ rộng cạnh quy về σ (chia 2.563) thật
  = sqrt((w0 / 2.563)² + σ²) với w0 là độ rộng cạnh trên ảnh đối chứng;
- mờ chuyển động góc θ: gradient còn lại vuông góc hướng chuyển động, nên hướng gradient trội
  kỳ vọng = (θ + 90) mod 180 (sai số theo hướng, chu kỳ 180°);
- nguồn sáng (nhân hệ số trên giá trị sRGB): nguồn sáng tuyến tính kỳ vọng
  ∝ nguồn sáng đối chứng · gains^2.2; sai số = góc giữa vector đo và vector kỳ vọng.

Chạy từ thư mục gốc repo (cần đã sinh bộ synthetic bằng scripts/build_calibration_set.py):
    python scripts/benchmark_measurement.py
    python scripts/benchmark_measurement.py --limit 2 --out data/real/benchmark_results/m
Ghi results.json + report.md vào data/real/benchmark_results/<timestamp>-measurement-synthetic/.
Không gọi Gemini.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from scripts.benchmark_scoring import estimation_summary  # noqa: E402
from scripts.build_calibration_set import CCT_GAINS, TINT_GREEN_GAIN  # noqa: E402
from src.analyzer_evaluator.measurement import MeasurementReport, measure_image  # noqa: E402

logger = logging.getLogger("benchmark_measurement")

SYNTHETIC_DIR = Path("data/synthetic/calibration")
RESULTS_DIR = Path("data/real/benchmark_results")
LUMA_NOISE_GAIN = math.sqrt(0.299**2 + 0.587**2 + 0.114**2)
GAUSSIAN_WIDTH_10_90 = 2.5631
SRGB_GAMMA = 2.2
# Ngưỡng "đạt" của plan §0.4 / brief measure
NOISE_TOLERANCE = 0.15
BLUR_TOLERANCE = 0.25
ANGLE_TOLERANCE_DEG = 3.0
MOTION_TOLERANCE_DEG = 15.0


def _value(report: MeasurementReport, section: str, key: str) -> Optional[float]:
    """Giá trị một key đo; None khi không ước lượng được."""
    estimate = getattr(report, section).get(key)
    return None if estimate is None else estimate.value


def _illuminant(report: MeasurementReport) -> Optional[np.ndarray]:
    """Vector chromaticity nguồn sáng (r, g, b) đo được."""
    parts = [_value(report, "color", f"illuminant_{c}") for c in "rgb"]
    if any(p is None for p in parts):
        return None
    return np.array(parts, dtype=np.float64)


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Góc giữa hai vector (độ)."""
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _orientation_error_deg(estimate: float, truth: float) -> float:
    """Sai số hướng (chu kỳ 180°), trong [0, 90]."""
    diff = abs(estimate - truth) % 180.0
    return min(diff, 180.0 - diff)


def _measure(path: Path) -> MeasurementReport:
    """Đọc ảnh PNG và đo toàn ảnh."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Missing image {path}")
    return measure_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).report


def _expected_illuminant(control: np.ndarray, gains: Sequence[float]) -> np.ndarray:
    """Nguồn sáng tuyến tính kỳ vọng khi nhân `gains` lên giá trị sRGB của ảnh đối chứng."""
    expected = control * np.power(np.asarray(gains, dtype=np.float64), SRGB_GAMMA)
    return expected / expected.sum()


def case_rows(
    stem: str, params: Dict[str, Any], report: MeasurementReport, control: MeasurementReport
) -> List[Dict[str, Any]]:
    """Các dòng (đại lượng, ước lượng, thật) của một biến thể có một degradation đo được."""
    rows: List[Dict[str, Any]] = []
    degs = params["degradations"]
    kinds = [d["type"] for d in degs]

    def add(quantity: str, estimate: Optional[float], truth: Optional[float], **extra: Any) -> None:
        if estimate is None or truth is None:
            rows.append({"case": stem, "quantity": quantity, "missing": True, **extra})
            return
        rows.append(
            {"case": stem, "quantity": quantity, "estimate": estimate, "truth": truth, **extra}
        )

    sigma0 = _value(control, "noise", "sigma_luma") or 0.0
    defocus0 = _value(control, "sharpness", "defocus_sigma") or 0.0
    for deg in degs:
        kind = deg["type"]
        if kind == "gaussian_noise":
            truth = math.hypot(sigma0, LUMA_NOISE_GAIN * deg["sigma"])
            # Nhiễu đi kèm gamma (combo) làm σ phụ thuộc độ sáng: ghi riêng
            name = "noise_sigma_luma" if kinds == ["gaussian_noise"] else "noise_sigma_luma_combo"
            add(name, _value(report, "noise", "sigma_luma"), truth, added_sigma=deg["sigma"])
        elif kind == "gaussian_blur":
            truth = math.hypot(defocus0, deg["sigma"])
            name = "blur" if kinds == ["gaussian_blur"] else "blur_combo"
            add(
                f"{name}_defocus_sigma",
                _value(report, "sharpness", "defocus_sigma"),
                truth,
                added_sigma=deg["sigma"],
            )
            # Độ rộng cạnh so với chính nó trên ảnh đối chứng (cạnh thật không phải bậc thang)
            width = _value(report, "sharpness", "edge_width_px")
            width0 = _value(control, "sharpness", "edge_width_px") or 0.0
            add(
                f"{name}_edge_width_sigma",
                None if width is None else width / GAUSSIAN_WIDTH_10_90,
                math.hypot(width0 / GAUSSIAN_WIDTH_10_90, deg["sigma"]),
                added_sigma=deg["sigma"],
            )
        elif kind == "motion_blur":
            angle = _value(report, "sharpness", "motion_angle_deg")
            truth_angle = (deg["angle"] + 90.0) % 180.0
            coherence = _value(report, "sharpness", "motion_coherence")
            coherence0 = _value(control, "sharpness", "motion_coherence")
            angle0 = _value(control, "sharpness", "motion_angle_deg")
            add(
                "motion_angle_deg",
                angle,
                truth_angle,
                length=deg["length"],
                coherence=coherence,
                control_coherence=coherence0,
                control_angle=angle0,
            )
        elif kind in ("illuminant_shift", "tint_green"):
            gains = (
                CCT_GAINS[deg["cct"]] if kind == "illuminant_shift" else (1.0, TINT_GREEN_GAIN, 1.0)
            )
            measured = _illuminant(report)
            reference = _illuminant(control)
            if measured is None or reference is None:
                add("illuminant_vector_error_deg", None, None)
                continue
            expected = _expected_illuminant(reference, gains)
            neutral = np.ones(3)
            rows.append(
                {
                    "case": stem,
                    "quantity": "illuminant_vector_error_deg",
                    "error": _vector_angle_deg(measured, expected),
                    "estimate_cast_deg": _vector_angle_deg(measured, neutral),
                    "truth_cast_deg": _vector_angle_deg(expected, neutral),
                    "label": deg.get("cct", "tint_green"),
                }
            )
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Tóm tắt sai số theo đại lượng."""
    by_quantity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_quantity[row["quantity"]].append(row)
    out: Dict[str, Any] = {}
    for quantity, items in sorted(by_quantity.items()):
        usable = [r for r in items if not r.get("missing")]
        missing = len(items) - len(usable)
        if quantity == "motion_angle_deg":
            errors = [_orientation_error_deg(r["estimate"], r["truth"]) for r in usable]
            summary = _error_stats(errors, MOTION_TOLERANCE_DEG)
        elif quantity == "illuminant_vector_error_deg":
            summary = _error_stats([r["error"] for r in usable], ANGLE_TOLERANCE_DEG)
        else:
            tolerance = NOISE_TOLERANCE if quantity.startswith("noise") else BLUR_TOLERANCE
            summary = estimation_summary(
                [(r["estimate"], r["truth"]) for r in usable], tolerance=tolerance
            )
            signed = [(r["estimate"] - r["truth"]) / r["truth"] for r in usable if r["truth"]]
            summary["mean_signed"] = round(float(np.mean(signed)), 4) if signed else None
        summary["missing"] = missing
        out[quantity] = summary
    return out


def _error_stats(errors: Sequence[float], tolerance: float) -> Dict[str, Any]:
    """MAE, median, tỉ lệ trong ngưỡng cho sai số tuyệt đối (độ)."""
    if not errors:
        return {"n": 0, "mae": None, "median": None, "within_tolerance": None}
    return {
        "n": len(errors),
        "mae": round(float(np.mean(errors)), 3),
        "median": round(float(np.median(errors)), 3),
        "within_tolerance": round(sum(e <= tolerance for e in errors) / len(errors), 3),
    }


def write_report(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    """Báo cáo markdown: bảng tổng và chi tiết từng dòng."""
    lines = [
        "# Measurement estimation error on the synthetic calibration set",
        "",
        "| quantity | n | MAE | median | within tolerance | mean signed | missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for quantity, s in summary.items():
        lines.append(
            f"| {quantity} | {s['n']} | {s['mae']} | {s['median']} | {s['within_tolerance']} | "
            f"{s.get('mean_signed', '–')} | {s['missing']} |"
        )
    lines += [
        "",
        "Relative error for noise/blur (tolerance 15 % / 25 %), degrees for angles "
        "(motion orientation 15°, illuminant 3°).",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append("- " + json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(limit: Optional[int], out_dir: Path) -> Dict[str, Any]:
    """Đo mọi biến thể, ghép với ảnh đối chứng cùng nguồn, ghi kết quả."""
    params_paths = sorted(SYNTHETIC_DIR.glob("*.params.json"))
    groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for params_path in params_paths:
        stem = params_path.name.removesuffix(".params.json")
        groups[stem.split("__", 1)[0]].append((stem, params_path))
    sources = sorted(groups)[:limit] if limit else sorted(groups)
    if not sources:
        raise SystemExit(f"No synthetic cases in {SYNTHETIC_DIR}; run build_calibration_set.py.")

    rows: List[Dict[str, Any]] = []
    seconds: List[float] = []
    for source in sources:
        control_path = SYNTHETIC_DIR / f"{source}__clean.png"
        control = _measure(control_path)
        for stem, params_path in groups[source]:
            params = json.loads(params_path.read_text(encoding="utf-8"))
            if not params["degradations"]:
                continue
            started = time.time()
            report = _measure(params_path.with_name(stem + ".png"))
            seconds.append(time.time() - started)
            rows.extend(case_rows(stem, params, report, control))
        logger.info("%s: %d variants measured", source, len(groups[source]))

    summary = summarize(rows)
    summary["_meta"] = {
        "sources": len(sources),
        "mean_seconds": round(float(np.mean(seconds)), 2) if seconds else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", {k: v for k, v in summary.items() if k != "_meta"}, rows)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, help="Chỉ lấy N ảnh nguồn đầu tiên")
    parser.add_argument("--out", type=Path, help="Thư mục kết quả")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_dir = args.out or RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-measurement-synthetic"
    summary = run(args.limit, out_dir)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Report written to %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
