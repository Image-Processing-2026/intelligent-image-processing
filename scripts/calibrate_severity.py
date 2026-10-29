"""
Khớp ngưỡng severity 0–3 cho từng lỗi đo được trên bộ synthetic (Analysis v2 P2, hợp đồng C4).

Luồng:
1. Đo `measure_image` trên mọi ảnh của data/synthetic/calibration/ (sinh bởi
   scripts/build_calibration_set.py). Bảng đặc trưng được cache trong
   data/synthetic/calibration/features.json (gitignored), kèm dấu vân tay mã nguồn đo: đổi mã đo
   thì tự đo lại.
2. Nhãn severity mỗi ca: đáp án trong params.json (bảng `_sev` của bench) cộng luật nhãn cho
   defect mới của P6 (motion_blur, haze, jpeg_artifacts) ở `case_labels`. Ca không có lỗi đó có
   nhãn 0.
3. Điểm của mỗi lỗi lấy từ src/analyzer_evaluator/measurement/calibration.py (cùng hàm lúc chạy).
   Tham số khớp trước: lệch thấp của σ defocus (hồi quy qua gốc trên ca mờ Gaussian) và cổng xác
   nhận mờ chuyển động (biên lớn nhất, không ca nào không mờ chuyển động lọt qua).
4. Ngưỡng: hồi quy thứ tự đơn giản — chọn ba ngưỡng tăng dần cực tiểu tổng |severity − nhãn| trên
   mọi ca (vét cạn trên các điểm giữa của điểm số đã sắp, hòa thì lấy khe rộng nhất).
5. Kiểm chéo bỏ-một-nguồn (5 ảnh nguồn): lỗi chỉ được hiệu chỉnh khi MAE cân bằng (trung bình
   MAE ca có lỗi và ca không lỗi) kiểm chéo thấp hơn bộ dự đoán rỗng (luôn 0); ngược lại ghi
   "không đo được" (C4 trả (0, 0.0)).
6. Ghi src/analyzer_evaluator/measurement/calibration.yaml và báo cáo vào
   data/real/benchmark_results/<timestamp>-calibration/ (gitignored).

Không hiệu chỉnh trên ảnh thật (plan §Phase 2); ảnh thật chỉ để kiểm định:
    python scripts/benchmark_analysis.py --set real --config measure_only

Chạy từ thư mục gốc repo:
    python scripts/calibrate_severity.py
    python scripts/calibrate_severity.py --refresh          # đo lại toàn bộ
    python scripts/calibrate_severity.py --dry-run          # không ghi calibration.yaml
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath("."))

from src.analyzer_evaluator.measurement import MeasurementReport, measure_image  # noqa: E402
from src.analyzer_evaluator.measurement import calibration as calib  # noqa: E402
from src.analyzer_evaluator.measurement.sharpness import PRE_SMOOTH_MIN_NOISE  # noqa: E402

logger = logging.getLogger("calibrate_severity")

SYNTHETIC_DIR = Path("data/synthetic/calibration")
FEATURES_FILE = SYNTHETIC_DIR / "features.json"
MEASUREMENT_DIR = Path("src/analyzer_evaluator/measurement")
RESULTS_DIR = Path("data/real/benchmark_results")
MAX_SEVERITY = 3
# Cạnh dài tối đa của bộ synthetic (build_calibration_set.DEFAULT_MAX_SIDE): thang tham chiếu cho độ mờ
REFERENCE_LONG_SIDE = 1024
# Lỗi phải thắng bộ dự đoán rỗng trong kiểm chéo ít nhất chừng này (MAE cân bằng)
MIN_CV_GAIN = 0.005
# Mức ám nhẹ (lane color) chỉ dùng để dò: không khớp ngưỡng nào, báo riêng trong report.md. Ở thang
# tuyệt đối chúng không tách khỏi ảnh sạch (docs/analysis_v2/status/color.md), nên đưa vào khớp sẽ
# hạ ngưỡng ám mà không tách được, và đổi ngưỡng lỗi khác so với bảng cũ.
PROBE_CCT = (5000, 7000)

# Ngưỡng cho nhãn level không phải lỗi (TechnicalMetrics.contrast_level = "high"): prior giữ từ
# analyzer cũ, không khớp; nằm trong calibration.yaml để analyzer chỉ có một nguồn ngưỡng.
LEVEL_PRIORS = {"contrast_high_std": 80.0}

# Đặc tả từng lỗi: hàm điểm (calibration.SCORE_FUNCTIONS), tham số cố định (prior, không khớp),
# lỗi gộp, và loại degradation bị loại khỏi phần khớp của chính lỗi đó.
DEFECT_SPECS: Dict[str, Dict[str, Any]] = {
    "noise": {
        "score": "noise_sigma",
        # dark_weight: prior cảm nhận (nhiễu vùng tối ít thấy hơn), bộ synthetic không có nhãn để khớp.
        # chroma_weight: quy σ chroma về thang luma (σ luma = 0.669·σ mỗi kênh RGB).
        "params": {"dark_weight": 0.5, "chroma_weight": 0.669},
    },
    "blur": {
        "score": "defocus",
        "params": {},
        "combine_max": ["motion_blur"],
        # Mờ chuyển động gần như không đổi trung vị defocus; do motion_blur đảm nhận (combine_max)
        "exclude_degradations": ["motion_blur"],
    },
    "motion_blur": {"score": "motion_extent", "params": {}},
    "underexposed": {"score": "darkness", "params": {}},
    "overexposed": {"score": "brightness", "params": {}},
    "low_contrast": {"score": "flatness", "params": {}},
    "color_cast_warm": {"score": "cast_warm", "params": {}},
    "color_cast_cool": {"score": "cast_cool", "params": {}},
    "color_cast_green": {"score": "cast_green", "params": {}},
    "oversaturated": {"score": "chroma", "params": {}},
    "undersaturated": {"score": "low_chroma", "params": {}},
    "haze": {"score": "dark_channel", "params": {}},
    "jpeg_artifacts": {"score": "blockiness", "params": {}},
    # Bộ synthetic không có ca cháy sáng có nhãn: ngưỡng prior theo định nghĩa blob cháy của P1
    # (blob ≥ 0.5 % diện tích là vùng cháy), không khớp.
    "highlight_clipping": {
        "score": "clip_blob",
        "params": {},
        "prior_thresholds": [0.005, 0.03, 0.10],
    },
    # Cần so chủ thể với phần còn lại (hai vùng); một RegionStats/MeasurementReport không đủ.
    "backlit_subject": {"score": None, "params": {}},
}


def measurement_fingerprint() -> str:
    """Băm mã nguồn các module đo (trừ calibration.py): đổi mã đo thì cache đặc trưng hết hạn."""
    digest = hashlib.sha1()
    for path in sorted(MEASUREMENT_DIR.glob("*.py")):
        if path.name != "calibration.py":
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def load_cases() -> List[Dict[str, Any]]:
    """Các ca synthetic: stem, nguồn, params."""
    cases = []
    for params_path in sorted(SYNTHETIC_DIR.glob("*.params.json")):
        stem = params_path.name.removesuffix(".params.json")
        cases.append(
            {
                "stem": stem,
                "source": stem.split("__", 1)[0],
                "params": json.loads(params_path.read_text(encoding="utf-8")),
                "image": params_path.with_name(stem + ".png"),
            }
        )
    if not cases:
        raise SystemExit(f"No synthetic cases in {SYNTHETIC_DIR}; run build_calibration_set.py.")
    return cases


def is_probe(case: Dict[str, Any]) -> bool:
    """Ca dò ám nhẹ (PROBE_CCT): không dùng để khớp."""
    return any(
        d["type"] == "illuminant_shift" and d["cct"] in PROBE_CCT
        for d in case["params"]["degradations"]
    )


def load_features(cases: Sequence[Dict[str, Any]], refresh: bool) -> Dict[str, MeasurementReport]:
    """Báo cáo đo của mọi ca, đọc từ cache nếu còn khớp mã đo, ngược lại đo lại và ghi cache."""
    fingerprint = measurement_fingerprint()
    cached: Dict[str, Any] = {}
    if FEATURES_FILE.exists() and not refresh:
        data = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
        if data.get("fingerprint") == fingerprint:
            cached = data["reports"]
        else:
            logger.info("Measurement code changed; re-measuring the synthetic set.")
    reports: Dict[str, MeasurementReport] = {}
    started = time.time()
    for case in cases:
        if case["stem"] in cached:
            reports[case["stem"]] = MeasurementReport.model_validate(cached[case["stem"]])
            continue
        bgr = cv2.imread(str(case["image"]))
        if bgr is None:
            raise FileNotFoundError(f"Missing image {case['image']}")
        reports[case["stem"]] = measure_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).report
    measured = len(cases) - len([c for c in cases if c["stem"] in cached])
    if measured:
        logger.info("Measured %d images in %.1fs", measured, time.time() - started)
        FEATURES_FILE.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "reports": {k: v.model_dump() for k, v in reports.items()},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return reports


def case_labels(params: Dict[str, Any]) -> Dict[str, int]:
    """Nhãn severity theo loại lỗi của một ca (lỗi không có mặt = 0).

    Đáp án của bench (`expected.defects`) cộng luật nhãn cho defect mới của P6 (lane calib):
    - motion_blur: cùng mức với đáp án `blur` của degradation mờ chuyển động;
    - haze: 2 khi độ truyền t < 0.7, ngược lại 1 (cùng luật `low_contrast` của bench);
    - jpeg_artifacts: q ≤ 25 → 2, q ≤ 50 → 1, cao hơn → 0 (q70 gần như không thấy khối).
    """
    labels: Dict[str, int] = defaultdict(int)
    for defect in params["expected"]["defects"]:
        labels[defect["type"]] = max(labels[defect["type"]], int(defect["severity"]))
    for deg in params["degradations"]:
        if deg["type"] == "motion_blur":
            labels["motion_blur"] = max(labels["motion_blur"], 1 if deg["length"] < 8 else 3)
        elif deg["type"] == "haze":
            labels["haze"] = max(labels["haze"], 2 if deg["transmission"] < 0.7 else 1)
        elif deg["type"] == "jpeg":
            quality = deg["quality"]
            labels["jpeg_artifacts"] = 2 if quality <= 25 else 1 if quality <= 50 else 0
    return dict(labels)


def fit_thresholds(scores: Sequence[float], labels: Sequence[int]) -> List[float]:
    """Ba ngưỡng tăng dần cực tiểu Σ|severity − nhãn|, severity = số ngưỡng ≤ điểm.

    Vét cạn trên các vị trí cắt của dãy điểm phân biệt đã sắp. Nhiều nghiệm cùng chi phí thì lấy
    nghiệm có tổng khe (khoảng cách giữa hai điểm hai bên mỗi ngưỡng, chuẩn hóa theo dải điểm) lớn
    nhất. Ngưỡng không đạt tới được là +inf.
    """
    if not scores:
        return [math.inf] * MAX_SEVERITY
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    groups = len(unique)
    index = np.searchsorted(unique, np.asarray(scores, dtype=np.float64))
    counts = np.zeros((groups, MAX_SEVERITY + 1))
    for g, label in zip(index, labels):
        counts[g, int(label)] += 1
    levels = np.arange(MAX_SEVERITY + 1)
    # cost[k, g]: chi phí gán nhóm g mức k; prefix[k, b] = Σ_{g < b} cost[k, g]
    cost = np.stack([(counts * np.abs(k - levels)).sum(axis=1) for k in levels])
    prefix = np.concatenate([np.zeros((MAX_SEVERITY + 1, 1)), np.cumsum(cost, axis=1)], axis=1)
    span = float(unique[-1] - unique[0]) or 1.0
    gaps = np.zeros(groups + 1)
    gaps[1:groups] = np.diff(unique) / span

    best: Tuple[float, float, Tuple[int, int, int]] = (math.inf, -math.inf, (groups,) * 3)
    b3 = np.arange(groups + 1)
    for b1 in range(groups + 1):
        for b2 in range(b1, groups + 1):
            tail = b3[b2:]
            total = (
                prefix[0, b1]
                + prefix[1, b2]
                - prefix[1, b1]
                + prefix[2, tail]
                - prefix[2, b2]
                + prefix[3, groups]
                - prefix[3, tail]
            )
            margin = gaps[b1] + gaps[b2] + gaps[tail]
            order = np.lexsort((-margin, total))[0]
            candidate = (float(total[order]), float(margin[order]), (b1, b2, int(tail[order])))
            if candidate[0] < best[0] - 1e-9 or (
                abs(candidate[0] - best[0]) <= 1e-9 and candidate[1] > best[1] + 1e-12
            ):
                best = candidate

    thresholds = []
    for boundary in best[2]:
        if boundary >= groups:
            thresholds.append(math.inf)
        elif boundary == 0:
            thresholds.append(-math.inf)
        else:
            thresholds.append(float((unique[boundary - 1] + unique[boundary]) / 2.0))
    return thresholds


def predict(score: Optional[float], thresholds: Sequence[float]) -> int:
    """Severity dự đoán; không đo được → 0."""
    return 0 if score is None else calib.severity_for_score(score, list(thresholds))


def fit_defocus_bias(
    cases: Sequence[Dict[str, Any]], reports: Dict[str, MeasurementReport]
) -> float:
    """Lệch thấp tương đối b của σ defocus: đo ≈ (1 − b)·thật, hồi quy qua gốc trên ca mờ Gaussian.

    σ thật = sqrt(σ0² + σ²), σ0 là σ defocus đo trên ảnh đối chứng `<nguồn>__clean`.
    """
    num = den = 0.0
    for case in cases:
        degs = case["params"]["degradations"]
        if [d["type"] for d in degs] != ["gaussian_blur"]:
            continue
        measured = calib._value(reports[case["stem"]], "sharpness.defocus_sigma")
        control = calib._value(reports[f"{case['source']}__clean"], "sharpness.defocus_sigma")
        if measured is None:
            continue
        truth = math.hypot(control or 0.0, degs[0]["sigma"])
        num += measured * truth
        den += truth * truth
    return round(1.0 - num / den, 4) if den else 0.0


def fit_motion_gate(
    cases: Sequence[Dict[str, Any]], reports: Dict[str, MeasurementReport]
) -> Dict[str, float]:
    """Cổng xác nhận mờ chuyển động: (min_anisotropy, min_extent) biên lớn nhất.

    Ràng buộc: không ca nào không có mờ chuyển động lọt qua cổng; tối đa số ca mờ chuyển động
    được xác nhận; hòa thì lấy biên tương đối nhỏ nhất (tới mọi ca hai phía) lớn nhất.
    """
    positives, negatives = [], []
    for case in cases:
        report = reports[case["stem"]]
        a = calib._value(report, "sharpness.motion_anisotropy")
        e = calib._value(report, "sharpness.motion_extent_sigma")
        if a is None or e is None:
            continue
        is_motion = any(d["type"] == "motion_blur" for d in case["params"]["degradations"])
        (positives if is_motion else negatives).append((a, e))
    pos = np.array(positives) if positives else np.zeros((0, 2))
    neg = np.array(negatives) if negatives else np.zeros((0, 2))

    def midpoints(values: np.ndarray) -> np.ndarray:
        ordered = np.unique(values)
        return (ordered[:-1] + ordered[1:]) / 2.0 if ordered.size > 1 else ordered

    all_points = np.vstack([pos, neg])
    best = (-1, -math.inf, (math.inf, math.inf))
    for a0 in midpoints(all_points[:, 0]):
        for e0 in midpoints(all_points[:, 1]):
            if neg.size and np.any((neg[:, 0] >= a0) & (neg[:, 1] >= e0)):
                continue
            confirmed = pos[(pos[:, 0] >= a0) & (pos[:, 1] >= e0)] if pos.size else pos
            margins = [
                float(np.min(np.minimum((confirmed[:, 0] - a0) / a0, (confirmed[:, 1] - e0) / e0)))
                if confirmed.size
                else 0.0
            ]
            if neg.size:
                margins.append(
                    float(np.min(np.maximum((a0 - neg[:, 0]) / a0, (e0 - neg[:, 1]) / e0)))
                )
            candidate = (len(confirmed), min(margins), (float(a0), float(e0)))
            if candidate[:2] > best[:2]:
                best = candidate
    return {
        "min_anisotropy": round(best[2][0], 4),
        "min_extent": round(best[2][1], 4),
        # Không có ca mờ chuyển động kèm nhiễu: thêm prior cho ảnh nhiễu (σ defocus kém tin cậy hơn)
        "noisy_sigma": PRE_SMOOTH_MIN_NOISE,
        "noisy_extra_extent": 0.2,
    }


def _fit_rows(
    defect: str,
    cases: Sequence[Dict[str, Any]],
    reports: Dict[str, MeasurementReport],
    calibration: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """(nguồn, điểm, nhãn) của các ca dùng để khớp ngưỡng của một lỗi."""
    excluded = set(DEFECT_SPECS[defect].get("exclude_degradations", []))
    rows = []
    for case in cases:
        if excluded & {d["type"] for d in case["params"]["degradations"]}:
            continue
        score = calib.defect_score(defect, reports[case["stem"]], calibration)
        rows.append(
            {
                "stem": case["stem"],
                "source": case["source"],
                "score": None if score is None else score[0],
                "label": case_labels(case["params"]).get(defect, 0),
            }
        )
    return rows


def _mae(rows: Sequence[Dict[str, Any]], thresholds: Sequence[float]) -> float:
    return float(np.mean([abs(predict(r["score"], thresholds) - r["label"]) for r in rows]))


def _fit(rows: Sequence[Dict[str, Any]]) -> List[float]:
    scored = [r for r in rows if r["score"] is not None]
    return fit_thresholds([r["score"] for r in scored], [r["label"] for r in scored])


def balanced_mae(rows: Sequence[Dict[str, Any]], predictions: Sequence[int]) -> float:
    """Trung bình của MAE trên ca có lỗi và MAE trên ca không lỗi.

    Ca không lỗi chiếm đa số nên MAE thường luôn ưu ái bộ dự đoán rỗng; MAE cân bằng đo xem số
    đo có mang thông tin về lỗi không.
    """
    errors = np.abs(np.asarray(predictions) - np.asarray([r["label"] for r in rows]))
    positive = np.asarray([r["label"] > 0 for r in rows])
    parts = [float(errors[mask].mean()) for mask in (positive, ~positive) if mask.any()]
    return float(np.mean(parts))


def cross_validate(rows: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    """MAE (mọi ca) và MAE cân bằng, kiểm chéo bỏ-một-nguồn."""
    predictions = [0] * len(rows)
    for source in sorted({r["source"] for r in rows}):
        thresholds = _fit([r for r in rows if r["source"] != source])
        for i, row in enumerate(rows):
            if row["source"] == source:
                predictions[i] = predict(row["score"], thresholds)
    labels = np.asarray([r["label"] for r in rows])
    return float(np.mean(np.abs(np.asarray(predictions) - labels))), balanced_mae(rows, predictions)


def calibrate(
    cases: Sequence[Dict[str, Any]], reports: Dict[str, MeasurementReport]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Khớp toàn bộ: trả (nội dung calibration.yaml, thống kê cho báo cáo)."""
    sources = sorted({c["source"] for c in cases})
    calibration: Dict[str, Any] = {
        "version": f"p2-{date.today():%Y%m%d}",
        "date": f"{date.today():%Y-%m-%d}",
        "dataset": {
            "path": SYNTHETIC_DIR.as_posix(),
            "generator": "scripts/build_calibration_set.py (seed 7, max side 1024)",
            "images": len(cases),
            "sources": sources,
            "measurement_fingerprint": measurement_fingerprint(),
        },
        "method": (
            "ordinal thresholds minimising sum |severity - label| over all synthetic cases; "
            "kept only when leave-one-source-out balanced MAE beats the all-zero predictor"
        ),
        "reference_long_side": REFERENCE_LONG_SIDE,
        "levels": dict(LEVEL_PRIORS),
        "defects": {},
    }
    for defect, spec in DEFECT_SPECS.items():
        entry: Dict[str, Any] = {"score": spec["score"], "params": dict(spec["params"])}
        if spec.get("combine_max"):
            entry["combine_max"] = list(spec["combine_max"])
        calibration["defects"][defect] = entry
    calibration["defects"]["blur"]["params"]["defocus_bias"] = fit_defocus_bias(cases, reports)
    calibration["defects"]["motion_blur"]["params"].update(fit_motion_gate(cases, reports))

    stats: Dict[str, Any] = {}
    for defect, spec in DEFECT_SPECS.items():
        entry = calibration["defects"][defect]
        if spec["score"] is None:
            entry.update({"thresholds": [], "fitted": False, "reason": "needs two regions"})
            stats[defect] = {"status": "not measurable"}
            continue
        if "prior_thresholds" in spec:
            entry.update({"thresholds": list(spec["prior_thresholds"]), "fitted": False})
            entry["reason"] = "no labelled synthetic cases; prior thresholds"
            stats[defect] = {"status": "prior"}
            continue
        rows = _fit_rows(defect, cases, reports, calibration)
        positives = sum(1 for r in rows if r["label"] > 0)
        thresholds = _fit(rows)
        cv_mae, cv_balanced = cross_validate(rows)
        null_balanced = balanced_mae(rows, [0] * len(rows))
        fitted = positives > 0 and cv_balanced < null_balanced - MIN_CV_GAIN
        stats[defect] = {
            "status": "fitted" if fitted else "not measurable",
            "cases": len(rows),
            "positives": positives,
            "unscored": sum(1 for r in rows if r["score"] is None),
            "mae_all": round(_mae(rows, thresholds), 4),
            "loso_mae": round(cv_mae, 4),
            "loso_balanced_mae": round(cv_balanced, 4),
            "null_balanced_mae": round(null_balanced, 4),
            "thresholds": thresholds,
        }
        keys = (
            "cases",
            "positives",
            "mae_all",
            "loso_mae",
            "loso_balanced_mae",
            "null_balanced_mae",
        )
        entry["fit"] = {k: stats[defect][k] for k in keys}
        if fitted:
            # Phần cải thiện so với bộ dự đoán rỗng trong kiểm chéo: hạ confidence lỗi khớp yếu
            reliability = round(max(0.0, 1.0 - cv_balanced / null_balanced), 3)
            entry.update(
                {
                    "thresholds": [round(t, 4) for t in thresholds],
                    "fitted": True,
                    "reliability": reliability,
                }
            )
        else:
            entry.update({"thresholds": [], "fitted": False})
            entry["reason"] = (
                "leave-one-source-out balanced MAE does not beat the all-zero predictor"
            )
    return calibration, stats


def evaluate(
    cases: Sequence[Dict[str, Any]],
    reports: Dict[str, MeasurementReport],
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """Chỉ số trên bộ synthetic với bảng hiệu chỉnh cuối (đã gộp combine_max)."""
    per_defect: Dict[str, Dict[str, Any]] = {}
    matched_errors: List[int] = []
    rows: List[Dict[str, Any]] = []
    for case in cases:
        labels = case_labels(case["params"])
        predicted = {
            defect: calib.severity_from_calibration(defect, reports[case["stem"]], calibration)[0]
            for defect in calibration["defects"]
        }
        rows.append({"stem": case["stem"], "labels": labels, "predicted": predicted})
        for defect, severity in predicted.items():
            label = labels.get(defect, 0)
            bucket = per_defect.setdefault(
                defect, {"tp": 0, "fp": 0, "fn": 0, "abs_err": 0, "n": 0, "matched_err": []}
            )
            bucket["n"] += 1
            bucket["abs_err"] += abs(severity - label)
            if severity >= 1 and label >= 1:
                bucket["tp"] += 1
                bucket["matched_err"].append(abs(severity - label))
                matched_errors.append(abs(severity - label))
            elif severity >= 1:
                bucket["fp"] += 1
            elif label >= 1:
                bucket["fn"] += 1
    summary = {
        defect: {
            "tp": b["tp"],
            "fp": b["fp"],
            "fn": b["fn"],
            "mae_all": round(b["abs_err"] / b["n"], 4),
            "mae_matched": round(float(np.mean(b["matched_err"])), 4) if b["matched_err"] else None,
        }
        for defect, b in per_defect.items()
    }
    return {
        "per_defect": summary,
        "mae_matched": round(float(np.mean(matched_errors)), 4) if matched_errors else None,
        "rows": rows,
    }


def _fmt_thresholds(thresholds: Sequence[float]) -> str:
    return ", ".join("∞" if math.isinf(t) and t > 0 else f"{t:.3f}" for t in thresholds) or "–"


def probe_rows(
    cases: Sequence[Dict[str, Any]],
    reports: Dict[str, MeasurementReport],
    calibration: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Điểm và severity ám màu của các ca dò (không khớp)."""
    rows = []
    for case in cases:
        report = reports[case["stem"]]
        row: Dict[str, Any] = {"stem": case["stem"], "labels": case_labels(case["params"])}
        for defect in ("color_cast_warm", "color_cast_cool"):
            score = calib.defect_score(defect, report, calibration)
            row[defect] = (
                None if score is None else round(score[0], 3),
                calib.severity_from_calibration(defect, report, calibration)[0],
            )
        rows.append(row)
    return rows


def write_report(
    path: Path,
    calibration: Dict[str, Any],
    stats: Dict,
    evaluation: Dict,
    probes: Sequence[Dict[str, Any]] = (),
) -> None:
    """Báo cáo markdown: tham số khớp, ngưỡng, kiểm chéo, chỉ số theo lỗi, ca dò."""
    blur = calibration["defects"]["blur"]["params"]
    gate = calibration["defects"]["motion_blur"]["params"]
    lines = [
        f"# Severity calibration `{calibration['version']}`",
        "",
        f"Synthetic set: {calibration['dataset']['images']} images, "
        f"sources {', '.join(calibration['dataset']['sources'])}.",
        f"Defocus bias: {blur['defocus_bias']}. Motion gate: anisotropy ≥ "
        f"{gate['min_anisotropy']}, extent ≥ {gate['min_extent']}.",
        f"MAE on matched pairs (benchmark definition): {evaluation['mae_matched']}.",
        "",
        "| defect | status | thresholds | positives | MAE all | LOSO MAE | LOSO bal. | null bal. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for defect, s in stats.items():
        if "cases" not in s:
            thresholds = calibration["defects"][defect]["thresholds"]
            lines.append(f"| {defect} | {s['status']} | {_fmt_thresholds(thresholds)} | | | | | |")
            continue
        lines.append(
            f"| {defect} | {s['status']} | {_fmt_thresholds(s['thresholds'])} | {s['positives']} | "
            f"{s['mae_all']} | {s['loso_mae']} | {s['loso_balanced_mae']} | "
            f"{s['null_balanced_mae']} |"
        )
    lines += [
        "",
        "## Final table (combine_max applied)",
        "",
        "| defect | tp | fp | fn | MAE all | MAE matched |",
        "|---|---|---|---|---|---|",
    ]
    for defect, s in evaluation["per_defect"].items():
        lines.append(
            f"| {defect} | {s['tp']} | {s['fp']} | {s['fn']} | {s['mae_all']} | "
            f"{s['mae_matched']} |"
        )
    lines += ["", "## Cases with errors", ""]
    for row in evaluation["rows"]:
        wrong = {
            d: (row["labels"].get(d, 0), p)
            for d, p in row["predicted"].items()
            if p != row["labels"].get(d, 0)
        }
        if wrong:
            text = ", ".join(f"{d} label {lab} → {pred}" for d, (lab, pred) in wrong.items())
            lines.append(f"- {row['stem']}: {text}")
    if probes:
        lines += [
            "",
            f"## Mild-cast probe (CCT {', '.join(map(str, PROBE_CCT))} K, not fitted)",
            "",
            "| case | label | ln(r/b) → warm | ln(b/r) → cool |",
            "|---|---|---|---|",
        ]
        for row in probes:
            label = ", ".join(f"{d} {s}" for d, s in row["labels"].items() if s)
            cells = [
                "–" if row[d][0] is None else f"{row[d][0]:+.3f} → {row[d][1]}"
                for d in ("color_cast_warm", "color_cast_cool")
            ]
            lines.append(f"| {row['stem']} | {label} | {cells[0]} | {cells[1]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


YAML_HEADER = """\
# Bảng hiệu chỉnh severity (Analysis v2 P2, hợp đồng C4). SINH TỰ ĐỘNG, không sửa tay:
#   python scripts/calibrate_severity.py
# severity = số ngưỡng (tăng dần) mà điểm đạt tới; điểm tính bằng hàm `score` trong
# src/analyzer_evaluator/measurement/calibration.py. thresholds rỗng = không đo được → (0, 0.0).
"""


def write_calibration(path: Path, calibration: Dict[str, Any]) -> None:
    """Ghi calibration.yaml (inf ghi thành .inf)."""
    text = yaml.safe_dump(calibration, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(YAML_HEADER + text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="Đo lại toàn bộ, bỏ cache")
    parser.add_argument("--dry-run", action="store_true", help="Không ghi calibration.yaml")
    parser.add_argument("--out", type=Path, help="Thư mục báo cáo")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    all_cases = load_cases()
    reports = load_features(all_cases, args.refresh)
    cases = [c for c in all_cases if not is_probe(c)]
    calibration, stats = calibrate(cases, reports)
    evaluation = evaluate(cases, reports, calibration)
    probes = probe_rows([c for c in all_cases if is_probe(c)], reports, calibration)

    out_dir = args.out or RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "calibration": calibration,
                "stats": stats,
                "evaluation": evaluation,
                "probes": probes,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", calibration, stats, evaluation, probes)
    if not args.dry_run:
        write_calibration(calib.CALIBRATION_PATH, calibration)
        logger.info("Wrote %s", calib.CALIBRATION_PATH)
    logger.info("MAE on matched pairs: %s", evaluation["mae_matched"])
    logger.info("Report written to %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
