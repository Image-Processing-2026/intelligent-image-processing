"""
Chấm điểm chẩn đoán (hàm thuần, không gọi model, không đọc file).

Đầu vào là dict thuần, đọc được từ manifest thật hoặc `params.json` synthetic:
- đáp án (expected defect): {"type", "region", "min_severity", "severity"}; `severity` là mức
  mong đợi, `min_severity` là mức tối thiểu để tính là đã nhận ra. Nếu thiếu `min_severity`
  thì dùng 1; nếu thiếu `severity` thì dùng `min_severity`.
- dự đoán (predicted defect): {"type", "region", "severity", "origin"}.

Chỉ số (theo docs/analysis_v2/contracts.md §C5):
- precision / recall theo từng loại defect (ghép một-đối-một theo loại và severity);
- độ đúng vùng trên các cặp đã ghép (vùng không là điều kiện ghép);
- FP rate trên ảnh sạch (không có đáp án defect): tỉ lệ ảnh có ít nhất một defect báo sai;
- MAE severity trên các cặp đã ghép; độ đúng vùng (region) trên các cặp đã ghép;
- preserve recall: tỉ lệ đặc điểm cần giữ mà model nhận ra;
- sai số ước lượng (synthetic): sai số tương đối, sai số góc nguồn sáng theo vòng tròn.

Quy tắc khớp loại tái dùng logic của scripts/benchmark_brain.py (lỗi tối trên chủ thể
chấp nhận cả backlit_subject lẫn underexposed; 'full' khớp mọi vùng; nhóm đồng nghĩa vùng).
"""

from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agent.regions import canonical_region

# Lỗi làm tối chủ thể: backlit_subject và underexposed trên mặt/người được coi là tương đương
DARK_TYPES = frozenset({"underexposed", "backlit_subject"})
SUBJECT_REGIONS = frozenset({"face", "person", "subject", "people", "portrait"})
# Các nhóm tên vùng đồng nghĩa mà VLM hay dùng
REGION_SYNONYMS: Tuple[frozenset, ...] = (
    SUBJECT_REGIONS,
    frozenset({"ground", "field", "grass", "foreground", "land", "floor"}),
    frozenset({"sky", "top", "clouds"}),
    frozenset({"background", "bokeh", "backdrop"}),
)


def region_matches(expected: str, predicted: str) -> bool:
    """Vùng dự đoán có khớp vùng mong đợi không ('full' khớp mọi vùng)."""
    expected, predicted = canonical_region(expected), canonical_region(predicted)
    if expected == predicted or "full" in (expected, predicted):
        return True
    return any(expected in group and predicted in group for group in REGION_SYNONYMS)


def type_matches(expected_type: str, expected_region: str, predicted_type: str) -> bool:
    """Loại lỗi khớp; lỗi tối trên chủ thể chấp nhận cả backlit_subject lẫn underexposed."""
    if predicted_type == expected_type:
        return True
    return {predicted_type, expected_type} <= DARK_TYPES and canonical_region(
        expected_region
    ) in SUBJECT_REGIONS


def min_severity(expected: Dict[str, Any]) -> int:
    """Mức tối thiểu để một đáp án được tính là đã nhận ra."""
    return int(expected.get("min_severity", 1))


def expected_severity(expected: Dict[str, Any]) -> int:
    """Mức severity mong đợi (dùng cho MAE)."""
    return int(expected.get("severity", min_severity(expected)))


def defect_matches(expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
    """Một defect dự đoán nhận ra một đáp án (loại + severity tối thiểu).

    Vùng không nằm trong điều kiện nhận ra: vùng sai được tính riêng qua region_accuracy, để
    một lỗi đúng loại nhưng khoanh sai vùng vẫn hiện trong recall và MAE."""
    return type_matches(expected["type"], expected["region"], predicted["type"]) and int(
        predicted["severity"]
    ) >= min_severity(expected)


def match_defects(
    expected: Sequence[Dict[str, Any]], predicted: Sequence[Dict[str, Any]]
) -> List[Tuple[int, int]]:
    """Ghép một-đối-một các đáp án với dự đoán: trả về các cặp (chỉ số đáp án, chỉ số dự đoán).

    Tham lam theo độ gần severity (|pred - expected|, rồi chỉ số nhỏ hơn) để kết quả tất định.
    Mỗi đáp án và mỗi dự đoán được dùng tối đa một lần, nên một lỗi báo lặp không được tính
    thành nhiều lần nhận ra."""
    candidates = []
    for i, exp in enumerate(expected):
        for j, pred in enumerate(predicted):
            if defect_matches(exp, pred):
                gap = abs(int(pred["severity"]) - expected_severity(exp))
                candidates.append((gap, i, j))
    pairs: List[Tuple[int, int]] = []
    used_expected: Set[int] = set()
    used_predicted: Set[int] = set()
    for _, i, j in sorted(candidates):
        if i in used_expected or j in used_predicted:
            continue
        used_expected.add(i)
        used_predicted.add(j)
        pairs.append((i, j))
    return sorted(pairs)


def score_image(
    expected_defects: Sequence[Dict[str, Any]],
    expected_preserve: Iterable[str],
    predicted_defects: Sequence[Dict[str, Any]],
    predicted_preserve: Iterable[str],
) -> Dict[str, Any]:
    """Chấm một ảnh: các cặp đã ghép, lỗi bỏ sót, lỗi báo sai và đặc điểm giữ được.

    Chỉ dự đoán có severity >= 1 mới được tính là defect (severity 0 là "không đáng kể")."""
    actionable = [d for d in predicted_defects if int(d["severity"]) >= 1]
    pairs = match_defects(expected_defects, actionable)
    matched_expected = {i for i, _ in pairs}
    matched_predicted = {j for _, j in pairs}
    expected_set = set(expected_preserve)
    predicted_set = set(predicted_preserve)
    return {
        "matched": [
            {
                "type": expected_defects[i]["type"],
                "expected_region": expected_defects[i]["region"],
                "predicted_region": actionable[j]["region"],
                "expected_severity": expected_severity(expected_defects[i]),
                "predicted_severity": int(actionable[j]["severity"]),
                "region_ok": region_matches(expected_defects[i]["region"], actionable[j]["region"]),
                "origin": actionable[j].get("origin", "vlm"),
            }
            for i, j in pairs
        ],
        "missed": [
            {"type": e["type"], "region": e["region"]}
            for i, e in enumerate(expected_defects)
            if i not in matched_expected
        ],
        "false": [
            {"type": d["type"], "region": d["region"], "severity": int(d["severity"])}
            for j, d in enumerate(actionable)
            if j not in matched_predicted
        ],
        "preserve_matched": sorted(expected_set & predicted_set),
        "preserve_missed": sorted(expected_set - predicted_set),
        "preserve_expected": len(expected_set),
        "is_clean": not expected_defects,
    }


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    """Tỉ lệ có làm tròn; None khi mẫu số bằng 0 (không tính được)."""
    return round(numerator / denominator, 3) if denominator else None


def aggregate(image_scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Gộp điểm từng ảnh (đầu ra của score_image) thành bảng chỉ số của một cấu hình."""
    per_type: Dict[str, Dict[str, int]] = {}

    def bucket(defect_type: str) -> Dict[str, int]:
        return per_type.setdefault(defect_type, {"tp": 0, "fp": 0, "fn": 0})

    for score in image_scores:
        for m in score["matched"]:
            bucket(m["type"])["tp"] += 1
        for f in score["false"]:
            bucket(f["type"])["fp"] += 1
        for miss in score["missed"]:
            bucket(miss["type"])["fn"] += 1

    by_type = {
        name: {
            **counts,
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
        }
        for name, counts in sorted(per_type.items())
    }
    tp = sum(c["tp"] for c in per_type.values())
    fp = sum(c["fp"] for c in per_type.values())
    fn = sum(c["fn"] for c in per_type.values())

    matched = [m for s in image_scores for m in s["matched"]]
    clean = [s for s in image_scores if s["is_clean"]]
    preserve_expected = sum(s["preserve_expected"] for s in image_scores)
    preserve_hits = sum(len(s["preserve_matched"]) for s in image_scores)
    return {
        "images": len(image_scores),
        "defects_expected": tp + fn,
        "defects_predicted": tp + fp,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "by_type": by_type,
        "clean_images": len(clean),
        "fp_image_rate": _ratio(sum(1 for s in clean if s["false"]), len(clean)),
        "fp_per_clean_image": _ratio(sum(len(s["false"]) for s in clean), len(clean)),
        "severity_mae": _ratio(
            sum(abs(m["predicted_severity"] - m["expected_severity"]) for m in matched),
            len(matched),
        ),
        "region_accuracy": _ratio(sum(m["region_ok"] for m in matched), len(matched)),
        "preserve_expected": preserve_expected,
        "preserve_recall": _ratio(preserve_hits, preserve_expected),
    }


def relative_error(estimate: float, truth: float) -> Optional[float]:
    """Sai số tương đối |ước lượng − thật| / |thật|; None khi thật bằng 0."""
    if truth == 0:
        return None
    return abs(estimate - truth) / abs(truth)


def angle_error_deg(estimate: float, truth: float) -> float:
    """Sai số góc theo vòng tròn (độ), trong [0, 180]: 359° và 1° cách nhau 2°."""
    diff = abs(estimate - truth) % 360.0
    return min(diff, 360.0 - diff)


def estimation_summary(
    pairs: Sequence[Tuple[float, float]], *, tolerance: float, circular: bool = False
) -> Dict[str, Any]:
    """Tóm tắt sai số ước lượng trên các cặp (ước lượng, thật).

    `tolerance` là ngưỡng: tỉ lệ `within_tolerance` là phần ca có sai số <= ngưỡng. Với
    `circular=True` sai số là góc theo vòng tròn (độ), ngược lại là sai số tương đối."""
    if circular:
        errors = [angle_error_deg(est, truth) for est, truth in pairs]
    else:
        errors = [e for e in (relative_error(est, truth) for est, truth in pairs) if e is not None]
    if not errors:
        return {"n": 0, "mae": None, "median": None, "within_tolerance": None}
    return {
        "n": len(errors),
        "mae": round(mean(errors), 4),
        "median": round(median(errors), 4),
        "within_tolerance": round(sum(e <= tolerance for e in errors) / len(errors), 3),
    }
