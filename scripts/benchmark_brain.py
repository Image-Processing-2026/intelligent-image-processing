"""
Chấm module brain (perceive + plan) của một model trên bộ ảnh thật trong
data/real_benchmark_manifest.json.

Mỗi ảnh chạy đúng như vòng 1 của pipeline: analyze_image → perceive → plan_treatment
(validate + preserve guard), rồi so với đáp án mong đợi trong manifest:
- defect_recall: tỉ lệ lỗi mong đợi mà model (origin "vlm") nhận ra;
- false_defects: lỗi model báo (severity >= 1) không có trong đáp án;
- preserve_recall: tỉ lệ đặc điểm cần giữ mà model nhận ra;
- empty_plan_ok: plan rỗng đúng khi ảnh không cần xử lý, và ngược lại;
- gamma_wrong_direction: gamma ngược với chính lỗi mà action đó nhắm tới
  (làm tối vùng thiếu sáng / làm sáng vùng cháy sáng);
- gamma_unexpected: hướng gamma không khớp với lỗi trong đáp án của ảnh.
Ảnh có expected.ambiguous = true vẫn được chạy nhưng không tính vào điểm tổng.
Ảnh mà perceive/plan rơi về luật dự phòng (server lỗi) được chạy lại sau --wait giây; nếu vẫn
fallback thì không chấm, chỉ đếm vào vlm_fallbacks.

Chạy từ thư mục gốc repo (cần GEMINI_API_KEY trong .env):
    python scripts/benchmark_brain.py --model gemma-4-31b-it
    python scripts/benchmark_brain.py --model gemini-3.5-flash-lite --ids backlit_01 noisy_02
    python scripts/benchmark_brain.py --model gemma-4-31b-it --resume data/real/benchmark_results/<folder>
Tiến độ được ghi vào <folder>/progress.log; results.json và report.md được cập nhật sau mỗi ảnh.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import cv2

sys.path.insert(0, os.path.abspath("."))

from src.agent.graph import plan_treatment  # noqa: E402
from src.agent.perception import perceive  # noqa: E402
from src.agent.regions import canonical_region  # noqa: E402
from src.analyzer_evaluator import analyze_image  # noqa: E402

logger = logging.getLogger("benchmark_brain")

MANIFEST = Path("data/real_benchmark_manifest.json")
IMAGE_DIR = Path("data/real/benchmark")
RESULTS_DIR = Path("data/real/benchmark_results")

# Lỗi làm tối chủ thể: backlit_subject và underexposed trên mặt/người được coi là tương đương
DARK_TYPES = {"underexposed", "backlit_subject"}
SUBJECT_REGIONS = {"face", "person", "subject", "people", "portrait"}
# Các nhóm tên vùng đồng nghĩa mà VLM hay dùng
REGION_SYNONYMS = (
    SUBJECT_REGIONS,
    {"ground", "field", "grass", "foreground", "land", "floor"},
    {"sky", "top", "clouds"},
    {"background", "bokeh", "backdrop"},
)
METRICS = ("defect_recall", "preserve_recall", "empty_plan_ok")


def _load_api_key() -> None:
    """Đọc GEMINI_API_KEY từ .env (file lưu có BOM nên load_dotenv() bỏ sót)."""
    if os.getenv("GEMINI_API_KEY") or not Path(".env").exists():
        return
    for line in Path(".env").read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("GEMINI_API_KEY="):
            os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def _region_matches(expected: str, predicted: str) -> bool:
    """Vùng dự đoán có khớp vùng mong đợi không (dễ dãi: 'full' khớp mọi vùng)."""
    expected, predicted = canonical_region(expected), canonical_region(predicted)
    if expected == predicted or "full" in (expected, predicted):
        return True
    return any(expected in group and predicted in group for group in REGION_SYNONYMS)


def _type_matches(expected: Dict[str, Any], defect: Dict[str, Any]) -> bool:
    """Loại lỗi khớp; lỗi tối trên chủ thể chấp nhận cả backlit_subject lẫn underexposed."""
    if defect["type"] == expected["type"]:
        return True
    return {defect["type"], expected["type"]} <= DARK_TYPES and canonical_region(
        expected["region"]
    ) in SUBJECT_REGIONS


def _matches(expected: Dict[str, Any], defect: Dict[str, Any]) -> bool:
    """Một lỗi dự đoán khớp một lỗi mong đợi (loại + vùng + severity tối thiểu)."""
    return (
        _type_matches(expected, defect)
        and _region_matches(expected["region"], defect["region"])
        and defect["severity"] >= expected["min_severity"]
    )


def _gamma_checks(
    actions: List[Dict[str, Any]], expected_defects: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """Kiểm tra hướng gamma (gamma > 1 làm sáng) so với lỗi action nhắm tới và với đáp án."""
    expected_types = {d["type"] for d in expected_defects}
    wrong, unexpected = [], []
    for action in actions:
        if action["operation"] != "gamma_correct":
            continue
        gamma = float(action["parameters"].get("gamma", 1.0))
        label = f"{action.get('target_prompt') or 'full'}:gamma={gamma:g}"
        issue = action.get("detected_issue") or ""
        if (issue in DARK_TYPES and gamma < 1) or (issue == "overexposed" and gamma > 1):
            wrong.append(f"{label} for {issue}")
        if (gamma > 1 and not expected_types & DARK_TYPES) or (
            gamma < 1 and "overexposed" not in expected_types
        ):
            unexpected.append(label)
    return {"gamma_wrong_direction": wrong, "gamma_unexpected": unexpected}


def score_case(entry: Dict[str, Any], diagnosis: Dict[str, Any], plan: Dict[str, Any]) -> Dict:
    """Chấm một ảnh theo đáp án trong manifest."""
    expected = entry["expected"]
    vlm_defects = [
        d for d in diagnosis["defects"] if d.get("origin") == "vlm" and d["severity"] >= 1
    ]
    found = [e for e in expected["defects"] if any(_matches(e, d) for d in vlm_defects)]
    false_defects = [
        f"{d['type']}@{d['region']}(s{d['severity']})"
        for d in vlm_defects
        if not any(_type_matches(e, d) for e in expected["defects"])
    ]
    predicted_preserve = {p["aspect"] for p in diagnosis.get("preserve", [])}
    expected_preserve = set(expected["preserve"])
    actions = plan["actions"]
    return {
        "defect_recall": len(found) / len(expected["defects"]) if expected["defects"] else None,
        "missed_defects": [
            f"{e['type']}@{e['region']}" for e in expected["defects"] if e not in found
        ],
        "false_defects": false_defects,
        "preserve_recall": (
            len(expected_preserve & predicted_preserve) / len(expected_preserve)
            if expected_preserve
            else None
        ),
        "missed_preserve": sorted(expected_preserve - predicted_preserve),
        "empty_plan_ok": (len(actions) == 0) == expected["empty_plan"],
        **_gamma_checks(actions, expected["defects"]),
    }


def run_case(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Chạy perceive + plan (vòng 1, không có intent hay case memory) cho một ảnh."""
    bgr = cv2.imread(str(IMAGE_DIR / entry["file"]))
    if bgr is None:
        raise FileNotFoundError(
            f"Missing image {entry['file']}; run data/download_real_benchmark.py"
        )
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    metrics = analyze_image(image).model_dump()
    started = time.time()
    diagnosis = perceive(image, metrics)
    plan = plan_treatment(image, metrics, diagnosis)
    return {
        "seconds": round(time.time() - started, 1),
        "diagnosis_source": diagnosis.source,
        "plan_source": plan.source,
        "diagnosis": diagnosis.model_dump(mode="json", exclude={"region_metrics"}),
        "plan": {
            "reasoning": plan.reasoning,
            "actions": [
                {
                    "operation": a.operation,
                    "target_prompt": a.target_prompt,
                    "detected_issue": a.detected_issue,
                    "parameters": a.parameters,
                }
                for a in plan.actions
            ],
        },
    }


def _fell_back(case: Dict[str, Any]) -> bool:
    """Perceive hoặc plan đã rơi về luật dự phòng → kết quả không phải của model, không chấm."""
    return "vlm_fallback" in (case.get("diagnosis_source"), case.get("plan_source"))


def _avg(values: List[Optional[float]]) -> Optional[float]:
    """Trung bình bỏ qua None."""
    values = [float(v) for v in values if v is not None]
    return round(mean(values), 3) if values else None


def summarize(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Điểm tổng (bỏ ảnh ambiguous và ảnh lỗi chạy) và điểm theo nhóm."""
    scored = [c for c in cases if "score" in c and not c["ambiguous"]]

    def block(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            **{m: _avg([c["score"][m] for c in items]) for m in METRICS},
            "false_defects": sum(len(c["score"]["false_defects"]) for c in items),
            "gamma_wrong_direction": sum(len(c["score"]["gamma_wrong_direction"]) for c in items),
            "gamma_unexpected": sum(len(c["score"]["gamma_unexpected"]) for c in items),
            "images": len(items),
        }

    groups = sorted({c["category"] for c in scored})
    return {
        "overall": block(scored),
        "by_category": {g: block([c for c in scored if c["category"] == g]) for g in groups},
        "vlm_fallbacks": sum(1 for c in cases if _fell_back(c)),
        "errors": sum(1 for c in cases if "error" in c),
        "mean_seconds": _avg([c.get("seconds") for c in cases]),
    }


def _fmt(value: Any) -> str:
    """Định dạng một ô trong bảng markdown."""
    if value is None:
        return "–"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def write_report(path: Path, model: str, summary: Dict[str, Any], cases: List[Dict]) -> None:
    """Ghi báo cáo markdown: bảng tổng, bảng theo nhóm, chi tiết từng ảnh."""
    cols = [*METRICS, "false_defects", "gamma_wrong_direction", "gamma_unexpected", "images"]
    lines = [
        f"# Brain benchmark: `{model}`",
        "",
        f"VLM fallbacks: {summary['vlm_fallbacks']} · errors: {summary['errors']} · "
        f"mean seconds/image: {summary['mean_seconds']}",
        "",
        "| group | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    rows = [("**overall**", summary["overall"]), *summary["by_category"].items()]
    for name, block in rows:
        lines.append(f"| {name} | " + " | ".join(_fmt(block[c]) for c in cols) + " |")
    lines += ["", "## Per image", ""]
    for c in cases:
        flag = " (ambiguous, not scored)" if c["ambiguous"] else ""
        lines.append(f"### {c['id']}{flag}")
        if "error" in c:
            lines += [f"- error: {c['error']}", ""]
            continue
        if "score" not in c:
            lines += [
                f"- not scored: fell back to rules (perceive={c['diagnosis_source']}, "
                f"plan={c['plan_source']})",
                "",
            ]
            continue
        s = c["score"]
        actions = ", ".join(
            f"{a['operation']}@{a['target_prompt'] or 'full'}{a['parameters']}"
            for a in c["plan"]["actions"]
        )
        lines += [
            f"- sources: perceive={c['diagnosis_source']}, plan={c['plan_source']}",
            f"- summary: {c['diagnosis'].get('summary', '')}",
            f"- defect_recall={_fmt(s['defect_recall'])} missed={s['missed_defects']} "
            f"false={s['false_defects']}",
            f"- preserve_recall={_fmt(s['preserve_recall'])} missed={s['missed_preserve']}",
            f"- empty_plan_ok={s['empty_plan_ok']} actions: {actions or '(none)'}",
            f"- gamma wrong={s['gamma_wrong_direction']} unexpected={s['gamma_unexpected']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save(out_dir: Path, model: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ghi results.json và report.md cho các ảnh đã chạy; trả về phần tổng hợp."""
    summary = summarize(cases)
    (out_dir / "results.json").write_text(
        json.dumps(
            {"model": model, "summary": summary, "cases": cases}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", model, summary, cases)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Gemini API model id")
    parser.add_argument("--ids", nargs="*", help="Only run these manifest ids")
    parser.add_argument(
        "--case-retries", type=int, default=2, help="Reruns of an image that fell back to rules"
    )
    parser.add_argument("--wait", type=int, default=30, help="Seconds to wait before a rerun")
    parser.add_argument(
        "--resume", type=Path, help="Existing result folder to continue (skips scored images)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _load_api_key()
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set; the benchmark would only measure the rule-based path.")
    os.environ["GEMINI_MODEL"] = args.model

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]
    if args.ids:
        entries = [e for e in entries if e["id"] in set(args.ids)]
    out_dir = args.resume or RESULTS_DIR / f"{args.model}_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Log tiến độ ra file để xem được giữa chừng (stdout có thể bị pipe giữ lại)
    file_handler = logging.FileHandler(out_dir / "progress.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Chạy tiếp: giữ ảnh đã chấm xong, chạy lại ảnh bị fallback hoặc chưa chạy
    done: Dict[str, Dict[str, Any]] = {}
    if args.resume and (out_dir / "results.json").exists():
        previous = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        done = {c["id"]: c for c in previous["cases"] if "score" in c}
        logger.info("Resuming %s: %d image(s) already scored", out_dir, len(done))

    cases: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        if entry["id"] in done:
            cases.append(done[entry["id"]])
            continue
        case: Dict[str, Any] = {
            "id": entry["id"],
            "category": entry["category"],
            "ambiguous": entry["expected"].get("ambiguous", False),
        }
        for attempt in range(1, args.case_retries + 2):
            try:
                case.update(run_case(entry))
            except Exception as exc:
                logger.error("Case %s failed: %s", entry["id"], exc)
                case["error"] = f"{type(exc).__name__}: {exc}"
                break
            if not _fell_back(case) or attempt > args.case_retries:
                break
            # Server lỗi kéo dài hơn retry trong call_gemini_json → chờ lâu rồi chạy lại cả ảnh
            logger.warning("Case %s fell back to rules; rerunning in %ds.", entry["id"], args.wait)
            time.sleep(args.wait)
        if "error" not in case and not _fell_back(case):
            case["score"] = score_case(entry, case["diagnosis"], case["plan"])
        cases.append(case)
        logger.info(
            "[%d/%d] %s perceive=%s plan=%s",
            index,
            len(entries),
            entry["id"],
            case.get("diagnosis_source"),
            case.get("plan_source"),
        )
        # Ghi sau từng ảnh để bị ngắt giữa chừng vẫn còn kết quả và chạy tiếp được
        summary = _save(out_dir, args.model, cases)

    summary = _save(out_dir, args.model, cases)
    logger.info("Overall: %s", summary["overall"])
    logger.info("Report written to %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
