"""
Benchmark chẩn đoán theo hợp đồng C5 (docs/analysis_v2/contracts.md).

Mỗi cấu hình là một hàm nhận (ảnh RGB uint8, thông tin ca) và trả về DiagnosisReport:
- baseline: luồng hiện tại, analyze_image → perceive. Từ lane fusion (P6) perceive đã gồm hợp
  nhất tất định nên baseline = fused; số của luồng v1 nằm trong docs/analysis_v2/baseline.md;
- vlm_only: perceive v2 tắt hợp nhất (FUSION_ENABLED = False): chỉ lời VLM sau preserve;
- vlm_only_n1: vlm_only với một lần quan sát (tách hiệu ứng của self-consistency);
- vlm_only_frame1024: vlm_only nhưng ảnh toàn khung thu về cạnh dài 1024 px, crop vẫn 100 %
  (câu hỏi mở plan §7: có cần ảnh toàn khung full-res khi đã có crop);
- measure_only: số đo toàn ảnh (P1) → severity theo calibration (C4); cần lane calib đã merge;
- fused: perceive v2 + hợp nhất P6 (fusion.fuse: số đo × VLM × preserve, verify câu hỏi đóng);
- fused_noverify: fused không gọi verify (đo tác dụng của bước kiểm chứng);
- fused_n3 / fused_n3_weak: fused với PERCEIVE_SAMPLES = 3, không/có dùng giả thuyết bị loại
  (dưới ngưỡng đồng thuận) làm bằng chứng yếu (brief fusion mục 12, kết luận trên hold-out).

Ba bộ dữ liệu:
- real: data/real_benchmark_manifest.json (ảnh Wikimedia Commons, đáp án do con người/agent đặt);
- holdout: data/holdout_manifest.json, cùng định dạng với real (ảnh ở `image_dir` của manifest,
  mặc định data/real/holdout). Chỉ để chấm kết quả cuối, không chỉnh gì trên bộ này;
- synthetic: data/synthetic/calibration/*.params.json (sinh bởi scripts/build_calibration_set.py).

Chấm điểm nằm trong scripts/benchmark_scoring.py. Mỗi lần chạy ghi results.json và report.md vào
data/real/benchmark_results/<timestamp>-<config>/ (thư mục này bị .gitignore) và in tổng kết.
Số lần gọi Gemini được đếm và ghi vào báo cáo.

Chạy từ thư mục gốc repo:
    python scripts/benchmark_analysis.py --set synthetic --config baseline
    python scripts/benchmark_analysis.py --set real --config baseline --limit 5
    python scripts/benchmark_analysis.py --set real --config vlm_only --model gemini-3.5-flash-lite
    python scripts/benchmark_analysis.py --set synthetic --config baseline --offline
    python scripts/benchmark_analysis.py --set real --config fused --vlm-cache data/real/vlm_cache
Set real cần GEMINI_API_KEY trong .env; nếu thiếu, script dừng để không đo nhầm luật dự phòng.
--offline bỏ qua Gemini (chẩn đoán theo luật, source=rule_based) để chạy không cần mạng.
--vlm-cache DIR lưu từng phản hồi Gemini theo băm nội dung yêu cầu (prompt, schema, ảnh,
temperature, lần gọi thứ mấy): chạy lại cùng yêu cầu không gọi Gemini, nên chỉnh phần hợp nhất
tất định không tốn lời gọi và không bị nhiễu do VLM trả khác nhau giữa các lần chạy.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from scripts.benchmark_scoring import aggregate, score_image  # noqa: E402
from src.agent import fusion, perception  # noqa: E402
from src.agent.perception import perceive  # noqa: E402
from src.agent.state import DEFECT_TYPES, Defect, DiagnosisReport  # noqa: E402
from src.agent.vlm_diagnostician import DEFAULT_GEMINI_MODEL  # noqa: E402
from src.analyzer_evaluator import analyze_image  # noqa: E402
from src.analyzer_evaluator.measurement import measure_image  # noqa: E402

logger = logging.getLogger("benchmark_analysis")

MANIFEST = Path("data/real_benchmark_manifest.json")
REAL_IMAGE_DIR = Path("data/real/benchmark")
HOLDOUT_MANIFEST = Path("data/holdout_manifest.json")
HOLDOUT_IMAGE_DIR = Path("data/real/holdout")
SYNTHETIC_DIR = Path("data/synthetic/calibration")
RESULTS_DIR = Path("data/real/benchmark_results")
# Nhóm ảnh không có lỗi cần sửa: dùng để đo FP rate
CLEAN_CATEGORIES = frozenset({"already_good", "intentional_style"})


@dataclass(frozen=True)
class CaseInfo:
    """Một ca đánh giá: ảnh và đáp án mong đợi (theo format defect của benchmark_scoring)."""

    case_id: str
    dataset: str  # "real" | "synthetic"
    category: str
    image_path: Path
    expected_defects: List[Dict[str, Any]] = field(default_factory=list)
    expected_preserve: List[str] = field(default_factory=list)
    ambiguous: bool = False
    params: Optional[Dict[str, Any]] = None


ConfigFn = Callable[[np.ndarray, CaseInfo], DiagnosisReport]


def _baseline(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """Luồng hiện tại: đo toàn cục rồi perceive (VLM hoặc luật dự phòng, kèm lỗi đo được)."""
    metrics = analyze_image(image).model_dump()
    return perceive(image, metrics)


class _patched:
    """Đặt tạm thuộc tính module (và biến môi trường) trong một lần chạy config."""

    def __init__(self, module: Any, env: Optional[Dict[str, str]] = None, **values: Any) -> None:
        self.module, self.values, self.env = module, values, env or {}
        self.saved: Dict[str, Any] = {}
        self.saved_env: Dict[str, Optional[str]] = {}

    def __enter__(self) -> None:
        for name, value in self.values.items():
            self.saved[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        for name, value in self.env.items():
            self.saved_env[name] = os.environ.get(name)
            os.environ[name] = value

    def __exit__(self, *exc: Any) -> None:
        for name, value in self.saved.items():
            setattr(self.module, name, value)
        for name, value in self.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _vlm_only(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """Perceive v2 tắt hợp nhất: chỉ còn lời VLM (sau khi bỏ lỗi mâu thuẫn preserve)."""
    with _patched(perception, FUSION_ENABLED=False):
        report = _baseline(image, case)
    report.defects = [d for d in report.defects if d.origin != "measured"]
    return report


def _vlm_only_n1(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """vlm_only với PERCEIVE_SAMPLES = 1 (không self-consistency)."""
    previous = perception.PERCEIVE_SAMPLES
    perception.PERCEIVE_SAMPLES = 1
    try:
        return _vlm_only(image, case)
    finally:
        perception.PERCEIVE_SAMPLES = previous


def _vlm_only_n3(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """vlm_only với PERCEIVE_SAMPLES = 3 (self-consistency, D3) để so với mặc định N = 1."""
    previous = perception.PERCEIVE_SAMPLES
    perception.PERCEIVE_SAMPLES = 3
    try:
        return _vlm_only(image, case)
    finally:
        perception.PERCEIVE_SAMPLES = previous


def _vlm_only_frame1024(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """vlm_only với ảnh toàn khung gửi VLM thu về cạnh dài 1024 px (crop vẫn 100 %)."""
    previous = perception.FULL_FRAME_MAX_SIDE
    perception.FULL_FRAME_MAX_SIDE = 1024
    try:
        return _vlm_only(image, case)
    finally:
        perception.FULL_FRAME_MAX_SIDE = previous


def _measure_only(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """Chỉ số đo toàn ảnh (P1) → severity theo bảng hiệu chỉnh C4; không gọi VLM.

    Chưa có `measurement.calibration` (lane calib) thì báo NotImplementedError. Sai số ước lượng
    thô (σ nhiễu, σ mờ, góc) xem scripts/benchmark_measurement.py.
    """
    try:
        from src.analyzer_evaluator.measurement.calibration import severity_from_measurement
    except ImportError as exc:
        raise NotImplementedError(
            "Config 'measure_only' needs src/analyzer_evaluator/measurement/calibration.py "
            "(calib lane, contracts.md §C4); use scripts/benchmark_measurement.py for raw errors."
        ) from exc
    report = measure_image(image).report
    defects = []
    for defect_type in DEFECT_TYPES:
        severity, confidence = severity_from_measurement(defect_type, report)
        if severity >= 1:
            defects.append(
                Defect(
                    type=defect_type,
                    region="full",
                    severity=severity,
                    evidence=f"measured, confidence {confidence:.2f}",
                    origin="measured",
                )
            )
    return DiagnosisReport(defects=defects, source="rule_based")


def _fused(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """Perceive v2 + hợp nhất P6 (mặc định của perceive): số đo × VLM × preserve × verify."""
    with _patched(perception, FUSION_ENABLED=True, VERIFY_ENABLED=True):
        return _baseline(image, case)


def _fused_noverify(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """fused không gọi verify: giả thuyết cần kiểm chứng được quyết định như khi verify lỗi."""
    with _patched(perception, FUSION_ENABLED=True, VERIFY_ENABLED=False):
        return _baseline(image, case)


def _fused_n3(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """fused với 3 lần quan sát (self-consistency), không dùng giả thuyết bị loại."""
    with _patched(perception, PERCEIVE_SAMPLES=3), _patched(fusion, USE_WEAK_REJECTED=False):
        return _fused(image, case)


def _fused_n3_weak(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
    """fused_n3 + giả thuyết bị loại (dưới ngưỡng đồng thuận) làm bằng chứng yếu."""
    with _patched(perception, PERCEIVE_SAMPLES=3), _patched(fusion, USE_WEAK_REJECTED=True):
        return _fused(image, case)


def _not_implemented(name: str, owner: str) -> ConfigFn:
    """Cấu hình chưa có: báo rõ lane nào sẽ điền vào."""

    def run(image: np.ndarray, case: CaseInfo) -> DiagnosisReport:
        raise NotImplementedError(
            f"Config '{name}' is not implemented yet; it waits for {owner} "
            "(see docs/analysis_v2/contracts.md §C5)."
        )

    return run


CONFIGS: Dict[str, ConfigFn] = {
    "baseline": _baseline,
    "vlm_only": _vlm_only,
    "vlm_only_n1": _vlm_only_n1,
    "vlm_only_n3": _vlm_only_n3,
    "vlm_only_frame1024": _vlm_only_frame1024,
    "measure_only": _measure_only,
    "fused": _fused,
    "fused_noverify": _fused_noverify,
    "fused_n3": _fused_n3,
    "fused_n3_weak": _fused_n3_weak,
}


def _load_api_key() -> None:
    """Đọc GEMINI_API_KEY từ .env (file lưu có BOM nên load_dotenv() bỏ sót)."""
    if os.getenv("GEMINI_API_KEY") or not Path(".env").exists():
        return
    for line in Path(".env").read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("GEMINI_API_KEY="):
            os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def load_real_cases(
    manifest: Path = MANIFEST, image_dir: Path = REAL_IMAGE_DIR, dataset: str = "real"
) -> List[CaseInfo]:
    """Các ca của bộ thật, đọc từ manifest (ảnh phải đã được tải bằng download_real_benchmark.py)."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    image_dir = Path(data.get("image_dir") or image_dir)
    cases = []
    for entry in data["images"]:
        expected = entry["expected"]
        cases.append(
            CaseInfo(
                case_id=entry["id"],
                dataset=dataset,
                category=entry["category"],
                image_path=image_dir / entry["file"],
                expected_defects=[
                    {
                        "type": d["type"],
                        "region": d["region"],
                        "min_severity": d.get("min_severity", 1),
                        "severity": d.get("severity", d.get("min_severity", 1)),
                    }
                    for d in expected["defects"]
                ],
                expected_preserve=list(expected["preserve"]),
                ambiguous=bool(expected.get("ambiguous", False)),
            )
        )
    return cases


def load_synthetic_cases() -> List[CaseInfo]:
    """Các ca synthetic: mỗi params.json cạnh một ảnh PNG."""
    cases = []
    for params_path in sorted(SYNTHETIC_DIR.glob("*.params.json")):
        params = json.loads(params_path.read_text(encoding="utf-8"))
        stem = params_path.name.removesuffix(".params.json")
        cases.append(
            CaseInfo(
                case_id=stem,
                dataset="synthetic",
                category="+".join(d["type"] for d in params["degradations"]) or "clean",
                image_path=params_path.with_name(stem + ".png"),
                expected_defects=[{**d, "min_severity": 1} for d in params["expected"]["defects"]],
                params=params,
            )
        )
    return cases


def _content_digest(item: Any) -> bytes:
    """Băm một phần nội dung gửi Gemini (chuỗi hoặc ảnh PIL)."""
    if isinstance(item, str):
        return item.encode("utf-8")
    if hasattr(item, "tobytes") and hasattr(item, "size"):
        return f"{item.mode}{item.size}".encode() + hashlib.blake2b(item.tobytes()).digest()
    return repr(item).encode("utf-8")


class VlmCache:
    """Cache phản hồi Gemini trên đĩa theo băm nội dung yêu cầu (kèm lần gọi thứ mấy)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._seen: Dict[str, int] = {}
        self._lock = threading.Lock()

    def key(self, system_prompt: str, contents: List[Any], schema: Any, temperature: Any) -> str:
        digest = hashlib.blake2b(digest_size=20)
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        digest.update(f"{model}|{temperature}|".encode())
        digest.update(system_prompt.encode("utf-8"))
        digest.update(json.dumps(schema, sort_keys=True, default=str).encode("utf-8"))
        for item in contents:
            digest.update(_content_digest(item))
        base = digest.hexdigest()
        # Cùng yêu cầu gọi nhiều lần (self-consistency) → mỗi lần một mục cache riêng
        with self._lock:
            index = self._seen.get(base, 0)
            self._seen[base] = index + 1
        return f"{base}-{index}"

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()

    def get(self, key: str) -> Any:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        (self.directory / f"{key}.json").write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )


VLM_CACHE: Optional[VlmCache] = None


class GeminiCallCounter:
    """
    Đếm số lần perceive/verify gọi Gemini thật (kể cả lần chạy lại vì fallback); với VLM_CACHE,
    phản hồi đã lưu được trả lại không gọi mạng và được đếm riêng (cache_hits).
    """

    def __init__(self) -> None:
        self.count = 0
        self.cache_hits = 0
        self._original = perception.call_gemini_json
        self._lock = threading.Lock()

    def __enter__(self) -> "GeminiCallCounter":
        cache = VLM_CACHE
        if cache is not None:
            cache.reset()

        def counted(
            api_key: str,
            system_prompt: str,
            contents: List[Any],
            schema: Any,
            temperature: Any = None,
        ) -> Any:
            key = cache.key(system_prompt, contents, schema, temperature) if cache else None
            if key is not None:
                cached = cache.get(key)
                if cached is not None:
                    with self._lock:
                        self.cache_hits += 1
                    return cached
            with self._lock:
                self.count += 1
            result = self._original(
                api_key, system_prompt, contents, schema, temperature=temperature
            )
            if key is not None:
                cache.put(key, result)
            return result

        perception.call_gemini_json = counted
        return self

    def __exit__(self, *exc: Any) -> None:
        perception.call_gemini_json = self._original


def _fell_back(report: DiagnosisReport) -> bool:
    """Perceive rơi về luật dự phòng do lỗi server → kết quả không phải của model."""
    return report.source == "vlm_fallback"


def run_case(
    config_fn: ConfigFn, case: CaseInfo, *, case_retries: int, wait: int
) -> Dict[str, Any]:
    """Chạy một ca và chấm điểm; ảnh bị fallback được chạy lại tối đa case_retries lần."""
    bgr = cv2.imread(str(case.image_path))
    if bgr is None:
        raise FileNotFoundError(f"Missing image {case.image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    record: Dict[str, Any] = {
        "id": case.case_id,
        "dataset": case.dataset,
        "category": case.category,
        "ambiguous": case.ambiguous,
    }
    with GeminiCallCounter() as counter:
        for attempt in range(1, case_retries + 2):
            started = time.time()
            report = config_fn(image, case)
            record["seconds"] = round(time.time() - started, 1)
            if not _fell_back(report) or attempt > case_retries:
                break
            logger.warning("Case %s fell back to rules; rerunning in %ds.", case.case_id, wait)
            time.sleep(wait)
    record["gemini_calls"] = counter.count
    record["gemini_cache_hits"] = counter.cache_hits
    record["source"] = report.source
    predicted = [
        {
            "type": d.type,
            "region": d.region,
            "severity": d.severity,
            "origin": d.origin,
            "confidence": d.confidence,
        }
        for d in report.defects
    ]
    record["predicted_defects"] = predicted
    # Giả thuyết bị loại do không đủ đồng thuận (self-consistency): không chấm, chỉ để xem
    record["rejected_defects"] = [
        {"type": d.type, "region": d.region, "severity": d.severity, "confidence": d.confidence}
        for d in report.rejected
    ]
    record["predicted_preserve"] = [p.aspect for p in report.preserve]
    record["ask_user"] = [
        {"type": d.type, "region": d.region, "severity": d.severity} for d in report.ask_user
    ]
    record["fusion_notes"] = list(report.fusion_notes)
    record["summary"] = report.summary
    if not _fell_back(report):
        record["score"] = score_image(
            case.expected_defects,
            case.expected_preserve,
            predicted,
            record["predicted_preserve"],
        )
    return record


def summarize(records: Sequence[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    """Tổng hợp chỉ số: bỏ ca ambiguous và ca fallback; FP rate tính riêng trên nhóm sạch."""
    scored = [r for r in records if "score" in r and not r["ambiguous"]]
    overall = aggregate([r["score"] for r in scored])
    # Ảnh sạch: nhóm đã biết (real) hoặc biến thể không có đáp án defect (synthetic)
    clean = [
        r
        for r in scored
        if r["score"]["is_clean"] and (dataset == "synthetic" or r["category"] in CLEAN_CATEGORIES)
    ]
    groups = sorted({r["category"] for r in scored})
    return {
        "overall": overall,
        "clean_fp_image_rate": aggregate([r["score"] for r in clean])["fp_image_rate"],
        "clean_images": len(clean),
        "by_category": {
            g: aggregate([r["score"] for r in scored if r["category"] == g]) for g in groups
        },
        "gemini_calls": sum(r.get("gemini_calls", 0) for r in records),
        "gemini_cache_hits": sum(r.get("gemini_cache_hits", 0) for r in records),
        "fallbacks": sum(1 for r in records if r.get("source") == "vlm_fallback"),
        "errors": sum(1 for r in records if "error" in r),
        "mean_seconds": round(
            sum(r.get("seconds", 0.0) for r in records) / max(1, len(records)), 1
        ),
    }


def _fmt(value: Any) -> str:
    """Định dạng một ô bảng markdown."""
    if value is None:
        return "–"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def write_report(
    path: Path, config: str, dataset: str, model: str, summary: Dict, records: List
) -> None:
    """Báo cáo markdown: chỉ số tổng, theo loại defect, theo nhóm, và chi tiết từng ca."""
    overall = summary["overall"]
    lines = [
        f"# Diagnosis benchmark: `{config}` on `{dataset}`",
        "",
        f"Model: `{model}` · Gemini calls: {summary['gemini_calls']} "
        f"(cache hits: {summary.get('gemini_cache_hits', 0)}) · "
        f"fallbacks: {summary['fallbacks']} · errors: {summary['errors']} · "
        f"mean seconds/case: {summary['mean_seconds']}",
        "",
        "| precision | recall | FP rate (clean) | severity MAE | region accuracy | preserve recall |",
        "|---|---|---|---|---|---|",
        f"| {_fmt(overall['precision'])} | {_fmt(overall['recall'])} | "
        f"{_fmt(summary['clean_fp_image_rate'])} (n={summary['clean_images']}) | "
        f"{_fmt(overall['severity_mae'])} | {_fmt(overall['region_accuracy'])} | "
        f"{_fmt(overall['preserve_recall'])} (n={overall['preserve_expected']}) |",
        "",
        "Đo hết các ca đã chấm (không tính ca ambiguous và ca rơi về luật dự phòng).",
        "",
        "## Theo loại defect",
        "",
        "| type | tp | fp | fn | precision | recall |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in overall["by_type"].items():
        lines.append(
            f"| {name} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{_fmt(row['precision'])} | {_fmt(row['recall'])} |"
        )
    lines += [
        "",
        "## Theo nhóm",
        "",
        "| group | images | precision | recall | FP image rate |",
        "|---|---|---|---|---|",
    ]
    for name, block in summary["by_category"].items():
        lines.append(
            f"| {name} | {block['images']} | {_fmt(block['precision'])} | "
            f"{_fmt(block['recall'])} | {_fmt(block['fp_image_rate'])} |"
        )
    lines += ["", "## Chi tiết từng ca", ""]
    for record in records:
        flag = " (ambiguous, not scored)" if record["ambiguous"] else ""
        lines.append(f"### {record['id']}{flag}")
        if "error" in record:
            lines += [f"- error: {record['error']}", ""]
            continue
        if "score" not in record:
            lines += [f"- not scored: fell back to rules (source={record['source']})", ""]
            continue
        s = record["score"]
        predicted = ", ".join(
            f"{d['type']}@{d['region']}(s{d['severity']},{d['origin']})"
            for d in record["predicted_defects"]
        )
        lines += [
            f"- source: {record['source']} · gemini calls: {record['gemini_calls']} · "
            f"{record['seconds']}s",
            f"- predicted: {predicted or '(none)'}",
            f"- matched: {[m['type'] for m in s['matched']]} · missed: {s['missed']} · "
            f"false: {s['false']}",
            f"- preserve matched: {s['preserve_matched']} · missed: {s['preserve_missed']}",
        ]
        lines += [f"  - fusion: {note}" for note in record.get("fusion_notes", [])]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    config: str,
    dataset: str,
    *,
    limit: Optional[int],
    ids: Optional[Sequence[str]],
    out_dir: Path,
    model: str,
    case_retries: int,
    wait: int,
) -> Dict[str, Any]:
    """Chạy một cấu hình trên một bộ, ghi results.json và report.md, trả về phần tổng hợp."""
    if dataset == "real":
        cases = load_real_cases()
    elif dataset == "holdout":
        cases = load_real_cases(HOLDOUT_MANIFEST, HOLDOUT_IMAGE_DIR, dataset="holdout")
    else:
        cases = load_synthetic_cases()
    if ids:
        cases = [c for c in cases if c.case_id in set(ids)]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise SystemExit(f"No cases found for set '{dataset}'.")

    config_fn = CONFIGS[config]
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        try:
            record = run_case(config_fn, case, case_retries=case_retries, wait=wait)
        except NotImplementedError:
            raise
        except Exception as exc:  # một ca lỗi không làm hỏng cả bộ
            logger.error("Case %s failed: %s", case.case_id, exc)
            record = {
                "id": case.case_id,
                "dataset": dataset,
                "category": case.category,
                "ambiguous": case.ambiguous,
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        logger.info("[%d/%d] %s source=%s", index, len(cases), case.case_id, record.get("source"))
        summary = summarize(records, dataset)
        (out_dir / "results.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "dataset": dataset,
                    "model": model,
                    "summary": summary,
                    "cases": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_report(out_dir / "report.md", config, dataset, model, summary, records)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--set", dest="dataset", required=True, choices=["real", "holdout", "synthetic"]
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--limit", type=int, help="Chỉ chạy N ca đầu tiên (kiểm thử nhanh)")
    parser.add_argument("--ids", nargs="*", help="Chỉ chạy các id này")
    parser.add_argument(
        "--out", type=Path, help="Thư mục kết quả (mặc định: tự đặt theo thời gian)"
    )
    parser.add_argument("--model", help="Model Gemini (ghi đè GEMINI_MODEL)")
    parser.add_argument(
        "--offline", action="store_true", help="Không gọi Gemini: chẩn đoán theo luật dự phòng"
    )
    parser.add_argument(
        "--vlm-cache", type=Path, help="Thư mục cache phản hồi Gemini (chạy lại không gọi mạng)"
    )
    parser.add_argument("--case-retries", type=int, default=2)
    parser.add_argument("--wait", type=int, default=30, help="Giây chờ trước khi chạy lại ca lỗi")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.offline:
        os.environ.pop("GEMINI_API_KEY", None)
    else:
        _load_api_key()
    global VLM_CACHE
    VLM_CACHE = VlmCache(args.vlm_cache) if args.vlm_cache else None
    if (
        args.dataset in ("real", "holdout")
        and args.config != "measure_only"
        and not os.getenv("GEMINI_API_KEY")
    ):
        raise SystemExit(
            "GEMINI_API_KEY is not set; the real set would only measure rule-based output."
        )
    if args.model:
        os.environ["GEMINI_MODEL"] = args.model
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    out_dir = (
        args.out or RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.config}-{args.dataset}"
    )
    try:
        summary = run_benchmark(
            args.config,
            args.dataset,
            limit=args.limit,
            ids=args.ids,
            out_dir=out_dir,
            model=model,
            case_retries=args.case_retries,
            wait=args.wait,
        )
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info("Overall: %s", summary["overall"])
    logger.info("Report written to %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
