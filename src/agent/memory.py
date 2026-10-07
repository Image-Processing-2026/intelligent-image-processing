"""
Bộ nhớ ca bệnh (Phase 5, tầng C của Knowledge Base).

Mỗi lần xử lý được ghi thành một "ca": chữ ký số liệu của ảnh gốc, loại cảnh, lỗi, ý định,
phác đồ, điểm, cùng tín hiệu từ người dùng (phiên bản đã chọn, góp ý). KHÔNG lưu ảnh.
Bộ nhớ được dùng để:
1. Truy xuất ca tương tự đưa vào prompt Plan ("KINH NGHIỆM"), để agent học từ góp ý cũ.
2. Suy ra phong cách người dùng hay chọn, làm phiên bản đề xuất mặc định.

Bộ nhớ chỉ hoạt động khi được cấu hình (configure_case_memory); mặc định tắt, nên test và
pipeline không cấu hình sẽ không ghi gì. Lưu trữ bằng SQLite (thư viện chuẩn), mở lười
ở lần dùng đầu tiên.
"""

import logging
import math
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from .state import DiagnosisReport, DoctorState

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path("data/memory/cases.sqlite")
MEMORY_ENV = "CASE_MEMORY_PATH"
DEFAULT_USER = "default"

MAX_SIMILAR = 3
# Khoảng cách lớn hơn mức này thì ca không đủ giống để làm kinh nghiệm
MAX_DISTANCE = 0.6
DEFECT_WEIGHT = 0.5
SCENE_BONUS = 0.2
# Sở thích phong cách: cần ít nhất chừng này lần chọn, và một phong cách chiếm đủ tỉ lệ
MIN_CHOICES = 3
MIN_STYLE_SHARE = 0.6


class CaseRecord(BaseModel):
    """Một ca đã xử lý cùng phản hồi của người dùng."""

    id: str
    user_id: str = DEFAULT_USER
    created_at: str
    scene_type: str = "other"
    signature: List[float] = Field(default_factory=list, description="Chữ ký số liệu ảnh gốc")
    defects: List[str] = Field(default_factory=list, description="'type@region:severity'")
    preserve: List[str] = Field(default_factory=list, description="'aspect@region'")
    intent_style: Optional[str] = None
    treatment: List[Dict[str, Any]] = Field(default_factory=list)
    final_score: Optional[float] = None
    decision: str = ""
    recommended_variant: Optional[str] = None
    chosen_variant: Optional[str] = None
    feedback: List[str] = Field(default_factory=list, description="'kind@region:strength'")
    feedback_texts: List[str] = Field(default_factory=list)

    @property
    def defect_types(self) -> set[str]:
        """Tập loại lỗi của ca (bỏ vùng và mức độ)."""
        return {item.split("@", 1)[0] for item in self.defects}


def metric_signature(metrics: Dict[str, Any]) -> List[float]:
    """
    Chữ ký số liệu chuẩn hóa về khoảng [0, 1] từ TechnicalMetrics (dict) của Module 1:
    độ sáng, tương phản, nhiễu, độ nét (log), tỉ lệ cháy sáng và bệt đen.
    """
    histogram = metrics.get("histogram_stats") or {}
    sharpness = float(metrics.get("sharpness_laplacian_var") or 0.0)
    values = [
        float(metrics.get("brightness_mean") or 0.0) / 255.0,
        float(metrics.get("contrast_std") or 0.0) / 128.0,
        float(metrics.get("noise_variance") or 0.0) / 30.0,
        math.log1p(sharpness) / math.log1p(10000.0),
        float(histogram.get("highlight_clip_ratio") or 0.0),
        float(histogram.get("shadow_clip_ratio") or 0.0),
    ]
    return [round(min(max(value, 0.0), 1.0), 4) for value in values]


def _distance(
    signature: Sequence[float],
    defect_types: set[str],
    scene_type: str,
    case: CaseRecord,
) -> float:
    """Khoảng cách giữa ảnh hiện tại và một ca: số liệu + loại lỗi (Jaccard) − thưởng cùng cảnh."""
    if signature and len(case.signature) == len(signature):
        numeric = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(signature, case.signature)) / len(signature)
        )
    else:
        numeric = 1.0
    union = defect_types | case.defect_types
    jaccard = len(defect_types & case.defect_types) / len(union) if union else 1.0
    bonus = SCENE_BONUS if scene_type != "other" and scene_type == case.scene_type else 0.0
    return max(numeric + DEFECT_WEIGHT * (1.0 - jaccard) - bonus, 0.0)


class CaseMemory:
    """Kho ca bệnh SQLite; an toàn khi gọi từ nhiều thread."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _db(self) -> sqlite3.Connection:
        """Kết nối mở lười; tạo thư mục và bảng ở lần dùng đầu tiên."""
        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS cases ("
                "id TEXT PRIMARY KEY, user_id TEXT, created_at TEXT, data TEXT)"
            )
            self._connection.commit()
        return self._connection

    def _save(self, case: CaseRecord) -> None:
        """Ghi (hoặc ghi đè) một ca."""
        with self._lock:
            db = self._db()
            db.execute(
                "INSERT OR REPLACE INTO cases (id, user_id, created_at, data) VALUES (?, ?, ?, ?)",
                (case.id, case.user_id, case.created_at, case.model_dump_json()),
            )
            db.commit()

    def get(self, case_id: str) -> Optional[CaseRecord]:
        """Đọc một ca theo id; None nếu không có."""
        with self._lock:
            row = self._db().execute("SELECT data FROM cases WHERE id = ?", (case_id,)).fetchone()
        return CaseRecord.model_validate_json(row[0]) if row else None

    def cases(self, user_id: Optional[str] = None) -> List[CaseRecord]:
        """Mọi ca (của một người dùng nếu có user_id), mới nhất trước."""
        query = "SELECT data FROM cases"
        params: Tuple[Any, ...] = ()
        if user_id is not None:
            query += " WHERE user_id = ?"
            params = (user_id,)
        with self._lock:
            rows = self._db().execute(query + " ORDER BY created_at DESC", params).fetchall()
        return [CaseRecord.model_validate_json(row[0]) for row in rows]

    # ---------------- ghi ----------------
    def record_run(self, state: DoctorState, user_id: str = DEFAULT_USER) -> str:
        """Ghi một lần xử lý đã xong; trả về id của ca."""
        # Import muộn: variants kéo theo executor/Module 2-3, memory cần nhẹ khi import
        from .variants import treatment_recipe

        history = state.get("history") or []
        first = history[0] if history else None
        diagnosis = first.diagnosis if first is not None else None
        metrics = (first.metrics_before if first is not None else None) or state.get(
            "technical_metrics", {}
        )
        scores = [
            float(item.eval_score["estimated_quality_score"])
            for item in history
            if not item.rolled_back and item.eval_score.get("estimated_quality_score") is not None
        ]
        intent = state.get("intent")
        case = CaseRecord(
            id=uuid.uuid4().hex,
            user_id=user_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            scene_type=diagnosis.scene_type if diagnosis else "other",
            signature=metric_signature(metrics),
            defects=[
                f"{d.type}@{d.region}:{d.severity}"
                for d in (diagnosis.actionable_defects if diagnosis else [])
            ],
            preserve=[f"{p.aspect}@{p.region}" for p in (diagnosis.preserve if diagnosis else [])],
            intent_style=intent.style if intent is not None else None,
            treatment=[a.model_dump(mode="json") for a in treatment_recipe(history)],
            final_score=max(scores) if scores else None,
            decision=str(state.get("decision") or ""),
            recommended_variant=state.get("recommended_variant"),
        )
        self._save(case)
        return case.id

    def record_choice(self, case_id: str, variant_id: str) -> bool:
        """Ghi phiên bản người dùng đã chọn/xuất; False nếu không có ca."""
        case = self.get(case_id)
        if case is None:
            return False
        self._save(case.model_copy(update={"chosen_variant": variant_id}))
        return True

    def record_feedback(self, case_id: str, text: str, adjustments: Sequence[Any]) -> bool:
        """Ghi một góp ý và các điều chỉnh đã hiểu được; False nếu không có ca."""
        case = self.get(case_id)
        if case is None:
            return False
        feedback = case.feedback + [f"{a.kind}@{a.region}:{a.strength}" for a in adjustments]
        texts = case.feedback_texts + [text]
        self._save(case.model_copy(update={"feedback": feedback, "feedback_texts": texts}))
        return True

    # ---------------- đọc ----------------
    def similar(
        self,
        diagnosis: Optional[DiagnosisReport],
        metrics: Dict[str, Any],
        user_id: Optional[str] = None,
        k: int = MAX_SIMILAR,
        exclude: Optional[str] = None,
    ) -> List[Tuple[CaseRecord, float]]:
        """k ca gần nhất (khoảng cách <= MAX_DISTANCE), gần nhất trước."""
        signature = metric_signature(metrics)
        defect_types = {d.type for d in diagnosis.actionable_defects} if diagnosis else set()
        scene_type = diagnosis.scene_type if diagnosis else "other"
        scored = [
            (case, round(_distance(signature, defect_types, scene_type, case), 4))
            for case in self.cases(user_id)
            if case.id != exclude
        ]
        close = [pair for pair in scored if pair[1] <= MAX_DISTANCE]
        return sorted(close, key=lambda pair: pair[1])[:k]

    def preferred_style(self, user_id: str = DEFAULT_USER) -> Optional[str]:
        """Phong cách người dùng chọn nhiều nhất, nếu đủ MIN_CHOICES lần và MIN_STYLE_SHARE."""
        choices = [case.chosen_variant for case in self.cases(user_id) if case.chosen_variant]
        if len(choices) < MIN_CHOICES:
            return None
        style, count = Counter(choices).most_common(1)[0]
        return style if count / len(choices) >= MIN_STYLE_SHARE else None

    def stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Thống kê gọn: số ca, lựa chọn phong cách, góp ý hay gặp, phong cách ưa thích."""
        cases = self.cases(user_id)
        feedback = Counter(item.split("@", 1)[0] for case in cases for item in case.feedback)
        return {
            "cases": len(cases),
            "choices": dict(Counter(c.chosen_variant for c in cases if c.chosen_variant)),
            "feedback": dict(feedback.most_common()),
            "preferred_style": self.preferred_style(user_id or DEFAULT_USER),
        }


def _describe_treatment(treatment: Sequence[Dict[str, Any]]) -> str:
    """Mô tả gọn một phác đồ, ví dụ 'gamma_correct(gamma=1.5) → clahe(clip_limit=1.5)'."""
    steps = []
    for action in treatment:
        params = ", ".join(f"{k}={v}" for k, v in (action.get("parameters") or {}).items())
        region = action.get("target_prompt", "full")
        where = "" if region == "full" else f"@{region}"
        steps.append(f"{action.get('operation')}{where}({params})")
    return " → ".join(steps) or "không xử lý"


def experience_prompt(cases: Sequence[Tuple[CaseRecord, float]]) -> str:
    """Khối 'KINH NGHIỆM TỪ CA TƯƠNG TỰ' cho prompt Plan; rỗng nếu không có ca."""
    if not cases:
        return ""
    lines = ["KINH NGHIỆM TỪ CA TƯƠNG TỰ (cùng người dùng, gần nhất trước):"]
    for case, distance in cases:
        defects = ", ".join(case.defects) or "không lỗi"
        line = (
            f"- Ca [{case.id[:8]}] cảnh {case.scene_type}, lỗi {defects} (khoảng cách {distance}): "
            f"phác đồ {_describe_treatment(case.treatment)}"
        )
        if case.final_score is not None:
            line += f", điểm {case.final_score}"
        if case.chosen_variant:
            line += f"; người dùng chọn bản '{case.chosen_variant}'"
        if case.feedback_texts:
            quoted = "; ".join(f'"{text}"' for text in case.feedback_texts)
            line += f"; người dùng góp ý: {quoted} ({', '.join(case.feedback)})"
        lines.append(line)
    return "\n".join(lines)


# ---------------- cấu hình toàn tiến trình ----------------
_MEMORY: Optional[CaseMemory] = None


def configure_case_memory(path: Optional[Union[str, Path]]) -> Optional[CaseMemory]:
    """Bật bộ nhớ tại path (None → tắt). Trả về kho đang dùng."""
    global _MEMORY
    _MEMORY = CaseMemory(path) if path else None
    return _MEMORY


def get_case_memory() -> Optional[CaseMemory]:
    """Kho ca bệnh đang bật, hoặc None nếu bộ nhớ tắt."""
    return _MEMORY


def remember_run(state: DoctorState, user_id: str = DEFAULT_USER) -> Optional[str]:
    """Ghi một lần xử lý nếu bộ nhớ đang bật; lỗi ghi không làm hỏng luồng chính."""
    memory = get_case_memory()
    if memory is None:
        return None
    try:
        return memory.record_run(state, user_id)
    except Exception as exc:
        logger.warning("Could not record the case: %s", exc)
        return None
