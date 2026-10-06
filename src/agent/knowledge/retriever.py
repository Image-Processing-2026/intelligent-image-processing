"""
Truy xuất tri thức cho một chẩn đoán.
- Card (tầng A): lọc theo cấu trúc (lỗi + vùng + severity, loại cảnh, preserve chặn, điều kiện
  loại trừ, chỉ số toàn cục) rồi xếp hạng theo severity, priority và độ khớp văn bản BM25.
- Nguyên lý (tầng B): xếp hạng BM25 theo chẩn đoán và các thao tác của card đã chọn.
KB nhỏ (vài chục mục) và truy vấn chủ yếu là định danh có cấu trúc, nên không dùng embedding:
không tốn thêm lời gọi API mỗi vòng và chạy hoàn toàn offline.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..state import Defect, DiagnosisReport
from .models import DefectCondition, KnowledgeBase, PlaybookCard, Principle
from .store import get_knowledge_base

MAX_CARDS = 4
MAX_PRINCIPLES = 3
# Trọng số của độ khớp văn bản (đã chuẩn hóa về [0, 1]) so với severity của lỗi khớp
TEXT_WEIGHT = 0.5
SCENE_BONUS = 1.0

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------
# BM25
# ---------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Tách từ (Unicode, chữ thường); định danh có '_' được thêm cả từng phần."""
    tokens: List[str] = []
    for token in _TOKEN_PATTERN.findall(text.casefold()):
        tokens.append(token)
        if "_" in token:
            tokens += [part for part in token.split("_") if part]
    return tokens


class BM25:
    """Okapi BM25 cho tập tài liệu nhỏ."""

    def __init__(self, documents: Sequence[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """Lập chỉ mục tần suất từ và IDF cho tập tài liệu đã tách từ."""
        self.k1 = k1
        self.b = b
        self.frequencies = [Counter(document) for document in documents]
        self.lengths = [len(document) for document in documents]
        self.average_length = sum(self.lengths) / max(len(documents), 1)
        document_frequency: Counter = Counter()
        for document in documents:
            document_frequency.update(set(document))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: Iterable[str]) -> List[float]:
        """Điểm BM25 của từng tài liệu cho một truy vấn."""
        terms = set(query)
        results: List[float] = []
        for frequencies, length in zip(self.frequencies, self.lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1e-9))
            for term in terms:
                tf = frequencies.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            results.append(score)
        return results


def _normalized(scores: List[float]) -> List[float]:
    """Chuẩn hóa điểm về [0, 1] theo điểm cao nhất (toàn 0 nếu không có điểm dương)."""
    top = max(scores, default=0.0)
    return [score / top if top > 0 else 0.0 for score in scores]


# ---------------------------------------------------------
# Khớp card với chẩn đoán
# ---------------------------------------------------------
@dataclass(frozen=True)
class CardMatch:
    """Một card áp dụng được, kèm các lỗi đã khớp (bindings) và điểm xếp hạng."""

    card: PlaybookCard
    score: float
    bindings: Tuple[Defect, ...] = ()


@dataclass
class KnowledgeContext:
    """Tri thức truy xuất cho một chẩn đoán."""

    cards: List[CardMatch] = field(default_factory=list)
    principles: List[Principle] = field(default_factory=list)

    @property
    def ids(self) -> List[str]:
        """id của các card và nguyên lý đã truy xuất, theo thứ tự xếp hạng."""
        return [match.card.id for match in self.cards] + [p.id for p in self.principles]


def _region_matches(selector: str, region: str) -> bool:
    """Bộ chọn vùng (full/region/any/tên vùng) có khớp vùng của lỗi không."""
    if selector == "any":
        return True
    if selector == "full":
        return region == "full"
    if selector == "region":
        return region != "full"
    return region == selector


def _matching_defects(condition: DefectCondition, defects: Sequence[Defect]) -> List[Defect]:
    """Các lỗi khớp loại, vùng và severity tối thiểu của một điều kiện."""
    return [
        defect
        for defect in defects
        if defect.type in condition.types
        and defect.severity >= condition.min_severity
        and _region_matches(condition.region, defect.region)
    ]


def _metric_ok(value: Any, operator: str, expected: Any) -> bool:
    """Giá trị chỉ số có thỏa một toán tử (in/lt/le/gt/ge) không; thiếu → False."""
    if value is None:
        return False
    if operator == "in":
        return value in expected
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return {
        "lt": number < expected,
        "le": number <= expected,
        "gt": number > expected,
        "ge": number >= expected,
    }[operator]


def _metrics_ok(card: PlaybookCard, metrics: Dict[str, Any]) -> bool:
    """Mọi điều kiện chỉ số toàn cục của card đều thỏa."""
    return all(
        _metric_ok(metrics.get(name), operator, expected)
        for name, condition in card.metrics.items()
        for operator, expected in condition.items()
    )


def _bindings(card: PlaybookCard, defects: Sequence[Defect]) -> Optional[List[Defect]]:
    """Lỗi khớp điều kiện của card; None nếu card không áp dụng."""
    if not card.defects:
        return []
    per_condition = [_matching_defects(condition, defects) for condition in card.defects]
    if card.match == "all" and not all(per_condition):
        return None
    bound: List[Defect] = []
    for matched in per_condition:
        bound += [defect for defect in matched if defect not in bound]
    return bound or None


def _blocked(card: PlaybookCard, diagnosis: DiagnosisReport, regions: Set[str]) -> bool:
    """Card bị chặn nếu một đặc điểm cần giữ (toàn ảnh hoặc trên vùng của lỗi) nằm trong blocked_by."""
    preserved = {
        item.aspect
        for item in diagnosis.preserve
        if item.region == "full" or item.region in regions
    }
    return bool(preserved & set(card.blocked_by))


def _card_text(card: PlaybookCard) -> List[str]:
    """Token văn bản của card dùng cho BM25."""
    parts = [card.id, card.title, card.rationale, *card.tags, *card.avoid, *card.scenes]
    parts += [defect_type for condition in card.defects for defect_type in condition.types]
    parts += [step.operation for step in card.recipe]
    return tokenize(" ".join(parts))


def _query_tokens(diagnosis: DiagnosisReport) -> List[str]:
    """Token truy vấn dựng từ chẩn đoán (cảnh, ánh sáng, lỗi, preserve)."""
    parts = [diagnosis.scene_type, diagnosis.lighting, diagnosis.summary]
    for defect in diagnosis.defects:
        parts += [defect.type, defect.region, defect.evidence]
    parts += [item.aspect for item in diagnosis.preserve]
    return tokenize(" ".join(parts))


def match_cards(
    diagnosis: DiagnosisReport,
    metrics: Optional[Dict[str, Any]] = None,
    kb: Optional[KnowledgeBase] = None,
) -> List[CardMatch]:
    """Mọi card áp dụng được cho chẩn đoán, xếp hạng giảm dần."""
    kb = kb or get_knowledge_base()
    metrics = metrics or {}
    defects = diagnosis.actionable_defects
    text_scores = _normalized(
        BM25([_card_text(card) for card in kb.cards]).scores(_query_tokens(diagnosis))
    )

    matches: List[CardMatch] = []
    for card, text_score in zip(kb.cards, text_scores):
        if card.scenes and diagnosis.scene_type not in card.scenes:
            continue
        bound = _bindings(card, defects)
        if bound is None:
            continue
        if any(_matching_defects(condition, defects) for condition in card.unless):
            continue
        if _blocked(card, diagnosis, {defect.region for defect in bound}):
            continue
        if not _metrics_ok(card, metrics):
            continue
        score = (
            card.priority / 100
            + sum(defect.severity for defect in bound)
            + (SCENE_BONUS if card.scenes else 0.0)
            + TEXT_WEIGHT * text_score
        )
        matches.append(CardMatch(card=card, score=round(score, 4), bindings=tuple(bound)))
    return sorted(matches, key=lambda match: match.score, reverse=True)


def retrieve(
    diagnosis: DiagnosisReport,
    metrics: Optional[Dict[str, Any]] = None,
    kb: Optional[KnowledgeBase] = None,
    max_cards: int = MAX_CARDS,
    max_principles: int = MAX_PRINCIPLES,
) -> KnowledgeContext:
    """Card phù hợp nhất và nguyên lý liên quan nhất cho một chẩn đoán."""
    kb = kb or get_knowledge_base()
    cards = match_cards(diagnosis, metrics, kb)[:max_cards]

    query = _query_tokens(diagnosis)
    for match in cards:
        query += tokenize(" ".join(step.operation for step in match.card.recipe))
    principle_scores = BM25(
        [tokenize(" ".join([p.title, *p.tags, p.text])) for p in kb.principles]
    ).scores(query)
    ranked = sorted(
        (pair for pair in zip(principle_scores, kb.principles) if pair[0] > 0),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return KnowledgeContext(cards=cards, principles=[p for _, p in ranked[:max_principles]])
