"""
Nạp Knowledge Base từ đĩa: card YAML (tầng A) và nguyên lý markdown (tầng B).
KB được kiểm tra khi nạp; card sai làm hỏng việc nạp kèm thông báo chỉ rõ file và card.
"""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import ValidationError

from .models import KnowledgeBase, PlaybookCard, Principle

DEFAULT_KB_DIR = Path(__file__).resolve().parent

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class KnowledgeBaseError(ValueError):
    """Knowledge Base không hợp lệ (YAML sai cú pháp hoặc card vi phạm schema)."""


def _load_cards(cards_dir: Path) -> List[PlaybookCard]:
    """Mỗi file YAML chứa một danh sách card."""
    cards: List[PlaybookCard] = []
    for path in sorted(cards_dir.glob("*.yaml")):
        try:
            raw_cards = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        except yaml.YAMLError as exc:
            raise KnowledgeBaseError(f"{path.name}: invalid YAML: {exc}") from exc
        if not isinstance(raw_cards, list):
            raise KnowledgeBaseError(f"{path.name}: expected a list of cards")
        for index, raw in enumerate(raw_cards):
            card_id = raw.get("id", f"#{index}") if isinstance(raw, dict) else f"#{index}"
            try:
                cards.append(PlaybookCard.model_validate(raw))
            except ValidationError as exc:
                raise KnowledgeBaseError(f"{path.name}: card '{card_id}': {exc}") from exc
    return cards


def _slugify(text: str) -> str:
    """Slug ASCII cho id: bỏ dấu tiếng Việt ('đ' → 'd'), chỉ giữ chữ, số và '-'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SLUG_PATTERN.sub("-", ascii_text).strip("-")


def _split_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_principles(path: Path) -> List[Principle]:
    """
    Tách file markdown theo tiêu đề '## '. Trong mỗi mục, các dòng 'tags:' và 'sources:'
    (nếu có) là siêu dữ liệu; phần còn lại là nội dung.
    """
    principles: List[Principle] = []
    sections = re.split(r"^## +", path.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]
    for section in sections:
        title, _, body = section.partition("\n")
        tags: List[str] = []
        sources: List[str] = []
        lines: List[str] = []
        for line in body.splitlines():
            lowered = line.strip().casefold()
            if lowered.startswith("tags:"):
                tags = _split_list(line.split(":", 1)[1])
            elif lowered.startswith("sources:"):
                sources = [s.strip() for s in line.split(":", 1)[1].split(";") if s.strip()]
            else:
                lines.append(line)
        principles.append(
            Principle(
                id=f"{path.stem}/{_slugify(title)}",
                title=title.strip(),
                # Gộp các dòng thành một đoạn (markdown xuống dòng mềm)
                text=" ".join(" ".join(lines).split()),
                tags=tags,
                sources=sources,
            )
        )
    return principles


def load_knowledge_base(kb_dir: Optional[Path] = None) -> KnowledgeBase:
    """Nạp và kiểm tra KB từ thư mục (mặc định: src/agent/knowledge)."""
    root = Path(kb_dir) if kb_dir is not None else DEFAULT_KB_DIR
    cards = _load_cards(root / "cards")
    principles: List[Principle] = []
    for path in sorted((root / "principles").glob("*.md")):
        principles += _parse_principles(path)
    try:
        return KnowledgeBase(cards=cards, principles=principles)
    except ValidationError as exc:
        raise KnowledgeBaseError(str(exc)) from exc


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """KB mặc định, nạp một lần cho cả tiến trình."""
    return load_knowledge_base()
