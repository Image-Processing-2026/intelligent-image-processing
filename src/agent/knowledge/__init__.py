"""
Knowledge Base của Module 4 (Phase 2): playbook card (tầng A) và nguyên lý (tầng B).
Dùng cho prompt giai đoạn Plan và cho rule engine offline.
"""

from .engine import diagnose_from_metrics, plan_actions_from_knowledge, with_metric_defects
from .formatting import format_context
from .models import KnowledgeBase, PlaybookCard, Principle
from .retriever import CardMatch, KnowledgeContext, match_cards, retrieve
from .store import KnowledgeBaseError, get_knowledge_base, load_knowledge_base

__all__ = [
    "CardMatch",
    "KnowledgeBase",
    "KnowledgeBaseError",
    "KnowledgeContext",
    "PlaybookCard",
    "Principle",
    "diagnose_from_metrics",
    "format_context",
    "get_knowledge_base",
    "load_knowledge_base",
    "match_cards",
    "plan_actions_from_knowledge",
    "retrieve",
    "with_metric_defects",
]
