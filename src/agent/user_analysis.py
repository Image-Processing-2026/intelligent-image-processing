"""
Bổ sung phân tích bằng lời của người dùng (chat sau bước Perceive).

Mỗi tin nhắn được parse thành các DiagnosisEdit (thêm lỗi model bỏ sót, bỏ lỗi model nhận
nhầm, đánh dấu điều cần giữ). Parse theo hai tầng như refine.py: luật tiếng Việt trước (chạy
offline), luật không hiểu và có GEMINI_API_KEY thì hỏi Gemini với JSON schema (chỉ gửi văn
bản, không gửi ảnh). apply_edits áp các edit lên chẩn đoán một cách tất định; người dùng có
thẩm quyền cao nhất nên edit ghi đè kết quả fusion và preserve của VLM.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .fusion import (
    CONFLICTING_DEFECTS,
    INTENTIONAL_ASPECT,
    NON_FIXABLE_DEFECTS,
    drop_preserved_defects,
)
from .regions import canonical_region
from .state import (
    DEFECT_TYPES,
    PRESERVE_ASPECTS,
    Defect,
    DiagnosisEdit,
    DiagnosisReport,
    Evidence,
    PreserveItem,
)
from .vlm_diagnostician import call_gemini_json

logger = logging.getLogger(__name__)

# Chủ thể và các vùng người: "mặt tối" có thể là cùng một lỗi ngược sáng trên chủ thể
SUBJECT_REGIONS = frozenset(("face", "person", "subject"))
NON_SUBJECT_REGIONS = frozenset(("full", "background", "sky"))

# ---------------------------------------------------------
# Luật tiếng Việt
# ---------------------------------------------------------
# Tách mệnh đề; không tách theo "còn"/"và" vì sẽ cắt rời vùng khỏi lỗi ("mặt còn tối")
_CLAUSE_SPLIT = re.compile(r"[,;.!\n]|\bnhưng\b|\bmà\b")
# (mẫu vùng, tên vùng); mẫu trước được ưu tiên
REGION_RULES: Tuple[Tuple[str, str], ...] = (
    (r"\bda\b|khuôn mặt|gương mặt|\bmặt\b", "face"),
    (r"bầu trời|\btrời\b", "sky"),
    (r"\bnền\b|\bphông\b|hậu cảnh", "background"),
    (r"\bngười\b|chủ thể|nhân vật|\bem bé\b", "person"),
)
# (aspect, mẫu nội dung sau "giữ"); chỉ coi là giữ khi mệnh đề có "giữ"
KEEP_RULES: Tuple[Tuple[str, str], ...] = (
    ("warm_tone", r"vàng|ấm"),
    ("cool_tone", r"xanh|lạnh"),
    ("low_key", r"tối|low[- ]?key|tâm trạng|huyền bí"),
    ("silhouette", r"bóng đen|silhouette|hình bóng"),
    ("high_key", r"sáng (trong|tràn)|high[- ]?key"),
    ("film_grain", r"hạt|film"),
    ("soft_focus", r"mờ ảo|soft|mềm"),
    ("muted_colors", r"màu (nhạt|dịu|trầm)"),
    ("vivid_colors", r"rực|màu (đậm|tươi)"),
)
# (loại lỗi, mẫu); mẫu cụ thể đặt trước mẫu chung ("mờ sương" trước "mờ", "rung" trước "mờ")
DEFECT_RULES: Tuple[Tuple[str, str], ...] = (
    ("haze", r"mờ sương|sương mù|bụi mờ|\bhaze\b"),
    ("motion_blur", r"rung tay|nhòe chuyển động|mờ chuyển động|motion"),
    ("jpeg_artifacts", r"vỡ (ảnh|hạt|khối)|ô vuông|nén (quá|nhiều)|\bjpeg\b"),
    ("color_cast_green", r"ám (xanh lá|lục)|ngả (xanh lá|lục)"),
    ("color_cast_warm", r"ám (vàng|cam|ấm)|ngả (vàng|cam)|(vàng|cam) quá|quá (vàng|cam)"),
    ("color_cast_cool", r"ám (xanh|lạnh)|ngả xanh|(xanh|lạnh) quá|quá (xanh|lạnh)"),
    ("oversaturated", r"rực quá|quá rực|lòe loẹt|màu gắt|quá đậm màu"),
    ("undersaturated", r"nhạt màu|màu nhạt|thiếu màu|xỉn"),
    ("low_contrast", r"\bbệt\b|thiếu tương phản|nhạt nhòa|phẳng"),
    ("noise", r"nhiễu|sạn|lấm tấm|\bhạt\b"),
    ("blur", r"\bmờ\b|nhòe|(không|chưa|thiếu) nét"),
    ("overexposed", r"cháy|chói|quá sáng|sáng quá|bệch"),
    ("underexposed", r"\btối\b|thiếu sáng|chìm"),
)
# Phủ định/chủ ý: "không (bị) X", "X là cố ý", "đừng sửa X"; "không nét" lại là lỗi mờ
_NEGATION = re.compile(r"\b(không|chẳng|chả)\b(?!\s*nét)|đừng (sửa|chỉnh|làm)|cố ý|chủ ý|chủ đích")
_INTENT_WORDS = re.compile(r"cố ý|chủ ý|chủ đích")
STRONG_PATTERN = re.compile(r"quá|rất|nhiều|hẳn|lắm|nặng")
WEAK_PATTERN = re.compile(r"hơi|chút|một tí|nhẹ")


def _clause_region(clause: str) -> str:
    """Vùng được nhắc trong một mệnh đề (mặc định toàn ảnh)."""
    for pattern, region in REGION_RULES:
        if re.search(pattern, clause):
            return region
    return "full"


def _clause_severity(clause: str) -> int:
    """Mức độ: 'quá/rất/nặng' → 3; 'hơi/chút' → 1; mặc định 2."""
    if WEAK_PATTERN.search(clause):
        return 1
    return 3 if STRONG_PATTERN.search(clause) else 2


def _first_defect(clause: str) -> Optional[str]:
    """Loại lỗi khớp đầu tiên theo DEFECT_RULES."""
    for defect_type, pattern in DEFECT_RULES:
        if re.search(pattern, clause):
            return defect_type
    return None


def _first_keep(clause: str) -> Optional[str]:
    """Aspect cần giữ nếu mệnh đề có 'giữ' + một từ khóa aspect."""
    match = re.search(r"giữ(.*)", clause)
    if match is None:
        return None
    for aspect, pattern in KEEP_RULES:
        if re.search(pattern, match.group(1)):
            return aspect
    return None


def validate_edits(raw: Any) -> List[DiagnosisEdit]:
    """
    Chuyển payload thô (list dict hoặc DiagnosisEdit) thành DiagnosisEdit hợp lệ. Edit hỏng
    bị bỏ qua và ghi warning; payload không phải list → rỗng. Không bao giờ raise.
    """
    if not isinstance(raw, (list, tuple)):
        return []
    edits: List[DiagnosisEdit] = []
    for item in raw:
        try:
            edits.append(
                item if isinstance(item, DiagnosisEdit) else DiagnosisEdit.model_validate(item)
            )
        except Exception as exc:
            logger.warning("Skipping invalid diagnosis edit %r: %s", item, exc)
    return edits


def parse_edits_rules(text: str) -> List[DiagnosisEdit]:
    """Tách tin nhắn thành mệnh đề; mỗi mệnh đề lấy tối đa một luật (keep > phủ định > thêm)."""
    edits: List[DiagnosisEdit] = []
    for raw in _CLAUSE_SPLIT.split(text.casefold()):
        clause = raw.strip()
        if not clause:
            continue
        region = _clause_region(clause)
        aspect = _first_keep(clause)
        if aspect is not None:
            edits.append(DiagnosisEdit(op="keep", aspect=aspect, region=region, source_text=clause))
            continue
        defect_type = _first_defect(clause)
        if defect_type is None:
            continue
        if _NEGATION.search(clause):
            edits.append(
                DiagnosisEdit(
                    op="remove_defect", type=defect_type, region=region, source_text=clause
                )
            )
            intentional = INTENTIONAL_ASPECT.get(defect_type)
            if intentional is not None and _INTENT_WORDS.search(clause):
                edits.append(
                    DiagnosisEdit(op="keep", aspect=intentional, region=region, source_text=clause)
                )
            continue
        edits.append(
            DiagnosisEdit(
                op="add_defect",
                type=defect_type,
                region=region,
                severity=_clause_severity(clause),
                source_text=clause,
            )
        )
    return edits


# ---------------------------------------------------------
# Gemini cho câu luật không hiểu
# ---------------------------------------------------------
EDITS_PROMPT = """
Người dùng đang bổ sung/sửa kết quả phân tích lỗi của một bức ảnh trước khi xử lý. Hãy chuyển
tin nhắn của họ thành các chỉnh sửa có cấu trúc:
- add_defect: người dùng nói ảnh có một lỗi mà phân tích chưa nêu (type thuộc danh sách cho
  phép, region là "full" hoặc tên vùng ngắn bằng tiếng Anh như "face", "sky", "background",
  "person", severity 1 nhẹ - 3 nặng).
- remove_defect: người dùng nói một lỗi đã nêu là sai hoặc là cố ý.
- keep: người dùng muốn giữ một đặc điểm thẩm mỹ (aspect thuộc danh sách cho phép).
Nếu tin nhắn không nói gì về lỗi hay điều cần giữ, trả về danh sách rỗng.
"""

EDITS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add_defect", "remove_defect", "keep"]},
                    "type": {"type": "string", "enum": list(DEFECT_TYPES)},
                    "aspect": {"type": "string", "enum": list(PRESERVE_ASPECTS)},
                    "region": {"type": "string"},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["op"],
            },
        }
    },
    "required": ["edits"],
}


def summarize_diagnosis(diagnosis: Optional[DiagnosisReport]) -> str:
    """Tóm tắt chẩn đoán hiện tại (type@region và preserve) cho prompt Gemini."""
    if diagnosis is None:
        return "Chẩn đoán hiện tại: (chưa có)"
    defects = ", ".join(
        f"{d.type}@{canonical_region(d.region)} ({d.severity})" for d in diagnosis.defects
    )
    preserve = ", ".join(f"{p.aspect}@{canonical_region(p.region)}" for p in diagnosis.preserve)
    return (
        f"Chẩn đoán hiện tại:\n- Lỗi: {defects or 'không có'}\n- Cần giữ: {preserve or 'không có'}"
    )


def _parse_vlm_edits(data: Dict[str, Any], text: str) -> List[DiagnosisEdit]:
    """Chuyển JSON của Gemini thành edit hợp lệ; edit hỏng bị bỏ và ghi log."""
    parsed: List[DiagnosisEdit] = []
    for raw in data.get("edits") or []:
        try:
            parsed.append(
                DiagnosisEdit(
                    op=raw.get("op"),
                    type=raw.get("type"),
                    aspect=raw.get("aspect"),
                    region=raw.get("region") or "full",
                    severity=int(raw.get("severity") or 2),
                    source_text=text.strip(),
                )
            )
        except (ValueError, TypeError, AttributeError) as exc:
            logger.warning("Dropping invalid analysis edit %s: %s", raw, exc)
    return parsed


def parse_edits(
    text: str, diagnosis: Optional[DiagnosisReport] = None
) -> Tuple[List[DiagnosisEdit], str]:
    """
    Phân tích một tin nhắn bổ sung phân tích: luật trước; luật không hiểu và có API key → VLM.

    Returns:
        (edits, nguồn "rule" | "vlm" | "vlm_fallback" | "none").
    """
    edits = parse_edits_rules(text)
    if edits:
        return edits, "rule"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not text.strip():
        return [], "none"
    prompt = f"{EDITS_PROMPT}\n{summarize_diagnosis(diagnosis)}"
    try:
        data = call_gemini_json(api_key, prompt, [text], EDITS_SCHEMA)
        edits = _parse_vlm_edits(data, text)
    except Exception as exc:
        logger.warning("Analysis edit parsing with Gemini failed (%s).", exc)
        return [], "vlm_fallback"
    return edits, "vlm" if edits else "none"


# ---------------------------------------------------------
# Áp edit lên chẩn đoán
# ---------------------------------------------------------
def _same_defect(defect: Defect, edit: DiagnosisEdit) -> bool:
    """Lỗi cùng loại và cùng vùng chuẩn với edit."""
    return defect.type == edit.type and canonical_region(defect.region) == edit.region


def _conflicting_aspects(defect_type: str) -> frozenset:
    """Các aspect cần giữ mâu thuẫn với một loại lỗi."""
    return frozenset(
        aspect for aspect, types in CONFLICTING_DEFECTS.items() if defect_type in types
    )


def _upgrade_backlit(defects: List[Defect], edit: DiagnosisEdit) -> bool:
    """
    'Mặt/chủ thể tối' mà chẩn đoán đã có ngược sáng trên chủ thể: nâng severity lỗi đó thay vì
    thêm lỗi mới. Trả True nếu đã xử lý.
    """
    if edit.type != "underexposed" or edit.region in NON_SUBJECT_REGIONS:
        return False
    found = False
    for defect in defects:
        region = canonical_region(defect.region)
        if defect.type != "backlit_subject" or region in NON_SUBJECT_REGIONS:
            continue
        if region == edit.region or (region in SUBJECT_REGIONS and edit.region in SUBJECT_REGIONS):
            defect.severity = max(defect.severity, edit.severity)
            defect.origin = "user"
            defect.confidence = 1.0
            defect.evidence_items.append(Evidence(source="user", text=edit.source_text))
            found = True
    return found


def _add_defect(
    defects: List[Defect], preserve: List[PreserveItem], edit: DiagnosisEdit
) -> List[PreserveItem]:
    """Thêm hoặc nâng severity một lỗi do người dùng báo; bỏ preserve mâu thuẫn. Trả preserve."""
    assert edit.type is not None
    conflicting = _conflicting_aspects(edit.type)
    preserve = [
        item
        for item in preserve
        if not (
            item.aspect in conflicting and canonical_region(item.region) in (edit.region, "full")
        )
    ]
    if _upgrade_backlit(defects, edit):
        return preserve
    existing = [defect for defect in defects if _same_defect(defect, edit)]
    for defect in existing:
        defect.severity = max(defect.severity, edit.severity)
        defect.origin = "user"
        defect.confidence = 1.0
        defect.evidence_items.append(Evidence(source="user", text=edit.source_text))
    if not existing:
        defects.append(
            Defect(
                type=edit.type,
                region=edit.region,
                severity=edit.severity,
                evidence=f"Người dùng bổ sung: {edit.source_text}".strip().rstrip(":"),
                origin="user",
                confidence=1.0,
                evidence_items=[Evidence(source="user", text=edit.source_text)],
                fixable=edit.type not in NON_FIXABLE_DEFECTS,
            )
        )
    return preserve


def apply_edits(diagnosis: DiagnosisReport, edits: List[DiagnosisEdit]) -> DiagnosisReport:
    """
    Áp các edit lên chẩn đoán theo thứ tự (bản sao, không sửa bản gốc); edit sau thắng edit
    trước trên cùng (loại lỗi, vùng).
    - add_defect: thêm lỗi origin="user" hoặc nâng severity lỗi sẵn có; bỏ preserve mâu thuẫn.
    - remove_defect: bỏ lỗi khỏi defects, đưa vào rejected, ghi fusion_notes.
    - keep: thêm preserve rồi bỏ các lỗi mâu thuẫn như apply_intent.
    """
    if not edits:
        return diagnosis
    updated = diagnosis.model_copy(deep=True)
    for edit in edits:
        if edit.op == "add_defect":
            updated.preserve = _add_defect(updated.defects, updated.preserve, edit)
            updated.rejected = [d for d in updated.rejected if not _same_defect(d, edit)]
        elif edit.op == "remove_defect":
            removed = [d for d in updated.defects if _same_defect(d, edit)]
            updated.defects = [d for d in updated.defects if not _same_defect(d, edit)]
            updated.rejected += removed
            updated.ask_user = [d for d in updated.ask_user if not _same_defect(d, edit)]
            updated.fusion_notes.append(f"{edit.type}@{edit.region}: user removed")
        else:
            assert edit.aspect is not None
            if all(
                (p.aspect, canonical_region(p.region)) != (edit.aspect, edit.region)
                for p in updated.preserve
            ):
                updated.preserve.append(
                    PreserveItem(
                        aspect=edit.aspect, region=edit.region, reason="Người dùng yêu cầu giữ"
                    )
                )
            updated.defects = drop_preserved_defects(updated)
    return updated
