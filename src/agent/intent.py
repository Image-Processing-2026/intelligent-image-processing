"""
Hiểu ý định người dùng trước khi xử lý (Phase 4).

Sau lần chẩn đoán đầu tiên, agent hỏi đúng MỘT câu: xử lý theo đề xuất của AI hay theo cách
khác. Đề xuất được dựng TẤT ĐỊNH từ chẩn đoán (lỗi sửa được và phần cần giữ); chọn "cách
khác" thì người dùng mô tả yêu cầu ở khung chat (ghi chú và edit), và yêu cầu đó được chuyển
thành IntentProfile mà không cần VLM. Ý định người dùng ghi đè chẩn đoán của VLM.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .fusion import VERIFY_PHRASES
from .perception import _drop_preserved_defects
from .regions import canonical_region
from .state import Defect, DiagnosisEdit, DiagnosisReport, IntentProfile, PreserveItem
from .user_analysis import apply_edits

MAX_QUESTIONS = 1
USER_DEFECT_SEVERITY = 2

APPROACH_QUESTION_ID = "approach"
APPROACH_RECOMMEND = "recommend"
APPROACH_CUSTOM = "custom"

# Sửa một loại lỗi nghĩa là bỏ các aspect cần giữ mâu thuẫn với nó
FIX_CONFLICTS: Dict[str, Tuple[str, ...]] = {
    "color_cast_warm": ("warm_tone",),
    "color_cast_cool": ("cool_tone",),
    "underexposed": ("low_key", "silhouette"),
    "backlit_subject": ("low_key", "silhouette"),
    "overexposed": ("high_key",),
    "noise": ("film_grain",),
    "blur": ("soft_focus",),
    "undersaturated": ("muted_colors",),
    "oversaturated": ("vivid_colors",),
}


class Choice(BaseModel):
    """Một lựa chọn trả lời, kèm tác động của nó lên ý định."""

    value: str
    label: str
    keep: Optional[str] = Field(default=None, description="Aspect cần giữ khi chọn")
    fix: Optional[str] = Field(default=None, description="Loại lỗi cần sửa khi chọn")


class Question(BaseModel):
    """Một câu hỏi làm rõ ý định người dùng."""

    id: str
    text: str
    choices: List[Choice]
    default: str


def _where(region: str) -> str:
    """Vùng viết cho người dùng: 'toàn ảnh' hoặc tên vùng."""
    return "toàn ảnh" if canonical_region(region) == "full" else canonical_region(region)


def recommendation_text(diagnosis: DiagnosisReport) -> str:
    """
    Đề xuất xử lý của AI, tất định từ chẩn đoán: các lỗi sửa được (nặng trước), phần sẽ giữ
    nguyên (preserve) và lỗi fusion còn nghi ngờ (ask_user, mặc định giữ nguyên).
    """
    fixes = [
        f"{VERIFY_PHRASES.get(d.type, (d.type,))[0]} @ {_where(d.region)} (mức {d.severity}/3)"
        for d in sorted(diagnosis.actionable_defects, key=lambda d: -d.severity)
    ]
    parts = ["Sửa: " + "; ".join(fixes) + "."] if fixes else []
    if diagnosis.preserve:
        kept = "; ".join(f"{p.aspect} @ {_where(p.region)}" for p in diagnosis.preserve)
        parts.append(f"Giữ nguyên: {kept}.")
    if diagnosis.ask_user:
        unsure = "; ".join(
            f"{VERIFY_PHRASES.get(d.type, (d.type,))[0]} @ {_where(d.region)}"
            for d in diagnosis.ask_user
        )
        parts.append(f"Chưa chắc là lỗi hay chủ ý (mặc định giữ nguyên): {unsure}.")
    return " ".join(parts)


def build_questions(diagnosis: Optional[DiagnosisReport]) -> List[Question]:
    """
    Câu hỏi cho một chẩn đoán: đúng một câu "theo đề xuất hay cách khác" khi AI có điều gì để
    đề xuất; rỗng khi ảnh không có lỗi cần xử lý. Tất định, nên sinh lại khi resume cho đúng
    cùng câu hỏi.
    """
    if diagnosis is None or not diagnosis.actionable_defects:
        return []
    return [
        Question(
            id=APPROACH_QUESTION_ID,
            text=(
                f"AI đề xuất xử lý như sau. {recommendation_text(diagnosis)} "
                "Bạn muốn xử lý theo đề xuất này hay theo cách khác?"
            ),
            choices=[
                Choice(value=APPROACH_RECOMMEND, label="Theo đề xuất của AI"),
                Choice(
                    value=APPROACH_CUSTOM,
                    label="Cách khác (mô tả yêu cầu của bạn ở khung chat bên dưới)",
                ),
            ],
            default=APPROACH_RECOMMEND,
        )
    ]


# ---------------------------------------------------------
# Ghi chú tự do → câu trả lời
# ---------------------------------------------------------
# (id câu hỏi, lựa chọn, mẫu regex trên ghi chú đã chữ thường); mẫu trước được ưu tiên
NOTE_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("warm", "fix", r"(khử|bớt|sửa|hết)\s*(ám\s*)?(vàng|ấm|cam)"),
    ("warm", "keep", r"giữ\s*(tông\s*|màu\s*|không khí\s*)?(vàng|ấm)|tông ấm|ấm áp"),
    ("cool", "fix", r"(khử|bớt|sửa|hết)\s*(ám\s*)?(xanh|lạnh)"),
    ("cool", "keep", r"giữ\s*(tông\s*|màu\s*)?(xanh|lạnh)"),
    ("mood", "keep_dark", r"giữ\s*(không khí\s*)?tối|tâm trạng|huyền bí|low[- ]?key|bóng đen"),
    ("mood", "brighten", r"làm sáng|sáng (rõ|lên|hơn)|thấy rõ mặt"),
    ("grain", "keep", r"giữ\s*hạt|kiểu film|chất film"),
    ("grain", "smooth", r"mịn|khử nhiễu|hết nhiễu|bớt nhiễu|hết hạt|bớt hạt"),
)

# Tác động mặc định khi ghi chú nói về một câu không được hỏi
DEFAULT_EFFECTS: Dict[Tuple[str, str], Choice] = {
    ("warm", "fix"): Choice(value="fix", label="", fix="color_cast_warm"),
    ("warm", "keep"): Choice(value="keep", label="", keep="warm_tone"),
    ("cool", "fix"): Choice(value="fix", label="", fix="color_cast_cool"),
    ("cool", "keep"): Choice(value="keep", label="", keep="cool_tone"),
    ("mood", "keep_dark"): Choice(value="keep_dark", label="", keep="low_key"),
    ("mood", "brighten"): Choice(value="brighten", label="", fix="underexposed"),
    ("grain", "keep"): Choice(value="keep", label="", keep="film_grain"),
    ("grain", "smooth"): Choice(value="smooth", label="", fix="noise"),
}


def parse_notes(notes: str) -> Dict[str, str]:
    """Suy câu trả lời từ ghi chú tự do (mỗi câu hỏi lấy mẫu khớp đầu tiên)."""
    text = notes.casefold()
    answers: Dict[str, str] = {}
    for question_id, value, pattern in NOTE_RULES:
        if question_id not in answers and re.search(pattern, text):
            answers[question_id] = value
    return answers


def intent_from_answers(
    questions: Sequence[Question],
    answers: Dict[str, str],
    notes: str = "",
    edits: Sequence[DiagnosisEdit] = (),
) -> IntentProfile:
    """
    Chuyển câu trả lời (và ghi chú) thành IntentProfile. Câu trả lời chọn trực tiếp thắng
    ghi chú; giá trị không có trong lựa chọn của câu hỏi bị bỏ qua. edits (đã parse ở tầng
    API) được gắn nguyên vẹn vào intent.edits.
    """
    by_id = {question.id: question for question in questions}
    merged = {**parse_notes(notes), **{k: v for k, v in answers.items() if v}}
    intent = IntentProfile(notes=notes.strip(), edits=list(edits))
    for question_id, value in merged.items():
        question = by_id.get(question_id)
        if question is not None:
            choice = next((c for c in question.choices if c.value == value), None)
        else:
            choice = DEFAULT_EFFECTS.get((question_id, value))
        if choice is None:
            continue
        intent.answers[question_id] = value
        if choice.keep and all(item.aspect != choice.keep for item in intent.keep):
            intent.keep.append(PreserveItem(aspect=choice.keep, reason="Người dùng yêu cầu giữ"))
        if choice.fix and choice.fix not in intent.fix:
            intent.fix.append(choice.fix)
    return intent


# ---------------------------------------------------------
# Áp ý định lên chẩn đoán
# ---------------------------------------------------------
def apply_intent(diagnosis: DiagnosisReport, intent: Optional[IntentProfile]) -> DiagnosisReport:
    """
    Ghi đè chẩn đoán bằng ý định người dùng (bản sao, không sửa bản gốc):
    1. Bỏ aspect cần giữ mâu thuẫn với lỗi người dùng muốn sửa.
    2. Thêm aspect người dùng muốn giữ.
    3. Bảo đảm lỗi người dùng muốn sửa có mặt với severity >= USER_DEFECT_SEVERITY.
    4. Bỏ lỗi mâu thuẫn với các aspect cần giữ.
    5. Áp các DiagnosisEdit (thêm/bỏ lỗi, giữ) từ chat bổ sung phân tích.
    """
    if intent is None:
        return diagnosis
    conflicting = {aspect for fix in intent.fix for aspect in FIX_CONFLICTS.get(fix, ())}
    preserve = [item for item in diagnosis.preserve if item.aspect not in conflicting]
    for item in intent.keep:
        if all((p.aspect, p.region) != (item.aspect, item.region) for p in preserve):
            preserve.append(item)

    defects = [defect.model_copy() for defect in diagnosis.defects]
    for fix in intent.fix:
        matching = [defect for defect in defects if defect.type == fix]
        if not matching:
            # Lỗi fusion đã hỏi người dùng (ask_user): sửa đúng vùng và mức đã chẩn đoán
            matching = [
                defect.model_copy(update={"origin": "user"})
                for defect in diagnosis.ask_user
                if defect.type == fix
            ]
            defects += matching
        for defect in matching:
            defect.severity = max(defect.severity, USER_DEFECT_SEVERITY)
        if not matching:
            defects.append(
                Defect(
                    type=fix,
                    severity=USER_DEFECT_SEVERITY,
                    evidence="Người dùng yêu cầu sửa.",
                    origin="user",
                )
            )

    updated = diagnosis.model_copy(update={"preserve": preserve, "defects": defects})
    updated.defects = _drop_preserved_defects(updated)
    # Edit từ chat bổ sung phân tích áp sau cùng: người dùng thắng cả câu trả lời trắc nghiệm
    return apply_edits(updated, intent.edits)


def _edit_label(edit: DiagnosisEdit) -> str:
    """Nhãn ngắn của một edit cho prompt: 'add underexposed@face (3)', 'keep warm_tone@full'."""
    if edit.op == "add_defect":
        return f"add {edit.type}@{edit.region} ({edit.severity})"
    if edit.op == "remove_defect":
        return f"remove {edit.type}@{edit.region}"
    return f"keep {edit.aspect}@{edit.region}"


def intent_prompt(intent: Optional[IntentProfile]) -> str:
    """Khối 'Ý ĐỊNH NGƯỜI DÙNG' cho prompt giai đoạn Plan; rỗng nếu không có ý định."""
    if intent is None:
        return ""
    lines: List[str] = []
    if intent.keep:
        lines.append("- Muốn giữ: " + ", ".join(f"{p.aspect}@{p.region}" for p in intent.keep))
    if intent.fix:
        lines.append("- Muốn sửa: " + ", ".join(intent.fix))
    if intent.edits:
        lines.append(
            "- Người dùng bổ sung phân tích: " + ", ".join(_edit_label(e) for e in intent.edits)
        )
    if intent.notes:
        lines.append(f"- Ghi chú: {intent.notes}")
    if not lines:
        return ""
    return "Ý ĐỊNH NGƯỜI DÙNG (ưu tiên cao nhất, ghi đè chẩn đoán):\n" + "\n".join(lines)
