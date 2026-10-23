"""
Hiểu ý định người dùng trước khi xử lý (Phase 4).

Sau lần chẩn đoán đầu tiên, agent hỏi tối đa MAX_QUESTIONS câu. Câu hỏi được sinh TẤT ĐỊNH
từ chẩn đoán: chỉ hỏi về chỗ thật sự mơ hồ (ám vàng là chủ ý hay lỗi, giữ không khí tối hay
làm sáng, giữ hạt hay làm mịn), cộng một câu về phong cách mong muốn. Mỗi lựa chọn mang sẵn
tác động (giữ một aspect, sửa một loại lỗi, hoặc chọn phong cách), nên câu trả lời được chuyển
thành IntentProfile mà không cần VLM. Ý định người dùng ghi đè chẩn đoán của VLM.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .perception import _drop_preserved_defects
from .state import Defect, DiagnosisReport, IntentProfile, PreserveItem

MAX_QUESTIONS = 3
# Số câu hỏi về chỗ mơ hồ tối đa (câu còn lại dành cho phong cách)
MAX_AMBIGUITY_QUESTIONS = MAX_QUESTIONS - 1
USER_DEFECT_SEVERITY = 2

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

STYLE_LABELS: Dict[str, str] = {
    "natural": "Tự nhiên (chỉnh nhẹ tay)",
    "balanced": "Cân bằng",
    "vivid": "Đậm nét (rực, tương phản cao)",
}


class Choice(BaseModel):
    """Một lựa chọn trả lời, kèm tác động của nó lên ý định."""

    value: str
    label: str
    keep: Optional[str] = Field(default=None, description="Aspect cần giữ khi chọn")
    fix: Optional[str] = Field(default=None, description="Loại lỗi cần sửa khi chọn")
    style: Optional[str] = Field(default=None, description="Phong cách khi chọn")


class Question(BaseModel):
    """Một câu hỏi làm rõ ý định người dùng."""

    id: str
    text: str
    choices: List[Choice]
    default: str


def _has_defect(diagnosis: DiagnosisReport, *types: str, full_only: bool = False) -> bool:
    """Chẩn đoán có lỗi (severity >= 1) thuộc một trong các loại này không."""
    return any(
        defect.type in types and (not full_only or defect.region == "full")
        for defect in diagnosis.actionable_defects
    )


def _has_preserve(diagnosis: DiagnosisReport, *aspects: str) -> bool:
    """Chẩn đoán có đặc điểm cần giữ thuộc một trong các aspect này không."""
    return any(item.aspect in aspects for item in diagnosis.preserve)


def _mood_question(diagnosis: DiagnosisReport) -> Optional[Question]:
    """Ảnh/chủ thể tối: giữ không khí tối hay làm sáng."""
    backlit = _has_defect(diagnosis, "backlit_subject")
    dark = backlit or _has_defect(diagnosis, "underexposed")
    intentional = _has_preserve(diagnosis, "low_key", "silhouette")
    if not (dark or intentional):
        return None
    subject = "Chủ thể đang tối do ngược sáng" if backlit else "Ảnh đang khá tối"
    return Question(
        id="mood",
        text=f"{subject}. Bạn muốn giữ không khí tối (low-key, bóng đổ) hay làm sáng rõ?",
        choices=[
            Choice(value="keep_dark", label="Giữ không khí tối", keep="low_key"),
            Choice(
                value="brighten",
                label="Làm sáng rõ",
                fix="backlit_subject" if backlit else "underexposed",
            ),
        ],
        default="keep_dark" if intentional else "brighten",
    )


def _tone_question(diagnosis: DiagnosisReport, warm: bool) -> Optional[Question]:
    """Ảnh có tông ấm/lạnh: chủ ý cần giữ hay ám màu cần khử."""
    cast, aspect = ("color_cast_warm", "warm_tone") if warm else ("color_cast_cool", "cool_tone")
    if not (_has_defect(diagnosis, cast) or _has_preserve(diagnosis, aspect)):
        return None
    tone = "vàng/ấm" if warm else "xanh/lạnh"
    scene = "đèn vàng, hoàng hôn" if warm else "tuyết, giờ xanh"
    return Question(
        id="warm" if warm else "cool",
        text=(
            f"Ảnh đang có tông {tone}. Đó là không khí bạn muốn giữ ({scene}) "
            "hay là ám màu cần khử?"
        ),
        choices=[
            Choice(value="keep", label=f"Giữ tông {tone.split('/')[1]}", keep=aspect),
            Choice(value="fix", label=f"Khử ám {tone.split('/')[0]}", fix=cast),
        ],
        default="keep" if _has_preserve(diagnosis, aspect) else "fix",
    )


def _grain_question(diagnosis: DiagnosisReport) -> Optional[Question]:
    """Ảnh có hạt: làm mịn hay giữ hạt kiểu film."""
    if not (_has_defect(diagnosis, "noise") or _has_preserve(diagnosis, "film_grain")):
        return None
    return Question(
        id="grain",
        text="Ảnh có hạt nhiễu. Bạn muốn làm mịn, hay giữ hạt như ảnh film?",
        choices=[
            Choice(value="smooth", label="Làm mịn", fix="noise"),
            Choice(value="keep", label="Giữ hạt film", keep="film_grain"),
        ],
        default="keep" if _has_preserve(diagnosis, "film_grain") else "smooth",
    )


def style_question() -> Question:
    """Câu hỏi phong cách, luôn được hỏi."""
    return Question(
        id="style",
        text="Bạn thích kết quả theo phong cách nào?",
        choices=[
            Choice(value=style, label=label, style=style) for style, label in STYLE_LABELS.items()
        ],
        default="balanced",
    )


def build_questions(diagnosis: Optional[DiagnosisReport]) -> List[Question]:
    """
    Câu hỏi cho một chẩn đoán: tối đa MAX_AMBIGUITY_QUESTIONS câu về chỗ mơ hồ (theo thứ tự
    ưu tiên: không khí tối, tông ấm, hạt, tông lạnh), rồi câu phong cách. Tất định, nên sinh
    lại khi resume cho đúng cùng câu hỏi.
    """
    ambiguities: List[Question] = []
    if diagnosis is not None:
        candidates = (
            _mood_question(diagnosis),
            _tone_question(diagnosis, warm=True),
            _grain_question(diagnosis),
            _tone_question(diagnosis, warm=False),
        )
        ambiguities = [question for question in candidates if question is not None]
    return [*ambiguities[:MAX_AMBIGUITY_QUESTIONS], style_question()]


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
    ("style", "natural", r"tự nhiên|nhẹ nhàng|nhẹ tay"),
    ("style", "vivid", r"rực|đậm|nổi bật|sống động"),
    ("style", "balanced", r"cân bằng"),
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
    **{("style", style): Choice(value=style, label="", style=style) for style in STYLE_LABELS},
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
    questions: Sequence[Question], answers: Dict[str, str], notes: str = ""
) -> IntentProfile:
    """
    Chuyển câu trả lời (và ghi chú) thành IntentProfile. Câu trả lời chọn trực tiếp thắng
    ghi chú; giá trị không có trong lựa chọn của câu hỏi bị bỏ qua.
    """
    by_id = {question.id: question for question in questions}
    merged = {**parse_notes(notes), **{k: v for k, v in answers.items() if v}}
    intent = IntentProfile(notes=notes.strip())
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
        if choice.style:
            intent.style = choice.style
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
    return updated


def intent_prompt(intent: Optional[IntentProfile]) -> str:
    """Khối 'Ý ĐỊNH NGƯỜI DÙNG' cho prompt giai đoạn Plan; rỗng nếu không có ý định."""
    if intent is None:
        return ""
    lines: List[str] = []
    if intent.style:
        lines.append(f"- Phong cách mong muốn: {STYLE_LABELS[intent.style]}")
    if intent.keep:
        lines.append("- Muốn giữ: " + ", ".join(f"{p.aspect}@{p.region}" for p in intent.keep))
    if intent.fix:
        lines.append("- Muốn sửa: " + ", ".join(intent.fix))
    if intent.notes:
        lines.append(f"- Ghi chú: {intent.notes}")
    if not lines:
        return ""
    return "Ý ĐỊNH NGƯỜI DÙNG (ưu tiên cao nhất, ghi đè chẩn đoán):\n" + "\n".join(lines)
