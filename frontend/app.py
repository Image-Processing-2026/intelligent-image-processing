"""
Gradio UI Demo for Intelligent Image Processing.
Giao diện người dùng trực quan: so sánh Before/After, Timeline, lý do chẩn đoán VLM.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from dotenv import load_dotenv

if __name__ == "__main__" and not __package__:
    # Direct script launch puts frontend/, rather than the repo root, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.executor import execute_plan
from src.agent.graph import run_pipeline
from src.agent.memory import (
    DEFAULT_MEMORY_PATH,
    DEFAULT_USER,
    MEMORY_ENV,
    configure_case_memory,
    get_case_memory,
    remember_run,
)
from src.agent.planner import sanitize_actions
from src.agent.refine import refine
from src.agent.session import SessionNotFoundError, answer_session, start_session
from src.agent.state import DiagnosisReport, RegionOperation, TreatmentPlan
from src.agent.variants import treatment_recipe
from src.analyzer_evaluator.analyzer import analyze_image

_SEVERITY_LABELS = {0: "không đáng kể", 1: "nhẹ", 2: "rõ", 3: "nặng"}


def format_diagnosis(diagnosis: Any) -> str:
    """Định dạng chẩn đoán giai đoạn Perceive thành Markdown cho UI."""
    if diagnosis is None:
        return ""
    if isinstance(diagnosis, dict):
        diagnosis = DiagnosisReport.model_validate(diagnosis)

    lighting = f" · ánh sáng: {diagnosis.lighting}" if diagnosis.lighting else ""
    lines = [f"- **Cảnh:** `{diagnosis.scene_type}`{lighting} · nguồn: `{diagnosis.source}`"]
    if diagnosis.summary:
        lines.append(f"- **Chẩn đoán:** {diagnosis.summary}")
    for defect in diagnosis.defects:
        severity = _SEVERITY_LABELS.get(defect.severity, str(defect.severity))
        evidence = f" — {defect.evidence}" if defect.evidence else ""
        lines.append(
            f"  - Lỗi **`{defect.type}`** ở `{defect.region}` "
            f"({severity}, {defect.origin}){evidence}"
        )
    for item in diagnosis.preserve:
        reason = f" — {item.reason}" if item.reason else ""
        lines.append(f"  - Giữ **`{item.aspect}`** ở `{item.region}`{reason}")
    for region, measured in diagnosis.region_metrics.items():
        gap = (
            f", lệch {measured.brightness_vs_rest:+.0f} so với phần còn lại"
            if measured.brightness_vs_rest is not None
            else ""
        )
        lines.append(
            f"  - Đo vùng `{region}` ({measured.backend}): "
            f"sáng {measured.brightness_mean:.0f}/255{gap}"
        )
    return "\n".join(lines)


def process_interface(
    input_image: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray],
    max_iters: int,
) -> Tuple[Optional[np.ndarray], str, List[Tuple[np.ndarray, str]], str, str]:
    """
    Xử lý ảnh qua chu trình LangGraph khép kín và định dạng kết quả cho UI.

    Returns:
        output_img: Ảnh kết quả sau điều trị (After).
        status_md: Markdown trạng thái, quyết định, số vòng lặp.
        gallery_images: Danh sách (ảnh, chú thích) cho Timeline.
        reasoning_md: Diễn giải chẩn đoán lâm sàng của VLM qua các vòng.
        details_json: Chuỗi JSON chi tiết kỹ thuật.
    """
    if input_image is None:
        return (
            None,
            "⚠️ **Vui lòng tải lên ảnh đầu vào để bắt đầu quá trình chẩn đoán & xử lý.**",
            [],
            "*Chưa có dữ liệu.*",
            "{}",
        )

    is_synthetic = ground_truth is not None
    result_state = run_pipeline(
        image=input_image,
        ground_truth=ground_truth,
        is_synthetic=is_synthetic,
        max_iterations=int(max_iters),
    )
    return _format_result(input_image, is_synthetic, max_iters, result_state)


def _format_result(
    input_image: np.ndarray,
    is_synthetic: bool,
    max_iters: int,
    result_state: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str, List[Tuple[np.ndarray, str]], str, str]:
    """Định dạng trạng thái cuối của pipeline thành 5 đầu ra của tab Chẩn đoán & Lịch sử."""
    output_img = result_state.get("current_image")
    decision = result_state.get("decision", "UNKNOWN")
    iters = max(0, result_state.get("iteration", 1) - 1)
    history = result_state.get("history", [])
    intermediates = result_state.get("intermediate_images", [])
    eval_result = result_state.get("evaluation_result", {})

    # 1. Trạng thái & Quyết định cuối
    decision_map = {
        "SHIP": ("", "SHIP — Đạt ngưỡng đánh giá của hệ thống"),
        "STOP_BEST_EFFORT": ("", "STOP_BEST_EFFORT — Đã dừng — xem kết quả và lịch sử"),
        "RE_PROCESS": ("🔄", "RE_PROCESS — Tiếp tục lặp"),
    }
    emoji, label = decision_map.get(decision, ("❓", decision))

    status_lines = [
        f"### {emoji} Quyết định: **{label}**",
        f"- **Số vòng lặp thực hiện:** {iters} / {int(max_iters)}",
    ]

    if is_synthetic and eval_result:
        psnr = eval_result.get("psnr", "N/A")
        ssim = eval_result.get("ssim", "N/A")
        status_lines.append(f"- **Chỉ số tham chiếu:** PSNR = `{psnr} dB` | SSIM = `{ssim}`")
    elif eval_result:
        score = eval_result.get("estimated_quality_score", "N/A")
        status_lines.append(f"- **Điểm đánh giá ước lượng:** `{score} / 100`")

    status_md = "\n".join(status_lines)

    # 2. Timeline Gallery (Before → Iteration 1 → ... → Final)
    gallery_images: List[Tuple[np.ndarray, str]] = []
    gallery_images.append((input_image, "1. Ảnh gốc (Before)"))
    for i, thumb in enumerate(intermediates):
        gallery_images.append((thumb, f"Vòng {i + 1}"))
    if output_img is not None:
        gallery_images.append((output_img, f"Kết quả cuối ({decision})"))

    # 3. Diễn giải chẩn đoán (Clinical Reasoning)
    reasoning_sections: List[str] = []
    for item in history:
        it = getattr(item, "iteration", None) or (
            item.get("iteration") if isinstance(item, dict) else "?"
        )
        plan = getattr(item, "plan", None) or (item.get("plan") if isinstance(item, dict) else None)
        item_decision = getattr(item, "decision", "") or (
            item.get("decision", "") if isinstance(item, dict) else ""
        )

        reasoning = ""
        actions: List[Any] = []
        knowledge: List[str] = []
        if plan:
            reasoning = getattr(plan, "reasoning", "") or (
                plan.get("reasoning", "") if isinstance(plan, dict) else ""
            )
            actions = getattr(plan, "actions", []) or (
                plan.get("actions", []) if isinstance(plan, dict) else []
            )
            knowledge = getattr(plan, "knowledge", []) or (
                plan.get("knowledge", []) if isinstance(plan, dict) else []
            )

        actions_text = []
        for a in actions:
            op = getattr(a, "operation", "") or (
                a.get("operation", "") if isinstance(a, dict) else ""
            )
            reg = getattr(a, "region_id", "") or (
                a.get("region_id", "") if isinstance(a, dict) else ""
            )
            params = getattr(a, "parameters", {}) or (
                a.get("parameters", {}) if isinstance(a, dict) else {}
            )
            params_str = ", ".join(f"`{k}={v}`" for k, v in params.items())
            actions_text.append(f"  - Thao tác **`{op}`** trên vùng `{reg}` ({params_str})")

        actions_block = "\n".join(actions_text) if actions_text else "  *(Không có thao tác)*"

        diagnosis = getattr(item, "diagnosis", None) or (
            item.get("diagnosis") if isinstance(item, dict) else None
        )
        diagnosis_md = format_diagnosis(diagnosis)

        reasoning_sections.append(
            f"#### 🔄 Vòng lặp {it} (Quyết định: `{item_decision}`)\n"
            + (f"{diagnosis_md}\n" if diagnosis_md else "")
            + f"- **Lập kế hoạch:** {reasoning}\n"
            + (
                f"- **Tri thức tham khảo:** {', '.join(f'`{k}`' for k in knowledge)}\n"
                if knowledge
                else ""
            )
            + f"- **Phác đồ thực thi:**\n{actions_block}"
        )

    reasoning_md = (
        "\n\n---\n\n".join(reasoning_sections)
        if reasoning_sections
        else "*Chưa có dữ liệu lịch sử.*"
    )

    # 4. JSON chi tiết kỹ thuật
    history_serialized = [
        item.model_dump()
        if hasattr(item, "model_dump")
        else (item if isinstance(item, dict) else str(item))
        for item in history
    ]
    details: Dict[str, Any] = {
        "final_decision": decision,
        "total_iterations": iters,
        "evaluation_result": eval_result,
        "final_technical_metrics": (
            analyze_image(output_img).model_dump() if output_img is not None else {}
        ),
        "history": history_serialized,
    }
    details_json = json.dumps(details, indent=2, ensure_ascii=False)

    return output_img, status_md, gallery_images, reasoning_md, details_json


VariantActions = Dict[str, List[Dict[str, Any]]]


RECOMMENDATION_REASONS = {
    "intent": "theo phong cách bạn đã chọn",
    "preference": "theo phong cách bạn hay chọn trước đây (bộ nhớ)",
}


def _variant_outputs(
    result_state: Dict[str, Any],
) -> Tuple[List[Tuple[np.ndarray, str]], str, Any, VariantActions]:
    """
    Đầu ra phần "Chọn phiên bản": gallery preview, mô tả xếp hạng, radio chọn (mặc định là
    phiên bản đề xuất) và phác đồ của từng phiên bản (giữ trong gr.State để render full-res).
    """
    variants = result_state.get("variants") or []
    if not variants:
        return [], "*Chỉ có một kết quả (số phiên bản = 1).*", gr.update(choices=[], value=None), {}

    recommended = result_state.get("recommended_variant")
    source = result_state.get("variant_ranking_source")
    source_text = "VLM critic so sánh trực quan" if source == "critic" else "điểm Module 1"
    gallery: List[Tuple[np.ndarray, str]] = []
    lines = [f"**Xếp hạng theo:** {source_text}"]
    reason = RECOMMENDATION_REASONS.get(result_state.get("recommended_reason") or "")
    if reason:
        lines.append(f"**Đề xuất:** {reason}")
    for variant in variants:
        star = " ⭐ đề xuất" if variant.id == recommended else ""
        gallery.append((variant.image, f"#{variant.rank} {variant.label}{star}"))
        note = f" — _{variant.critic_note}_" if variant.critic_note else ""
        lines.append(
            f"{variant.rank}. **{variant.label}**{star}: {variant.description} "
            f"(điểm `{variant.quality_score}`){note}"
        )
    choices = [(variant.label, variant.id) for variant in variants]
    actions = {
        variant.id: [action.model_dump(mode="json") for action in variant.actions]
        for variant in variants
    }
    return (
        gallery,
        "\n".join(lines),
        gr.update(choices=choices, value=recommended if recommended in actions else variants[0].id),
        actions,
    )


def process_with_variants(
    input_image: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray],
    max_iters: int,
    num_variants: int,
) -> Tuple[Any, ...]:
    """
    Chạy pipeline một lần, trả về 5 đầu ra của process_interface cộng 4 đầu ra của phần
    chọn phiên bản (gallery, mô tả, radio, phác đồ) và id ca trong bộ nhớ (None nếu tắt).
    """
    if input_image is None:
        empty = process_interface(None, None, max_iters)
        return (*empty, [], "*Chưa có phiên bản.*", gr.update(choices=[], value=None), {}, None)
    is_synthetic = ground_truth is not None
    result_state = run_pipeline(
        image=input_image,
        ground_truth=ground_truth,
        is_synthetic=is_synthetic,
        max_iterations=int(max_iters),
        num_variants=int(num_variants),
    )
    return (
        *_format_result(input_image, is_synthetic, max_iters, result_state),
        *_variant_outputs(result_state),
        remember_run(result_state),
    )


def _record_choice(case_id: Optional[str], variant_id: str) -> None:
    """Ghi phiên bản người dùng chọn vào bộ nhớ (nếu bộ nhớ bật và có ca)."""
    memory = get_case_memory()
    if memory is not None and case_id:
        memory.record_choice(case_id, variant_id)


def render_chosen_variant(
    input_image: Optional[np.ndarray],
    choice: Optional[str],
    variant_actions: Optional[VariantActions],
    case_id: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Render phiên bản đã chọn ở độ phân giải gốc: áp lại đúng phác đồ của phiên bản (giữ
    nguyên thứ tự, đã kẹp tham số) lên ảnh đầu vào, và ghi lựa chọn vào bộ nhớ.
    """
    if input_image is None or not choice or not variant_actions or choice not in variant_actions:
        return None, "⚠️ **Hãy chạy xử lý với số phiên bản > 1 rồi chọn một phiên bản.**"
    actions = sanitize_actions(
        [RegionOperation.model_validate(raw) for raw in variant_actions[choice]]
    )
    rendered = execute_plan(input_image, TreatmentPlan(reasoning=choice, actions=actions))
    _record_choice(case_id, choice)
    height, width = rendered.shape[:2]
    return rendered, f"✅ Đã xuất phiên bản **{choice}** ở độ phân giải gốc ({width}×{height})."


# ==================== Trò chuyện với Doctor (Phase 4) ====================
ChatHistory = List[Dict[str, str]]
QUESTION_SLOTS = 3


def _question_updates(questions: List[Dict[str, Any]]) -> List[Any]:
    """Cập nhật QUESTION_SLOTS ô radio theo danh sách câu hỏi (ô thừa bị ẩn)."""
    updates: List[Any] = []
    for slot in range(QUESTION_SLOTS):
        if slot < len(questions):
            question = questions[slot]
            updates.append(
                gr.update(
                    visible=True,
                    label=question["text"],
                    choices=[(c["label"], c["value"]) for c in question["choices"]],
                    value=question["default"],
                )
            )
        else:
            updates.append(gr.update(visible=False, choices=[], value=None))
    return updates


def chat_start(
    input_image: Optional[np.ndarray], history: Optional[ChatHistory]
) -> Tuple[Any, ...]:
    """
    Bắt đầu phiên: phân tích, chẩn đoán rồi hỏi ý định.

    Returns:
        (lịch sử chat, session_id, câu hỏi (dict), 3 cập nhật radio).
    """
    history = list(history or [])
    if input_image is None:
        history.append({"role": "assistant", "content": "Hãy tải một ảnh lên trước nhé."})
        return (history, None, [], *_question_updates([]))
    result = start_session(input_image)
    questions = [question.model_dump() for question in result.questions]
    summary = result.diagnosis.summary if result.diagnosis else ""
    history.append(
        {
            "role": "assistant",
            "content": f"🩺 **Chẩn đoán:** {summary}\n\nTrước khi xử lý, mình muốn hỏi bạn "
            f"{len(questions)} câu để chỉnh đúng ý bạn (chọn ở bên dưới).",
        }
    )
    return (history, result.session_id, questions, *_question_updates(questions))


def chat_answer(
    session_id: Optional[str],
    questions: Optional[List[Dict[str, Any]]],
    answer_1: Optional[str],
    answer_2: Optional[str],
    answer_3: Optional[str],
    notes: str,
    history: Optional[ChatHistory],
) -> Tuple[Any, ...]:
    """
    Gửi câu trả lời, chạy tiếp pipeline.

    Returns:
        (lịch sử chat, ảnh kết quả, gallery phiên bản, cập nhật radio phiên bản,
        phác đồ từng phiên bản, phác đồ hiện tại, id ca trong bộ nhớ).
    """
    history = list(history or [])
    empty = (None, [], gr.update(choices=[], value=None), {}, [], None)
    if not session_id:
        history.append({"role": "assistant", "content": "Hãy bấm 'Phân tích & hỏi ý' trước."})
        return (history, *empty)
    questions = questions or []
    raw = (answer_1, answer_2, answer_3)
    answers = {q["id"]: value for q, value in zip(questions, raw) if value}
    chosen = []
    for question in questions:
        value = answers.get(question["id"])
        label = next((c["label"] for c in question["choices"] if c["value"] == value), None)
        if label:
            chosen.append(label)
    user_text = "; ".join(chosen) + (f". {notes.strip()}" if notes.strip() else "")
    history.append({"role": "user", "content": user_text or "(dùng lựa chọn mặc định)"})
    try:
        result = answer_session(session_id, answers, notes)
    except (SessionNotFoundError, ValueError):
        history.append(
            {"role": "assistant", "content": "Phiên đã kết thúc. Hãy bấm 'Phân tích & hỏi ý' lại."}
        )
        return (history, *empty)
    state = result.state or {}
    gallery, _, choice_update, variant_actions = _variant_outputs(state)
    recommended = state.get("recommended_variant")
    current = variant_actions.get(recommended) if recommended else None
    if current is None:
        current = [
            action.model_dump(mode="json") for action in treatment_recipe(state.get("history", []))
        ]
    label = next((v.label for v in state.get("variants") or [] if v.id == recommended), None)
    message = f"✅ Xong ({state.get('decision')})."
    if label:
        reason = RECOMMENDATION_REASONS.get(state.get("recommended_reason") or "")
        why = f" ({reason})" if reason else ""
        message += f" Mình đề xuất bản **{label}**{why}; bạn có thể chọn bản khác bên dưới."
    message += " Nếu chưa ưng, hãy góp ý (ví dụ: 'da hơi vàng', 'trời gắt quá')."
    history.append({"role": "assistant", "content": message})
    output = state.get("current_image")
    if current and recommended and recommended != "balanced":
        output = _render_actions(state["original_image"], current)
    case_id = remember_run(state, state.get("user_id") or DEFAULT_USER)
    return (history, output, gallery, choice_update, variant_actions, current, case_id)


def _render_actions(image: np.ndarray, actions: List[Dict[str, Any]]) -> np.ndarray:
    """Render một phác đồ (dạng dict) lên ảnh gốc, giữ nguyên thứ tự thao tác."""
    parsed = sanitize_actions([RegionOperation.model_validate(raw) for raw in actions])
    return execute_plan(image, TreatmentPlan(reasoning="render", actions=parsed))


def chat_select_variant(
    input_image: Optional[np.ndarray],
    choice: Optional[str],
    variant_actions: Optional[VariantActions],
    case_id: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    """
    Chọn một phiên bản: render full-res, lấy phác đồ của nó làm phác đồ hiện tại và ghi
    lựa chọn vào bộ nhớ.
    """
    if input_image is None or not choice or not variant_actions or choice not in variant_actions:
        return None, []
    actions = variant_actions[choice]
    _record_choice(case_id, choice)
    return _render_actions(input_image, actions), actions


def chat_feedback(
    input_image: Optional[np.ndarray],
    current_actions: Optional[List[Dict[str, Any]]],
    feedback: str,
    history: Optional[ChatHistory],
    case_id: Optional[str] = None,
) -> Tuple[Any, ...]:
    """
    Chỉnh theo góp ý trên phác đồ hiện tại; góp ý hiểu được được ghi vào bộ nhớ.

    Returns:
        (lịch sử chat, ảnh mới, phác đồ mới, ô góp ý đã xóa).
    """
    history = list(history or [])
    if input_image is None or not feedback.strip():
        return (history, gr.update(), current_actions or [], feedback)
    history.append({"role": "user", "content": feedback})
    actions = [RegionOperation.model_validate(raw) for raw in current_actions or []]
    result = refine(input_image, actions, feedback)
    if result.adjustments:
        reply = "🔧 Đã chỉnh: " + "; ".join(result.notes) + "."
        memory = get_case_memory()
        if memory is not None and case_id:
            memory.record_feedback(case_id, feedback, result.adjustments)
    else:
        reply = (
            "Mình chưa hiểu cần chỉnh gì. Bạn thử nói cụ thể hơn, ví dụ 'tối quá', "
            "'da hơi vàng', 'màu rực quá', 'còn nhiễu'."
        )
    history.append({"role": "assistant", "content": reply})
    new_actions = [action.model_dump(mode="json") for action in result.actions]
    return (history, result.image, new_actions, "")


LOGGER = logging.getLogger(__name__)
CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
WAITING = "Đang xử lý, thời gian phụ thuộc ảnh và số vòng"
ERROR = "Không thể hoàn tất xử lý. Vui lòng thử lại."


def processing_reset():
    """Clear transient results only; never delete persisted case memory."""
    return (
        None,
        "Chọn ảnh rồi bấm Phân tích & xử lý.",
        [],
        "",
        "{}",
        [],
        "Chưa có phiên bản.",
        gr.update(choices=[], value=None),
        {},
        None,
    )


def chat_reset():
    return (
        [],
        None,
        [],
        *_question_updates([]),
        None,
        [],
        gr.update(choices=[], value=None),
        {},
        [],
        None,
        "",
        "",
    )


def ui_job(callback, args, outputs, controls, empty, status, gates):
    """Yield real waiting state, then results and unlocked controls even on failure."""
    # Keep component metadata (radio choices/visibility) while a job is running.
    # Results are replaced on success and fully cleared on failure.
    waiting = {}
    for control in controls:
        previous = waiting.get(control)
        waiting[control] = gr.update(
            **(
                {k: v for k, v in previous.items() if k != "__type__"}
                if isinstance(previous, dict) and previous.get("__type__") == "update"
                else {}
            ),
            interactive=False,
        )
    yield {**waiting, status: WAITING}
    try:
        result = callback(*args)
        values = dict(zip(outputs, result))
        message = "Hoàn tất."
    except Exception:
        LOGGER.exception("UI operation failed")
        values = dict(zip(outputs, empty))
        message = ERROR
    unlocked = {control: gr.update(interactive=True) for control in controls}
    unlocked.update(
        {control: gr.update(interactive=bool(check(values))) for control, check in gates.items()}
    )
    for control, update in unlocked.items():
        previous = values.get(control)
        if isinstance(previous, dict) and previous.get("__type__") == "update":
            values[control] = {**previous, **update}
        elif control in values:
            values[control] = gr.update(value=previous, interactive=update["interactive"])
        else:
            values[control] = update
    yield {**values, status: message}


def create_app() -> gr.Blocks:
    """Two-tab demo with isolated image/session state and serialized UI jobs."""
    with gr.Blocks(title="AI Image Doctor", analytics_enabled=False, fill_width=True) as demo:
        with gr.Column(elem_id="doctor-app"):
            gr.Markdown(
                "# AI Image Doctor\n**Bản demo báo cáo tiến độ**\n\n"
                "Chẩn đoán và cải thiện ảnh bằng AI kết hợp xử lý ảnh cổ điển"
            )
            with gr.Tabs():
                with gr.TabItem("Xử lý ảnh"):
                    with gr.Row():
                        with gr.Column(min_width=280, elem_classes="doctor-card"):
                            gr.Markdown("### 1. Chọn ảnh")
                            in_img = gr.Image(label="Ảnh gốc", type="numpy", height=300)
                            samples = [
                                ("Thiếu sáng", "coffee_underexposed.jpg"),
                                ("Nhiễu", "astro_noisy.jpg"),
                                ("Ám màu", "cat_warm_cast.jpg"),
                            ]
                            sample_buttons = []
                            with gr.Row():
                                for label, name in samples:
                                    path = Path(__file__).resolve().parents[1] / "data/real" / name
                                    if path.is_file():
                                        sample_buttons.append(
                                            (gr.Button(label, size="sm", min_width=0), path)
                                        )
                            with gr.Accordion("Tuỳ chọn xử lý", open=False):
                                max_iter_slider = gr.Slider(
                                    1, 5, value=3, step=1, label="Số vòng tối đa"
                                )
                                variants_slider = gr.Slider(
                                    1, 3, value=3, step=1, label="Số phiên bản"
                                )
                                gt_img = gr.Image(
                                    label="Ảnh chuẩn tương ứng (ground truth)", type="numpy"
                                )
                                gr.Markdown(
                                    "Chỉ dùng khi có ảnh chuẩn tương ứng để đánh giá thử nghiệm"
                                )
                            with gr.Row():
                                btn_run = gr.Button(
                                    "Phân tích & xử lý", variant="primary", min_width=0
                                )
                                btn_reset = gr.Button("Làm mới", min_width=0)
                        with gr.Column(min_width=280, elem_classes="doctor-card"):
                            gr.Markdown("### 2. Xem kết quả")
                            out_img = gr.Image(
                                label="Ảnh sau xử lý",
                                type="numpy",
                                height=300,
                                interactive=False,
                                buttons=["download", "fullscreen"],
                            )
                            status_box = gr.Markdown(processing_reset()[1])
                            job_status = gr.Markdown("")
                    with gr.Accordion("Các phiên bản ảnh", open=False) as variants_panel:
                        variants_gallery = gr.Gallery(
                            label="Các phiên bản", columns=3, object_fit="contain"
                        )
                        variants_box = gr.Markdown("Chưa có phiên bản.")
                        variant_choice = gr.Radio(label="Phiên bản đã chọn", choices=[])
                        btn_render = gr.Button("Áp dụng phiên bản đã chọn", interactive=False)
                    variant_actions, case_state = gr.State({}), gr.State(None)
                    gr.Markdown("### 3. Quá trình xử lý")
                    gallery = gr.Gallery(
                        label="Timeline qua từng vòng", columns=4, object_fit="contain"
                    )
                    reasoning_box = gr.Markdown("")
                    with gr.Accordion("Dữ liệu kỹ thuật (JSON)", open=False):
                        details_box = gr.Code(value="{}", language="json", label="Chi tiết JSON")
                with gr.TabItem("Trò chuyện & tinh chỉnh"):
                    gr.Markdown("Chọn ảnh → Phân tích và trả lời câu hỏi → Góp ý để tinh chỉnh")
                    with gr.Row():
                        with gr.Column(min_width=280, elem_classes="doctor-card"):
                            chat_img = gr.Image(label="Ảnh của bạn", type="numpy", height=300)
                            btn_start = gr.Button("1. Phân tích & hỏi ý", variant="primary")
                            question_radios = [
                                gr.Radio(visible=False, choices=[]) for _ in range(QUESTION_SLOTS)
                            ]
                            notes_box = gr.Textbox(
                                label="Mong muốn thêm", placeholder="Giữ tông ấm, làm rõ khuôn mặt"
                            )
                            btn_answer = gr.Button("2. Xử lý theo ý tôi", interactive=False)
                        with gr.Column(min_width=280, elem_classes="doctor-card"):
                            chatbot = gr.Chatbot(label="AI Image Doctor", height=360)
                            chat_out = gr.Image(
                                label="Kết quả", type="numpy", interactive=False, height=300
                            )
                    chat_status = gr.Markdown("")
                    chat_gallery = gr.Gallery(
                        label="Các phiên bản", columns=3, object_fit="contain"
                    )
                    chat_choice = gr.Radio(label="Chọn phiên bản", choices=[])
                    feedback_box = gr.Textbox(
                        label="3. Góp ý để chỉnh tiếp", placeholder="Da hơi vàng, trời gắt quá"
                    )
                    btn_feedback = gr.Button("Chỉnh theo góp ý", interactive=False)
                    session_state, questions_state = gr.State(None), gr.State([])
                    chat_variant_actions, current_actions, chat_case = (
                        gr.State({}),
                        gr.State([]),
                        gr.State(None),
                    )

            process_outputs = [
                out_img,
                status_box,
                gallery,
                reasoning_box,
                details_box,
                variants_gallery,
                variants_box,
                variant_choice,
                variant_actions,
                case_state,
            ]
            process_controls = [
                in_img,
                gt_img,
                max_iter_slider,
                variants_slider,
                btn_run,
                btn_reset,
                variant_choice,
                btn_render,
                *[b for b, _ in sample_buttons],
            ]

            def render_gate(v):
                return v.get(variant_choice, {}).get("value") in v.get(variant_actions, {})

            def reset_process():
                return (
                    *processing_reset(),
                    gr.update(interactive=False),
                    gr.update(open=False),
                    "",
                )

            reset_outputs = [*process_outputs, btn_render, variants_panel, job_status]
            for image in [in_img, gt_img]:
                image.change(reset_process, outputs=reset_outputs, queue=False)
            btn_reset.click(
                lambda: (None, None, *reset_process()),
                outputs=[in_img, gt_img, *reset_outputs],
                queue=False,
            )
            for button, path in sample_buttons:
                button.click(lambda path=path: str(path), outputs=in_img, queue=False)

            def process_job(*args):
                for update in ui_job(
                    process_with_variants,
                    args,
                    process_outputs,
                    process_controls,
                    processing_reset(),
                    job_status,
                    {btn_render: render_gate},
                ):
                    update[variants_panel] = gr.update(open=bool(update.get(variant_actions)))
                    yield update

            def render_job(*args):
                # Preserve variant state on success; discard every stale result on failure.
                yield {
                    **{c: gr.update(interactive=False) for c in process_controls},
                    job_status: WAITING,
                }
                try:
                    image, message = render_chosen_variant(*args)
                    values = {out_img: image, status_box: message}
                    valid = bool(args[1] and args[2] and args[1] in args[2])
                except Exception:
                    LOGGER.exception("Variant rendering failed")
                    values = dict(zip(process_outputs, processing_reset()))
                    message, valid = ERROR, False
                for control in process_controls:
                    previous = values.get(control)
                    values[control] = (
                        {**previous, "interactive": True}
                        if isinstance(previous, dict) and previous.get("__type__") == "update"
                        else gr.update(interactive=True)
                    )
                yield {
                    **values,
                    btn_render: gr.update(interactive=valid),
                    job_status: message,
                }

            variant_choice.input(
                lambda choice, actions: gr.update(interactive=bool(choice in (actions or {}))),
                [variant_choice, variant_actions],
                btn_render,
                queue=False,
            )
            common = dict(
                trigger_mode="once",
                concurrency_limit=1,
                concurrency_id="doctor-ui",
                show_progress="full",
            )
            btn_run.click(
                process_job,
                [in_img, gt_img, max_iter_slider, variants_slider],
                list(
                    dict.fromkeys([*process_outputs, *process_controls, job_status, variants_panel])
                ),
                **common,
            )
            btn_render.click(
                render_job,
                [in_img, variant_choice, variant_actions, case_state],
                list(dict.fromkeys([*process_outputs, *process_controls, job_status])),
                **common,
            )

            chat_outputs = [
                chatbot,
                session_state,
                questions_state,
                *question_radios,
                chat_out,
                chat_gallery,
                chat_choice,
                chat_variant_actions,
                current_actions,
                chat_case,
                notes_box,
                feedback_box,
            ]
            chat_controls = [
                chat_img,
                btn_start,
                btn_answer,
                btn_feedback,
                chat_choice,
                notes_box,
                feedback_box,
                *question_radios,
            ]
            gates = {
                btn_answer: lambda v: v.get(session_state),
                btn_feedback: lambda v: v.get(chat_out) is not None,
            }

            def reset_chat():
                return (
                    *chat_reset(),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "",
                )

            chat_img.change(
                reset_chat,
                outputs=[*chat_outputs, btn_answer, btn_feedback, chat_status],
                queue=False,
            )

            def chat_job(kind, *args):
                def operation(*args):
                    values = list(chat_reset())
                    if kind == "start":
                        values[:6] = chat_start(*args)
                    elif kind == "answer":
                        result = chat_answer(*args)
                        values[0] = result[0]
                        # answer_session consumes the session; require a new start for another answer.
                        values[1:6] = [None, [], *_question_updates([])]
                        values[6:12] = result[1:]
                    elif kind == "feedback":
                        result = chat_feedback(*args[:5])
                        values = list(args[5:])
                        values[0], values[6], values[10], values[13] = result
                    else:
                        values = list(args[4:])
                        values[6], values[10] = chat_select_variant(*args[:4])
                    return tuple(values)

                yield from ui_job(
                    operation, args, chat_outputs, chat_controls, chat_reset(), chat_status, gates
                )

            # Named generators are required for Gradio streaming detection.
            def answer_job(*args):
                yield from chat_job("answer", *args)

            def start_job(*args):
                yield from chat_job("start", *args)

            def feedback_job(*args):
                yield from chat_job("feedback", *args)

            def select_job(*args):
                yield from chat_job("select", *args)

            btn_start.click(
                start_job,
                [chat_img, chatbot],
                list(dict.fromkeys([*chat_outputs, *chat_controls, chat_status])),
                **common,
            )
            btn_answer.click(
                answer_job,
                [session_state, questions_state, *question_radios, notes_box, chatbot],
                list(dict.fromkeys([*chat_outputs, *chat_controls, chat_status])),
                **common,
            )
            btn_feedback.click(
                feedback_job,
                [chat_img, current_actions, feedback_box, chatbot, chat_case, *chat_outputs],
                list(dict.fromkeys([*chat_outputs, *chat_controls, chat_status])),
                **common,
            )
            chat_choice.input(
                select_job,
                [chat_img, chat_choice, chat_variant_actions, chat_case, *chat_outputs],
                list(dict.fromkeys([*chat_outputs, *chat_controls, chat_status])),
                **common,
            )
    return demo.queue()


if __name__ == "__main__":
    # Chỉ nạp .env khi chạy UI, không nạp khi test import module này
    load_dotenv(encoding="utf-8-sig")
    # Bộ nhớ ca bệnh (Phase 5): CASE_MEMORY_PATH rỗng → tắt
    configure_case_memory(os.getenv(MEMORY_ENV, str(DEFAULT_MEMORY_PATH)))
    app = create_app()
    app.launch(
        server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="teal"), css=CSS
    )
