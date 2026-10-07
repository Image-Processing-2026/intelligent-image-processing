"""
Xây dựng đồ thị trạng thái LangGraph (LangGraph Orchestration Graph).
Quản lý chu trình khép kín: Analyze -> Diagnose -> Plan -> Process -> Evaluate -> Decide.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Đảm bảo tương thích ngược nếu langchain phiên bản cũ/mới bị xung đột module attribute
try:
    import langchain

    if not hasattr(langchain, "debug"):
        langchain.debug = False
except ImportError:
    pass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send, interrupt

from src.analyzer_evaluator.analyzer import analyze_image
from src.analyzer_evaluator.no_reference_eval import evaluate_no_reference
from src.analyzer_evaluator.reference_eval import evaluate_reference

from .executor import execute_plan
from .imaging import downscale
from .intent import apply_intent, build_questions, intent_from_answers, intent_prompt
from .memory import DEFAULT_USER, experience_prompt, get_case_memory
from .perception import perceive
from .planner import action_region, apply_preserve_guard, validate_and_sort_plan
from .state import DiagnosisReport, DoctorState, HistoryItem, IntentProfile, TreatmentPlan
from .variants import (
    is_noisy,
    merged_preserve,
    rank_variants,
    render_variant,
    styles_for,
    treatment_recipe,
)
from .vlm_diagnostician import diagnose_and_plan

logger = logging.getLogger(__name__)

# Ngưỡng chất lượng cho quyết định dừng (Synthetic Benchmark)
PSNR_THRESHOLD = 28.0  # dB — Chất lượng phục hồi chấp nhận được
SSIM_THRESHOLD = 0.88  # Tương đồng cấu trúc chấp nhận được
PSNR_DEGRADATION = 1.5  # dB — Ngưỡng suy thoái cho phép giữa 2 vòng

# Ngưỡng chất lượng cho quyết định dừng (ảnh thực, điểm heuristic No-Reference 0–100)
REAL_TARGET_SCORE = 85.0  # Đạt mức này → SHIP, tránh xử lý quá tay
REAL_MIN_GAIN = 1.0  # Điểm tăng ít hơn mức này so với vòng trước → đã bão hòa, dừng
# Vòng vừa xử lý lỗi từ mức này trở lên → chưa SHIP theo điểm toàn cục, chẩn đoán lại để xác nhận
# (điểm toàn cục gần như không phản ánh lỗi trên vùng nhỏ như khuôn mặt ngược sáng)
VERIFY_SEVERITY = 2


# ---------------------------------------------------------
# Các Node thực thi trong đồ thị LangGraph
# ---------------------------------------------------------
def analyze_node(state: DoctorState) -> Dict[str, Any]:
    """Node 1: Phân tích các chỉ số kỹ thuật của bức ảnh hiện tại."""
    metrics = analyze_image(state["current_image"])
    return {"technical_metrics": metrics.model_dump()}


def perceive_node(state: DoctorState) -> Dict[str, Any]:
    """Node 2: Giai đoạn Perceive — chẩn đoán cảnh, lỗi theo vùng và điều cần giữ."""
    diagnosis = perceive(
        image=state["current_image"],
        metrics=state["technical_metrics"],
        iteration=state["iteration"],
        history=state.get("history", []),
        original_image=state.get("original_image"),
    )
    # Ý định người dùng (đã hỏi ở vòng 1) ghi đè chẩn đoán của mọi vòng
    return {"diagnosis": apply_intent(diagnosis, state.get("intent"))}


def clarify_node(state: DoctorState) -> Dict[str, Any]:
    """
    Node hỏi ý định (Phase 4): chỉ ở vòng 1, khi chạy tương tác và chưa có ý định.
    interrupt() dừng graph và trả câu hỏi cho người dùng; khi resume với
    {"answers": {...}, "notes": "..."} node chạy lại từ đầu, nên phần dựng câu hỏi phải
    tất định (build_questions đáp ứng điều này).
    """
    if not state.get("interactive") or state.get("intent") is not None or state["iteration"] > 1:
        return {}
    diagnosis = state.get("diagnosis")
    questions = build_questions(diagnosis)
    response = interrupt(
        {
            "questions": [question.model_dump() for question in questions],
            "diagnosis": diagnosis.model_dump(mode="json") if diagnosis is not None else None,
        }
    )
    response = response if isinstance(response, dict) else {}
    intent = intent_from_answers(
        questions, dict(response.get("answers") or {}), str(response.get("notes") or "")
    )
    updated = apply_intent(diagnosis, intent) if diagnosis is not None else None
    return {"intent": intent, "diagnosis": updated}


def plan_treatment(
    image: np.ndarray,
    metrics: Dict[str, Any],
    diagnosis: Optional[DiagnosisReport],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,
    intent: Optional[IntentProfile] = None,
    experience_text: str = "",
) -> TreatmentPlan:
    """
    Giai đoạn Plan: lập kế hoạch từ chẩn đoán, chuẩn hóa và áp preserve guard.
    experience_text: kinh nghiệm từ ca tương tự trong bộ nhớ (chỉ dùng cho prompt VLM).
    Ý định người dùng đã được áp vào chẩn đoán; ở đây chỉ đưa thêm vào prompt VLM.
    Chẩn đoán không còn lỗi nào (severity >= 1) → plan rỗng, không gọi VLM lần hai.
    """
    if diagnosis is not None and not diagnosis.actionable_defects:
        return TreatmentPlan(
            iteration=iteration,
            reasoning=f"Chẩn đoán không còn lỗi cần xử lý. {diagnosis.summary}".strip(),
            actions=[],
            source=diagnosis.source,
        )
    plan = diagnose_and_plan(
        image=image,
        metrics=metrics,
        iteration=iteration,
        history=history,
        original_image=original_image,
        diagnosis=diagnosis,
        intent_text=intent_prompt(intent),
        experience_text=experience_text,
    )
    validated_plan = validate_and_sort_plan(plan)
    if diagnosis is not None:
        validated_plan = apply_preserve_guard(validated_plan, diagnosis.preserve)
    return validated_plan


def _experience(state: DoctorState) -> Tuple[str, List[str]]:
    """
    Kinh nghiệm từ bộ nhớ ca bệnh cho vòng 1 (các vòng sau đã có phản hồi của chính vòng
    trước): (khối prompt, id các ca đã dùng). Bộ nhớ tắt hoặc lỗi → không có kinh nghiệm.
    """
    memory = get_case_memory()
    if memory is None or state["iteration"] > 1:
        return "", []
    try:
        cases = memory.similar(
            state.get("diagnosis"),
            state["technical_metrics"],
            user_id=state.get("user_id") or DEFAULT_USER,
        )
    except Exception as exc:
        logger.warning("Case memory lookup failed: %s", exc)
        return "", []
    return experience_prompt(cases), [f"case:{case.id[:8]}" for case, _ in cases]


def diagnose_and_plan_node(state: DoctorState) -> Dict[str, Any]:
    """Node 3: Lập kế hoạch điều trị từ chẩn đoán của giai đoạn Perceive."""
    experience_text, case_ids = _experience(state)
    plan = plan_treatment(
        image=state["current_image"],
        metrics=state["technical_metrics"],
        diagnosis=state.get("diagnosis"),
        iteration=state["iteration"],
        history=state.get("history", []),
        original_image=state.get("original_image"),
        intent=state.get("intent"),
        experience_text=experience_text,
    )
    if case_ids and plan.source == "vlm":
        plan.knowledge = plan.knowledge + case_ids
    return {"treatment_plan": plan}


def process_node(state: DoctorState) -> Dict[str, Any]:
    """Node 4: Thực thi kế hoạch điều trị (Module 2 & Module 3)."""
    prev_img = state["current_image"].copy()
    if not state.get("treatment_plan") or not state["treatment_plan"].actions:
        return {"current_image": state["current_image"], "previous_image": prev_img}

    processed_img = execute_plan(state["current_image"], state["treatment_plan"])
    return {"current_image": processed_img, "previous_image": prev_img}


def evaluate_node(state: DoctorState) -> Dict[str, Any]:
    """Node 5: Đánh giá chất lượng sau khi xử lý (Module 1)."""
    is_syn = state.get("is_synthetic", False)
    gt = state.get("ground_truth_image")
    prev_img = (
        state["previous_image"]
        if state.get("previous_image") is not None
        else state["original_image"]
    )
    iteration = state.get("iteration", 1)

    if is_syn and gt is not None:
        eval_metrics = evaluate_reference(
            current_image=state["current_image"],
            ground_truth_image=gt,
            iteration=iteration,
            previous_image=prev_img,
        )
    else:
        eval_metrics = evaluate_no_reference(
            current_image=state["current_image"],
            previous_image=prev_img,
            iteration=iteration,
        )

    # Chuẩn hóa về dict thuần trước khi đưa vào state: DoctorState khai báo
    # evaluation_result là Dict, HistoryItem.eval_score và ProcessResponse
    # cũng yêu cầu dict (model thô gây ValidationError và không JSON-serializable).
    return {"evaluation_result": eval_metrics.model_dump()}


def _is_degraded_synthetic(eval_result: Dict[str, Any], history: List[HistoryItem]) -> bool:
    """Phát hiện suy thoái (Synthetic): PSNR vòng này thấp hơn vòng trước quá ngưỡng."""
    psnr = float(eval_result.get("psnr", 0.0))
    ssim = float(eval_result.get("ssim", 0.0))

    # Đã đạt chất lượng mục tiêu thì không coi là suy thoái
    if psnr >= PSNR_THRESHOLD and ssim >= SSIM_THRESHOLD:
        return False

    if history:
        prev_eval = history[-1].eval_score or {}
        prev_psnr = float(prev_eval.get("psnr", 0.0))
        return prev_psnr - psnr > PSNR_DEGRADATION
    return False


def _is_degraded_real(eval_result: Dict[str, Any], history: List[HistoryItem]) -> bool:
    """Phát hiện suy thoái (Real): Degradation Guard của Module 1 hoặc quality score giảm."""
    # Module 1 Degradation Guard (nhiễu bùng nổ hoặc xuất hiện cháy sáng)
    if eval_result.get("quality_improved") is False:
        return True

    if history:
        current_quality = float(eval_result.get("estimated_quality_score", 50.0))
        prev_eval = history[-1].eval_score or {}
        prev_quality = float(prev_eval.get("estimated_quality_score", 50.0))
        return current_quality < prev_quality
    return False


def _decide_synthetic(eval_result: Dict[str, Any], history: List[HistoryItem]) -> str:
    """Logic quyết định dành cho ảnh nhân tạo có Ground-Truth."""
    psnr = float(eval_result.get("psnr", 0.0))
    ssim = float(eval_result.get("ssim", 0.0))

    # Đã đạt chất lượng mục tiêu
    if psnr >= PSNR_THRESHOLD and ssim >= SSIM_THRESHOLD:
        return "SHIP"

    if _is_degraded_synthetic(eval_result, history):
        return "STOP_BEST_EFFORT"

    # Chưa đạt, còn dư vòng → tiếp tục
    return "RE_PROCESS"


def _decide_real(eval_result: Dict[str, Any], history: List[HistoryItem]) -> str:
    """
    Logic quyết định dành cho ảnh thực không có Ground-Truth.
    - Suy thoái → STOP_BEST_EFFORT (decide_node sẽ rollback).
    - Điểm chất lượng đạt REAL_TARGET_SCORE → SHIP.
    - Điểm tăng chưa tới REAL_MIN_GAIN so với vòng trước → bão hòa, STOP_BEST_EFFORT.
    - Còn lại → RE_PROCESS.
    """
    if _is_degraded_real(eval_result, history):
        return "STOP_BEST_EFFORT"

    score = eval_result.get("estimated_quality_score")
    if score is None:
        return "RE_PROCESS"
    score = float(score)
    if score >= REAL_TARGET_SCORE:
        return "SHIP"

    if history:
        prev_score = (history[-1].eval_score or {}).get("estimated_quality_score")
        if prev_score is not None and score - float(prev_score) < REAL_MIN_GAIN:
            return "STOP_BEST_EFFORT"

    # Chưa đạt, còn dư vòng → tiếp tục
    return "RE_PROCESS"


def _needs_verification(
    diagnosis: Optional[DiagnosisReport], plan: Optional[TreatmentPlan]
) -> bool:
    """
    True nếu kế hoạch vòng này đã tác động lên một lỗi severity >= VERIFY_SEVERITY
    (action trên đúng vùng của lỗi, hoặc trên toàn ảnh) → cần Perceive lại để xác nhận.
    Lỗi không có action nào nhắm tới (ví dụ ám xanh lá mà toolbox không sửa được) không
    chặn SHIP, vì chẩn đoán lại cũng không đổi được gì.
    """
    if diagnosis is None or plan is None:
        return False
    touched = {action_region(action) for action in plan.actions}
    return any(
        defect.severity >= VERIFY_SEVERITY and ("full" in touched or defect.region in touched)
        for defect in diagnosis.defects
    )


def _create_thumbnail(image: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Resize ảnh xuống thumbnail để tiết kiệm bộ nhớ."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size or max(h, w) == 0:
        return image.copy()
    scale = max_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decide_node(state: DoctorState) -> Dict[str, Any]:
    """Node 6: Đưa ra quyết định dừng lại (SHIP) hay thử lại (RE_PROCESS)."""
    iteration = state["iteration"]
    max_iters = state.get("max_iterations", 3)
    is_syn = state.get("is_synthetic", False)
    eval_result = state.get("evaluation_result") or {}
    history = list(state.get("history") or [])
    plan = state.get("treatment_plan") or TreatmentPlan(
        iteration=iteration, reasoning="No treatment plan provided", actions=[]
    )

    # Cập nhật lịch sử (Zero-Redundant Compute: tái sử dụng technical_metrics từ evaluate_node)
    metrics_after = eval_result.get("technical_metrics")
    if not metrics_after:
        metrics_after = analyze_image(state["current_image"]).model_dump()

    history_entry = HistoryItem(
        iteration=iteration,
        plan=plan,
        metrics_before=state.get("technical_metrics", {}),
        metrics_after=metrics_after,
        eval_score=eval_result,
        decision="PENDING",  # sẽ cập nhật bên dưới
        diagnosis=state.get("diagnosis"),
    )

    # ====== LOGIC RA QUYẾT ĐỊNH ======
    has_actions = bool(state.get("treatment_plan") and state["treatment_plan"].actions)

    # 0. Phát hiện suy thoái ở MỌI vòng (kể cả vòng cuối) trước khi xét giới hạn vòng lặp
    # Kế hoạch có thao tác nhưng không pixel nào đổi (mọi action bị executor bỏ qua, ví dụ
    # không tìm thấy vùng): không phải suy thoái, nhưng lặp lại cũng vô ích → dừng, không rollback
    previous_image = state.get("previous_image")
    unchanged = (
        has_actions
        and previous_image is not None
        and np.array_equal(state["current_image"], previous_image)
    )
    degraded = (
        has_actions
        and not unchanged
        and (
            _is_degraded_synthetic(eval_result, history)
            if is_syn
            else _is_degraded_real(eval_result, history)
        )
    )

    if degraded:
        decision = "STOP_BEST_EFFORT"

    elif unchanged:
        logger.warning(
            "Iteration %d changed no pixels (every action was skipped); stopping.", iteration
        )
        decision = "STOP_BEST_EFFORT"

    # 1. Kế hoạch rỗng → VLM cho rằng ảnh đã tốt
    elif not has_actions:
        decision = "SHIP"

    else:
        # 2. Phân nhánh Synthetic vs Real (xét trước giới hạn vòng để vòng cuối đạt
        #    mục tiêu vẫn được báo SHIP)
        decision = (
            _decide_synthetic(eval_result, history)
            if is_syn
            else _decide_real(eval_result, history)
        )
        # 3. Ảnh thực: SHIP theo điểm chỉ được chấp nhận khi chẩn đoán xác nhận lỗi rõ đã hết.
        #    Vòng cuối không còn cơ hội xác nhận → giữ SHIP (đã đạt điểm mục tiêu)
        if (
            decision == "SHIP"
            and not is_syn
            and iteration < max_iters
            and _needs_verification(state.get("diagnosis"), plan)
        ):
            decision = "RE_PROCESS"
        # 4. Giới hạn cứng: hết vòng lặp mà chưa đạt → dừng nỗ lực tốt nhất
        if decision == "RE_PROCESS" and iteration >= max_iters:
            decision = "STOP_BEST_EFFORT"

    # Rollback: trả về ảnh trước vòng xử lý này nếu vòng này làm chất lượng xấu đi
    final_image = state["current_image"]
    rolled_back = False
    if degraded:
        if previous_image is not None:
            final_image = previous_image
            rolled_back = True
            logger.warning(
                "Iteration %d degraded image quality; rolling back to the previous image.",
                iteration,
            )
        else:
            logger.warning(
                "Iteration %d degraded image quality but no previous image is available; "
                "keeping the current image.",
                iteration,
            )

    history_entry.decision = decision
    history_entry.rolled_back = rolled_back
    new_history = history + [history_entry]

    # Lưu ảnh trung gian dạng thumbnail sau mỗi vòng
    # (giữ ảnh thực tế vừa tạo ra làm bằng chứng, kể cả khi đã rollback)
    thumbnail = _create_thumbnail(state["current_image"])
    new_intermediates = list(state.get("intermediate_images") or []) + [thumbnail]

    return {
        "current_image": final_image,
        "rolled_back": rolled_back,
        "iteration": iteration + 1,
        "decision": decision,
        "history": new_history,
        "intermediate_images": new_intermediates,
    }


def should_continue(state: DoctorState) -> Union[str, List[Send]]:
    """
    Điều hướng sau decide: lặp lại, kết thúc, hoặc (num_variants > 1) tỏa ra các nhánh render
    phiên bản song song bằng Send, mỗi nhánh một phong cách.
    """
    if state["decision"] == "RE_PROCESS":
        return "re_process"
    style_ids = styles_for(state.get("num_variants", 1))
    intent = state.get("intent")
    if intent is not None and intent.style and intent.style not in style_ids:
        style_ids.append(intent.style)
    if len(style_ids) < 2:
        return "ship"
    history = state.get("history") or []
    payload = {
        "original": downscale(state["original_image"]),
        "recipe": treatment_recipe(history),
        "preserve": merged_preserve(history),
        "noisy": is_noisy(history),
    }
    return [Send("render_variant", {**payload, "style_id": style}) for style in style_ids]


def render_variant_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Node render một phong cách trên ảnh preview (chạy song song qua Send)."""
    variant = render_variant(
        payload["original"],
        payload["style_id"],
        payload["recipe"],
        payload["preserve"],
        payload["noisy"],
    )
    return {"variant_candidates": [variant]}


def rank_variants_node(state: DoctorState) -> Dict[str, Any]:
    """Node gom các phiên bản: bỏ trùng, xếp hạng (VLM critic hoặc điểm Module 1)."""
    history = state.get("history") or []
    first = history[0].diagnosis if history else None
    diagnosis = (
        first.model_copy(update={"preserve": merged_preserve(history)})
        if first is not None
        else None
    )
    ranked, recommended, source = rank_variants(
        downscale(state["original_image"]), state.get("variant_candidates") or [], diagnosis
    )
    # Ưu tiên: phong cách người dùng nói rõ > phong cách họ hay chọn > xếp hạng
    available = {variant.id for variant in ranked}
    reason = source
    intent = state.get("intent")
    memory = get_case_memory()
    if intent is not None and intent.style in available:
        recommended, reason = intent.style, "intent"
    elif memory is not None:
        try:
            preferred = memory.preferred_style(state.get("user_id") or DEFAULT_USER)
        except Exception as exc:
            logger.warning("Case memory preference lookup failed: %s", exc)
            preferred = None
        if preferred in available:
            recommended, reason = preferred, "preference"
    return {
        "variants": ranked,
        "recommended_variant": recommended,
        "variant_ranking_source": source,
        "recommended_reason": reason if ranked else None,
    }


# ---------------------------------------------------------
# Xây dựng và biên dịch đồ thị
# ---------------------------------------------------------
def build_doctor_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Khởi tạo StateGraph của LangGraph (checkpointer cần cho chạy tương tác/interrupt)."""
    workflow = StateGraph(DoctorState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("perceive", perceive_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("diagnose_and_plan", diagnose_and_plan_node)
    workflow.add_node("process", process_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("render_variant", render_variant_node)
    workflow.add_node("rank_variants", rank_variants_node)

    workflow.set_entry_point("analyze")

    workflow.add_edge("analyze", "perceive")
    workflow.add_edge("perceive", "clarify")
    workflow.add_edge("clarify", "diagnose_and_plan")
    workflow.add_edge("diagnose_and_plan", "process")
    workflow.add_edge("process", "evaluate")
    workflow.add_edge("evaluate", "decide")

    workflow.add_conditional_edges(
        "decide",
        should_continue,
        {"re_process": "analyze", "ship": END, "render_variant": "render_variant"},
    )
    # LangGraph chờ mọi nhánh Send xong rồi mới chạy rank_variants một lần
    workflow.add_edge("render_variant", "rank_variants")
    workflow.add_edge("rank_variants", END)

    return workflow.compile(checkpointer=checkpointer)


def run_pipeline(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    is_synthetic: bool = False,
    max_iterations: int = 3,
    num_variants: int = 1,
    intent: Optional[IntentProfile] = None,
    user_id: str = DEFAULT_USER,
) -> DoctorState:
    """
    Hàm giao tiếp ngoài để chạy toàn bộ chu trình xử lý ảnh.
    user_id: chủ của các ca trong bộ nhớ (kinh nghiệm và sở thích được tách theo người dùng).
    num_variants > 1 (tối đa 3): sau vòng lặp, sinh các phiên bản phong cách trên ảnh preview
    (state["variants"], đã xếp hạng); current_image vẫn là kết quả full-res của phác đồ.
    intent: ý định người dùng có sẵn (không hỏi). Chạy tương tác: xem src/agent/session.py.
    """
    app = build_doctor_graph()
    initial_state = initial_doctor_state(
        image, ground_truth, is_synthetic, max_iterations, num_variants, intent, user_id=user_id
    )
    return app.invoke(initial_state)


def initial_doctor_state(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    is_synthetic: bool = False,
    max_iterations: int = 3,
    num_variants: int = 1,
    intent: Optional[IntentProfile] = None,
    interactive: bool = False,
    user_id: str = DEFAULT_USER,
) -> DoctorState:
    """State khởi đầu của pipeline (dùng chung cho chạy thường và chạy tương tác)."""
    initial_state: DoctorState = {
        "original_image": image,
        "current_image": image.copy(),
        "previous_image": None,
        "ground_truth_image": ground_truth,
        "is_synthetic": is_synthetic,
        "iteration": 1,
        "max_iterations": max_iterations,
        "technical_metrics": {},
        "diagnosis": None,
        "treatment_plan": None,
        "evaluation_result": {},
        "history": [],
        "intermediate_images": [],
        "decision": "INITIALIZING",
        "rolled_back": False,
        "error_message": None,
        "num_variants": num_variants,
        "variant_candidates": [],
        "variants": [],
        "recommended_variant": None,
        "variant_ranking_source": None,
        "interactive": interactive,
        "intent": intent,
        "user_id": user_id,
        "recommended_reason": None,
    }
    return initial_state
