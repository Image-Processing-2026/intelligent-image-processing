"""
Xây dựng đồ thị trạng thái LangGraph (LangGraph Orchestration Graph).
Quản lý chu trình khép kín: Analyze -> Diagnose -> Plan -> Process -> Evaluate -> Decide.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

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
from langgraph.types import interrupt

from src.analyzer_evaluator.analyzer import analyze_image
from src.analyzer_evaluator.measurement.diff import (
    DefectChange,
    black_lift,
    compare,
    region_stats,
    side_effects,
)
from src.analyzer_evaluator.metadata import ImageMetadata
from src.analyzer_evaluator.no_reference_eval import evaluate_no_reference
from src.analyzer_evaluator.reference_eval import evaluate_reference

from .executor import execute_plan, image_cache_key, subject_cache_key
from .fusion import (
    backlit_progress,
    backlit_resolution_cause,
    measure_hypothesis,
    plan_contraindications,
)
from .intent import apply_intent, build_questions, intent_from_answers, intent_prompt
from .knowledge.retriever import face_located
from .memory import DEFAULT_USER, experience_prompt, get_case_memory
from .perception import _measure, perceive, resolve_subject_masks
from .planner import (
    action_region,
    apply_contraindications,
    apply_preserve_guard,
    dedupe_nested_regions,
    normalize_defect_ref,
    validate_and_sort_plan,
)
from .regions import canonical_region, is_person_region
from .review import derive_outcome, review_candidate, review_context_from_state, same_image
from .state import (
    DiagnosisReport,
    DoctorState,
    ExecutedAction,
    HistoryItem,
    IntentProfile,
    ReviewResult,
    SideEffectCheck,
    TreatmentPlan,
)
from .user_analysis import validate_edits
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
# Tác dụng phụ do xử lý (P7, measurement.diff) từ mức này → coi như suy thoái, rollback
SIDE_EFFECT_ROLLBACK_SEVERITY = 2


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
        metadata=state.get("image_metadata"),
    )
    # Ý định người dùng (đã hỏi ở vòng 1) ghi đè chẩn đoán của mọi vòng
    return {"diagnosis": apply_intent(diagnosis, state.get("intent"))}


def clarify_node(state: DoctorState) -> Dict[str, Any]:
    """
    Node hỏi ý định (Phase 4): chỉ ở vòng 1, khi chạy tương tác và chưa có ý định.
    interrupt() dừng graph và trả câu hỏi cho người dùng; khi resume với
    {"answers": {...}, "notes": "...", "edits": [...]} node chạy lại từ đầu, nên phần dựng câu
    hỏi phải tất định (build_questions đáp ứng điều này). edits đã được parse ở tầng API (không
    parse văn bản trong node); edit hỏng bị bỏ qua với warning.
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
        questions,
        dict(response.get("answers") or {}),
        str(response.get("notes") or ""),
        validate_edits(response.get("edits")),
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
    validated_plan = dedupe_nested_regions(validate_and_sort_plan(plan))
    if diagnosis is not None:
        validated_plan = apply_preserve_guard(validated_plan, diagnosis.preserve)
        validated_plan = apply_contraindications(validated_plan, plan_contraindications(diagnosis))
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
        return {
            "current_image": state["current_image"],
            "previous_image": prev_img,
            "executed_actions": [],
        }

    # Vùng Perceive đã định vị bằng bbox dùng lại mask đã đo (cache theo ảnh gốc của lượt chạy)
    diagnosis = state.get("diagnosis")
    report: List[ExecutedAction] = []
    processed_img = execute_plan(
        state["current_image"],
        state["treatment_plan"],
        subject_boxes=diagnosis.subject_boxes if diagnosis is not None else None,
        cache_key=image_cache_key(state["original_image"]),
        report=report,
    )
    return {
        "current_image": processed_img,
        "previous_image": prev_img,
        "executed_actions": report,
    }


def _evaluate_image(
    state: DoctorState, image: np.ndarray, previous: np.ndarray, iteration: int
) -> Dict[str, Any]:
    """
    Đánh giá một ảnh theo ADR-002: tham chiếu khi là synthetic có ground truth, ngược lại
    no-reference. Chuẩn hóa về dict thuần trước khi đưa vào state: DoctorState khai báo
    evaluation_result là Dict, HistoryItem.eval_score và ProcessResponse cũng yêu cầu dict
    (model thô gây ValidationError và không JSON-serializable).
    """
    gt = state.get("ground_truth_image")
    if state.get("is_synthetic", False) and gt is not None:
        eval_metrics = evaluate_reference(
            current_image=image,
            ground_truth_image=gt,
            iteration=iteration,
            previous_image=previous,
        )
    else:
        eval_metrics = evaluate_no_reference(
            current_image=image,
            previous_image=previous,
            iteration=iteration,
        )
    return eval_metrics.model_dump()


def evaluate_node(state: DoctorState) -> Dict[str, Any]:
    """Node 5: Đánh giá chất lượng sau khi xử lý (Module 1)."""
    prev_img = (
        state["previous_image"]
        if state.get("previous_image") is not None
        else state["original_image"]
    )
    iteration = state.get("iteration", 1)
    return {
        "evaluation_result": _evaluate_image(state, state["current_image"], prev_img, iteration)
    }


def _reference_met(eval_result: Dict[str, Any]) -> bool:
    """True nếu chỉ số tham chiếu (synthetic) đạt cả ngưỡng PSNR lẫn SSIM."""
    psnr, ssim = eval_result.get("psnr"), eval_result.get("ssim")
    return (
        psnr is not None
        and ssim is not None
        and float(psnr) >= PSNR_THRESHOLD
        and float(ssim) >= SSIM_THRESHOLD
    )


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


def _plan_targets(plan: TreatmentPlan) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    (vùng đã tác động, các lỗi 'type@region' mà action tự khai qua defect_ref). Action trên vùng
    con (face) khai lỗi của vùng cha (person) bằng defect_ref nên vẫn được theo dõi.
    """
    touched = {action_region(action) for action in plan.actions}
    refs: Set[Tuple[str, str]] = set()
    for action in plan.actions:
        ref = normalize_defect_ref(action.defect_ref)
        if ref is not None:
            defect_type, _, region = ref.partition("@")
            refs.add((defect_type, region))
    return touched, refs


def _is_targeted(
    defect_type: str, region: str, touched: Set[str], refs: Set[Tuple[str, str]]
) -> bool:
    """Lỗi được kế hoạch nhắm tới: action toàn ảnh, trên đúng vùng, hoặc khai qua defect_ref."""
    return "full" in touched or region in touched or (defect_type, region) in refs


def _needs_verification(
    diagnosis: Optional[DiagnosisReport], plan: Optional[TreatmentPlan]
) -> bool:
    """
    True nếu kế hoạch vòng này đã tác động lên một lỗi severity >= VERIFY_SEVERITY
    (action trên đúng vùng của lỗi, trên toàn ảnh, hoặc khai lỗi qua defect_ref)
    → cần Perceive lại để xác nhận.
    Lỗi không có action nào nhắm tới (ví dụ ám xanh lá mà toolbox không sửa được) không
    chặn SHIP, vì chẩn đoán lại cũng không đổi được gì.
    """
    if diagnosis is None or plan is None:
        return False
    touched, refs = _plan_targets(plan)
    return any(
        defect.severity >= VERIFY_SEVERITY
        and _is_targeted(defect.type, canonical_region(defect.region), touched, refs)
        for defect in diagnosis.defects
    )


def _face_mask(state: DoctorState) -> Optional[np.ndarray]:
    """Mask mặt trên ảnh gốc (cùng cache với Perceive/executor); không định vị được → None."""
    diagnosis = state.get("diagnosis")
    original = state.get("original_image")
    if diagnosis is None or original is None or not face_located(diagnosis):
        return None
    boxes = {canonical_region(name): box for name, box in diagnosis.subject_boxes.items()}
    masks = resolve_subject_masks(
        original, ["face"], boxes, subject_cache_key(image_cache_key(original), original.shape)
    )
    return masks["face"].mask if "face" in masks else None


def _effects_between(
    reference: np.ndarray, current: np.ndarray, face: Optional[np.ndarray]
) -> List[Dict[str, Any]]:
    """Tác dụng phụ (kể cả black_lift ngoài mặt) khi đi từ ảnh tham chiếu tới ảnh hiện tại."""
    found = side_effects(reference, current)
    lift = black_lift(reference, current, face)
    if lift is not None:
        found.append(lift)
    return [effect.model_dump() for effect in found]


def _side_effects_check(state: DoctorState) -> SideEffectCheck:
    """
    Tác dụng phụ của vòng vừa xử lý, đo hai chiều: ảnh trước vòng → hiện tại (previous, dùng để
    rollback) và ảnh gốc → hiện tại (original, chỉ ghi nhận). black_lift chỉ ghi nhận (severity
    bị chặn dưới ngưỡng rollback). Phép đo lỗi trả status 'unavailable', không bao giờ trả danh
    sách rỗng giả; không có ảnh nào để so thì 'not_run'.
    """
    previous = state.get("previous_image")
    original = state.get("original_image")
    current = state["current_image"]
    if previous is None and original is None:
        return SideEffectCheck(status="not_run")
    try:
        face = _face_mask(state)
        from_previous = _effects_between(previous, current, face) if previous is not None else []
        if original is None:
            from_original: List[Dict[str, Any]] = []
        elif previous is not None and (previous is original or np.array_equal(previous, original)):
            from_original = list(from_previous)  # vòng 1: hai chiều trùng nhau
        else:
            from_original = _effects_between(original, current, face)
    except Exception as exc:
        logger.warning("Side-effect analysis failed: %s", exc)
        return SideEffectCheck(status="unavailable", error=str(exc))
    return SideEffectCheck(status="ok", previous=from_previous, original=from_original)


def _targeted_defects(
    diagnosis: Optional[DiagnosisReport], plan: Optional[TreatmentPlan]
) -> List[Tuple[str, str]]:
    """
    Các lỗi (loại, vùng chuẩn hóa) toolbox sửa được mà kế hoạch vòng này nhắm tới: action trên
    đúng vùng của lỗi, trên toàn ảnh, hoặc khai lỗi qua defect_ref (cùng quy ước với
    _needs_verification).
    """
    if diagnosis is None or plan is None:
        return []
    touched, refs = _plan_targets(plan)
    targeted: List[Tuple[str, str]] = []
    for defect in diagnosis.actionable_defects:
        region = canonical_region(defect.region)
        if (defect.type, region) in targeted:
            continue
        if _is_targeted(defect.type, region, touched, refs):
            targeted.append((defect.type, region))
    return targeted


def _verify_backlit(
    changes: List[DefectChange],
    masks: Dict[str, Any],
    before: Any,
    after: Any,
    people: Set[str],
) -> None:
    """
    Kiểm chứng chéo ngược sáng ở chủ thể người bằng vùng mặt (R-06), sửa trực tiếp `changes`:
    ghi nguyên nhân hết lỗi (độ tối giảm hay khoảng cách nền-chủ thể < BACKLIT_MIN_GAP_L) và
    đổi 'resolved' thành 'unverified' nếu mặt không sáng lên tương ứng. Không có mask mặt thì
    không đổi gì.
    """
    if "face" not in masks:
        return
    regions = {name: subject.mask for name, subject in masks.items()}
    before_stats, after_stats = region_stats(before, regions), region_stats(after, regions)
    for change in changes:
        if change.type != "backlit_subject" or change.region not in people:
            continue
        if change.status == "resolved":
            change.cause = backlit_resolution_cause(change.region, after_stats)
        progress = backlit_progress(change.region, before_stats, after_stats)
        if progress is None:
            continue
        change.detail = progress.detail
        if change.status == "resolved" and not progress.consistent:
            logger.warning(
                "Backlit '%s' reads resolved but the face did not follow.", change.region
            )
            change.status = "unverified"


def _track_defects(state: DoctorState, plan: TreatmentPlan) -> List[DefectChange]:
    """
    Trạng thái (resolved/improved/unchanged/worse/unverified) của các lỗi mục tiêu so với ảnh
    gốc, đo trên đúng vùng của lỗi bằng cùng mask Perceive đã dựng (cache theo ảnh gốc của lượt
    chạy). Ngược sáng ở chủ thể người được đo thêm trên vùng mặt (nếu có mask mặt): 'resolved'
    mà mặt không sáng lên tương ứng → 'unverified'.
    Không theo dõi được lỗi nào (thiếu số đo/mask) hoặc lỗi đo → [], decide_node dùng logic cũ.
    """
    diagnosis = state.get("diagnosis")
    original = state.get("original_image")
    defects = _targeted_defects(diagnosis, plan)
    if diagnosis is None or original is None or not defects:
        return []
    try:
        before, after = _measure(original), _measure(state["current_image"])
        if before is None or after is None:
            return []
        boxes = {canonical_region(name): box for name, box in diagnosis.subject_boxes.items()}
        people = {
            region
            for defect_type, region in defects
            if defect_type == "backlit_subject" and is_person_region(region)
        }
        names = {region for _, region in defects if region != "full"}
        masks = resolve_subject_masks(
            original,
            sorted(names | ({"face"} if people else set())),
            boxes,
            subject_cache_key(image_cache_key(original), original.shape),
        )
        long_side = max(original.shape[:2])

        def assess(
            defect_type: str, region: str, report: Any, stats: Dict[str, Any]
        ) -> Optional[Tuple[int, float]]:
            measured = measure_hypothesis(defect_type, region, report, stats, long_side)
            return None if measured is None else (measured.severity, measured.confidence)

        diff = compare(
            original,
            state["current_image"],
            {name: subject.mask for name, subject in masks.items()},
            defects=defects,
            original_measurement=before,
            current_measurement=after,
            assess=assess,
        )
        changes = [change for change in diff.defects if change.before >= 1]
        _verify_backlit(changes, masks, before, after, people)
    except Exception as exc:
        logger.warning("Defect tracking failed: %s", exc)
        return []
    return changes


def _decide_tracked(changes: List[DefectChange]) -> str:
    """
    Quyết định theo trạng thái lỗi mục tiêu (không kể trường hợp worse, đã xử lý như suy thoái):
    - có lỗi unverified (số đo vùng cha mâu thuẫn với vùng con) → STOP_BEST_EFFORT: không SHIP
      theo số đo, cũng không lặp vì Perceive vòng sau chỉ đo lại đúng số đo đó;
    - mọi lỗi resolved → SHIP;
    - có lỗi resolved/improved nhưng còn lỗi chưa hết → RE_PROCESS (decide_node áp giới hạn vòng);
    - mọi lỗi unchanged → STOP_BEST_EFFORT (xử lý không tác động được thêm).
    """
    statuses = {change.status for change in changes}
    if "unverified" in statuses:
        return "STOP_BEST_EFFORT"
    if statuses == {"resolved"}:
        return "SHIP"
    if statuses & {"resolved", "improved"}:
        return "RE_PROCESS"
    return "STOP_BEST_EFFORT"


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
    # Ảnh thực: thành công = lỗi đã chẩn đoán được sửa trên đúng vùng của nó (đo so với ảnh gốc),
    # không phải điểm toàn ảnh tăng. Có theo dõi được lỗi nào thì chỉ lỗi `worse` mới là suy thoái
    tracked = _track_defects(state, plan) if has_actions and not unchanged and not is_syn else []
    history_entry.defect_status = [change.model_dump() for change in tracked]
    if is_syn:
        regressed = _is_degraded_synthetic(eval_result, history)
    elif tracked:
        regressed = any(change.status == "worse" for change in tracked)
    else:
        regressed = _is_degraded_real(eval_result, history)
    degraded = has_actions and not unchanged and regressed
    # P7: tác dụng phụ do xử lý của vòng này (halo, nhiễu vùng tối tăng, banding…), ảnh thực.
    # Severity ≥ SIDE_EFFECT_ROLLBACK_SEVERITY → coi như suy thoái → rollback
    check = (
        _side_effects_check(state)
        if has_actions and not unchanged and not is_syn and previous_image is not None
        else SideEffectCheck()
    )
    history_entry.side_effect_check = check
    history_entry.executed = list(state.get("executed_actions") or [])
    history_entry.side_effects = list(check.previous)
    risky = [
        (effect["kind"], effect["severity"])
        for effect in check.previous
        if effect["severity"] >= SIDE_EFFECT_ROLLBACK_SEVERITY
    ]
    if risky:
        logger.warning(
            "Iteration %d introduced processing side effects %s; treating it as degraded.",
            iteration,
            risky,
        )
        degraded = True

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
        if is_syn:
            decision = _decide_synthetic(eval_result, history)
        elif tracked:
            decision = _decide_tracked(tracked)
        else:
            decision = _decide_real(eval_result, history)
            # 3. Ảnh thực không theo dõi được lỗi: SHIP theo điểm chỉ được chấp nhận khi chẩn
            #    đoán xác nhận lỗi rõ đã hết. Vòng cuối không còn cơ hội xác nhận → giữ SHIP
            if (
                decision == "SHIP"
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

    update: Dict[str, Any] = {
        "current_image": final_image,
        "rolled_back": rolled_back,
        "iteration": iteration + 1,
        "decision": decision,
        "history": new_history,
        "intermediate_images": new_intermediates,
    }
    if rolled_back:
        # evaluation_result đang mô tả ảnh bị loại: tính lại cho ảnh thực sự được trả
        update["evaluation_result"] = _evaluate_returned(state, final_image, iteration)
    return update


def _evaluate_returned(state: DoctorState, image: np.ndarray, iteration: int) -> Dict[str, Any]:
    """
    Đánh giá ảnh được trả sau rollback/từ chối, so với ảnh gốc (không còn "ảnh trước vòng").
    Lỗi đo → {} chứ không giữ số của ảnh khác.
    """
    try:
        return _evaluate_image(state, image, state["original_image"], iteration)
    except Exception as exc:
        logger.warning("Re-evaluating the returned image failed: %s", exc)
        return {}


def should_continue(state: DoctorState) -> str:
    """Điều hướng sau decide: lặp lại vòng xử lý hoặc sang bước review rồi kết thúc."""
    return "re_process" if state["decision"] == "RE_PROCESS" else "ship"


def review_node(state: DoctorState) -> Dict[str, Any]:
    """
    Node 7: review ảnh SẼ TRẢ (R-08). Chỉ chạy cho ảnh thật đã bị đổi so với ảnh gốc.
    REJECT → trả ảnh trước vòng cuối (hoặc ảnh gốc), tính lại evaluation cho ảnh đó.
    Các verdict khác không đổi ảnh; kết cục do finalize_node quyết định.
    """
    returned = state["current_image"]
    original = state["original_image"]
    history = list(state.get("history") or [])
    reference_run = bool(state.get("is_synthetic", False)) and (
        state.get("ground_truth_image") is not None
    )
    if reference_run or same_image(returned, original):
        return {"review": ReviewResult(status="not_run", reason="nothing to review")}

    # Sau rollback ảnh trả chính là ảnh trước vòng cuối: không còn ảnh "trước" để đối chiếu
    previous = None if state.get("rolled_back") else state.get("previous_image")
    review = review_candidate(
        original,
        previous,
        returned,
        review_context_from_state(history, state.get("intent")),
    )
    update: Dict[str, Any] = {"review": review}
    if history:
        history[-1] = history[-1].model_copy(update={"review": review})
        update["history"] = history
    if review.status == "completed" and review.verdict == "REJECT":
        fallback = previous if previous is not None else original
        logger.warning("Review rejected the result; returning the previous image.")
        update["current_image"] = fallback
        update["rolled_back"] = True
        update["evaluation_result"] = _evaluate_returned(
            state, fallback, int(state.get("iteration", 1))
        )
    return update


def finalize_node(state: DoctorState) -> Dict[str, Any]:
    """
    Node 8: chốt kết cục (R-01). `decision` cuối chỉ là SHIP khi kết cục là ACCEPTED; các kết
    cục khác là STOP_BEST_EFFORT (pipeline đã dừng nhưng chưa có ai xác nhận ảnh tốt).
    """
    outcome = derive_outcome(
        original=state["original_image"],
        returned=state["current_image"],
        is_synthetic=bool(state.get("is_synthetic", False))
        and state.get("ground_truth_image") is not None,
        reference_met=_reference_met(state.get("evaluation_result") or {}),
        history=list(state.get("history") or []),
        intent=state.get("intent"),
        review=state.get("review"),
    )
    return {
        "outcome": outcome,
        "decision": "SHIP" if outcome.status == "ACCEPTED" else "STOP_BEST_EFFORT",
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
    workflow.add_node("review", review_node)
    workflow.add_node("finalize", finalize_node)

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
        {"re_process": "analyze", "ship": "review"},
    )
    workflow.add_edge("review", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=checkpointer)


def run_pipeline(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    is_synthetic: bool = False,
    max_iterations: int = 3,
    intent: Optional[IntentProfile] = None,
    user_id: str = DEFAULT_USER,
    image_metadata: Optional[ImageMetadata] = None,
) -> DoctorState:
    """
    Hàm giao tiếp ngoài để chạy toàn bộ chu trình xử lý ảnh.
    user_id: chủ của các ca trong bộ nhớ (kinh nghiệm và sở thích được tách theo người dùng).
    intent: ý định người dùng có sẵn (không hỏi). Chạy tương tác: xem src/agent/session.py.
    image_metadata: metadata của file gốc (metadata.read_metadata); chỉ được lưu vào state.
    """
    app = build_doctor_graph()
    initial_state = initial_doctor_state(
        image,
        ground_truth,
        is_synthetic,
        max_iterations,
        intent,
        user_id=user_id,
        image_metadata=image_metadata,
    )
    return app.invoke(initial_state)


def initial_doctor_state(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    is_synthetic: bool = False,
    max_iterations: int = 3,
    intent: Optional[IntentProfile] = None,
    interactive: bool = False,
    user_id: str = DEFAULT_USER,
    image_metadata: Optional[ImageMetadata] = None,
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
        "interactive": interactive,
        "intent": intent,
        "user_id": user_id,
        "image_metadata": image_metadata,
        "executed_actions": [],
        "outcome": None,
        "review": None,
    }
    return initial_state
