"""
Xây dựng đồ thị trạng thái LangGraph (LangGraph Orchestration Graph).
Quản lý chu trình khép kín: Analyze -> Diagnose -> Plan -> Process -> Evaluate -> Decide.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Đảm bảo tương thích ngược nếu langchain phiên bản cũ/mới bị xung đột module attribute
try:
    import langchain

    if not hasattr(langchain, "debug"):
        langchain.debug = False
except ImportError:
    pass

from langgraph.graph import END, StateGraph

from src.analyzer_evaluator.analyzer import analyze_image
from src.analyzer_evaluator.no_reference_eval import evaluate_no_reference
from src.analyzer_evaluator.reference_eval import evaluate_reference

from .executor import execute_plan
from .planner import validate_and_sort_plan
from .state import DoctorState, HistoryItem, TreatmentPlan
from .vlm_diagnostician import diagnose_and_plan

# Ngưỡng chất lượng cho quyết định dừng (Synthetic Benchmark)
PSNR_THRESHOLD = 28.0  # dB — Chất lượng phục hồi chấp nhận được
SSIM_THRESHOLD = 0.88  # Tương đồng cấu trúc chấp nhận được
PSNR_DEGRADATION = 1.5  # dB — Ngưỡng suy thoái cho phép giữa 2 vòng


# ---------------------------------------------------------
# Các Node thực thi trong đồ thị LangGraph
# ---------------------------------------------------------
def analyze_node(state: DoctorState) -> Dict[str, Any]:
    """Node 1: Phân tích các chỉ số kỹ thuật của bức ảnh hiện tại."""
    metrics = analyze_image(state["current_image"])
    return {"technical_metrics": metrics.model_dump()}


def diagnose_and_plan_node(state: DoctorState) -> Dict[str, Any]:
    """Node 2 & 3: Gọi VLM chẩn đoán bệnh và tạo kế hoạch điều trị."""
    plan = diagnose_and_plan(
        image=state["current_image"],
        metrics=state["technical_metrics"],
        iteration=state["iteration"],
        history=state.get("history", []),
    )
    validated_plan = validate_and_sort_plan(plan)
    return {"treatment_plan": validated_plan}


def process_node(state: DoctorState) -> Dict[str, Any]:
    """Node 4: Thực thi kế hoạch điều trị (Module 2 & Module 3)."""
    if not state.get("treatment_plan") or not state["treatment_plan"].actions:
        return {"current_image": state["current_image"]}

    processed_img = execute_plan(state["current_image"], state["treatment_plan"])
    return {"current_image": processed_img}


def evaluate_node(state: DoctorState) -> Dict[str, Any]:
    """Node 5: Đánh giá chất lượng sau khi xử lý (Module 1)."""
    is_syn = state.get("is_synthetic", False)
    gt = state.get("ground_truth_image")

    if is_syn and gt is not None:
        eval_metrics = evaluate_reference(state["current_image"], gt)
    else:
        eval_metrics = evaluate_no_reference(
            current_image=state["current_image"], previous_image=state["original_image"]
        )

    return {"evaluation_result": eval_metrics}


def _decide_synthetic(eval_result: Dict[str, Any], history: List[HistoryItem]) -> str:
    """Logic quyết định dành cho ảnh nhân tạo có Ground-Truth."""
    psnr = float(eval_result.get("psnr", 0.0))
    ssim = float(eval_result.get("ssim", 0.0))

    # Đã đạt chất lượng mục tiêu
    if psnr >= PSNR_THRESHOLD and ssim >= SSIM_THRESHOLD:
        return "SHIP"

    # Kiểm tra suy thoái: PSNR vòng này thấp hơn vòng trước quá ngưỡng
    if history:
        prev_eval = history[-1].eval_score or {}
        prev_psnr = float(prev_eval.get("psnr", 0.0))
        if prev_psnr - psnr > PSNR_DEGRADATION:
            return "STOP_BEST_EFFORT"

    # Chưa đạt, còn dư vòng → tiếp tục
    return "RE_PROCESS"


def _decide_real(eval_result: Dict[str, Any], history: List[HistoryItem]) -> str:
    """Logic quyết định dành cho ảnh thực không có Ground-Truth."""
    current_quality = float(eval_result.get("estimated_quality_score", 50.0))

    # Kiểm tra suy thoái: quality score vòng này tệ hơn vòng trước
    if history:
        prev_eval = history[-1].eval_score or {}
        prev_quality = float(prev_eval.get("estimated_quality_score", 50.0))
        if current_quality < prev_quality:
            return "STOP_BEST_EFFORT"

    # Chưa đạt, còn dư vòng → tiếp tục
    return "RE_PROCESS"


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

    # Cập nhật lịch sử
    history_entry = HistoryItem(
        iteration=iteration,
        plan=plan,
        metrics_before=state.get("technical_metrics", {}),
        metrics_after=analyze_image(state["current_image"]).model_dump(),
        eval_score=eval_result,
        decision="PENDING",  # sẽ cập nhật bên dưới
    )

    # ====== LOGIC RA QUYẾT ĐỊNH ======

    # 1. Giới hạn cứng: Hết vòng lặp → dừng nỗ lực tốt nhất
    if iteration >= max_iters:
        decision = "STOP_BEST_EFFORT"

    # 2. Kế hoạch rỗng → VLM cho rằng ảnh đã tốt
    elif not state.get("treatment_plan") or not state["treatment_plan"].actions:
        decision = "SHIP"

    # 3. Phân nhánh Synthetic vs Real
    elif is_syn:
        decision = _decide_synthetic(eval_result, history)
    else:
        decision = _decide_real(eval_result, history)

    history_entry.decision = decision
    new_history = history + [history_entry]

    # Lưu ảnh trung gian dạng thumbnail sau mỗi vòng
    thumbnail = _create_thumbnail(state["current_image"])
    new_intermediates = list(state.get("intermediate_images") or []) + [thumbnail]

    return {
        "iteration": iteration + 1,
        "decision": decision,
        "history": new_history,
        "intermediate_images": new_intermediates,
    }


def should_continue(state: DoctorState) -> str:
    """Hàm điều hướng có rẽ nhánh tiếp tục lặp hay kết thúc."""
    if state["decision"] == "RE_PROCESS":
        return "re_process"
    return "ship"


# ---------------------------------------------------------
# Xây dựng và biên dịch đồ thị
# ---------------------------------------------------------
def build_doctor_graph():
    """Khởi tạo StateGraph của LangGraph."""
    workflow = StateGraph(DoctorState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("diagnose_and_plan", diagnose_and_plan_node)
    workflow.add_node("process", process_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("decide", decide_node)

    workflow.set_entry_point("analyze")

    workflow.add_edge("analyze", "diagnose_and_plan")
    workflow.add_edge("diagnose_and_plan", "process")
    workflow.add_edge("process", "evaluate")
    workflow.add_edge("evaluate", "decide")

    workflow.add_conditional_edges(
        "decide", should_continue, {"re_process": "analyze", "ship": END}
    )

    return workflow.compile()


def run_pipeline(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    is_synthetic: bool = False,
    max_iterations: int = 3,
) -> DoctorState:
    """Hàm giao tiếp ngoài để chạy toàn bộ chu trình xử lý ảnh."""
    app = build_doctor_graph()
    initial_state: DoctorState = {
        "original_image": image,
        "current_image": image.copy(),
        "ground_truth_image": ground_truth,
        "is_synthetic": is_synthetic,
        "iteration": 1,
        "max_iterations": max_iterations,
        "technical_metrics": {},
        "treatment_plan": None,
        "evaluation_result": {},
        "history": [],
        "intermediate_images": [],
        "decision": "INITIALIZING",
        "error_message": None,
    }
    return app.invoke(initial_state)
