"""
Xây dựng đồ thị trạng thái LangGraph (LangGraph Orchestration Graph).
Quản lý chu trình khép kín: Analyze -> Diagnose -> Plan -> Process -> Evaluate -> Decide.
"""

from typing import Any, Dict, Optional

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
from .state import DoctorState, HistoryItem
from .vlm_diagnostician import diagnose_and_plan


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

    # Chuẩn hóa về dict thuần trước khi đưa vào state: DoctorState khai báo
    # evaluation_result là Dict, HistoryItem.eval_score và ProcessResponse
    # cũng yêu cầu dict (model thô gây ValidationError và không JSON-serializable).
    return {"evaluation_result": eval_metrics.model_dump()}


def decide_node(state: DoctorState) -> Dict[str, Any]:
    """Node 6: Đưa ra quyết định dừng lại (SHIP) hay thử lại (RE_PROCESS)."""
    iteration = state["iteration"]
    max_iters = state.get("max_iterations", 3)

    # Cập nhật lịch sử
    history_entry = HistoryItem(
        iteration=iteration,
        plan=state["treatment_plan"],
        metrics_before=state["technical_metrics"],
        metrics_after=analyze_image(state["current_image"]).model_dump(),
        eval_score=state["evaluation_result"],
        decision="SHIP",
    )
    new_history = list(state.get("history", [])) + [history_entry]

    # Điều kiện dừng
    if iteration >= max_iters:
        decision = "STOP_BEST_EFFORT"
    else:
        # Nếu đã đạt chỉ số tốt hoặc không còn thao tác cần làm
        if not state.get("treatment_plan") or not state["treatment_plan"].actions:
            decision = "SHIP"
        else:
            # Tiếp tục vòng lặp nếu còn dư số lần
            decision = "SHIP" if iteration >= 2 else "RE_PROCESS"

    history_entry.decision = decision
    return {"iteration": iteration + 1, "decision": decision, "history": new_history}


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
        "decision": "INITIALIZING",
        "error_message": None,
    }
    return app.invoke(initial_state)
