"""
Module AI Agent & LangGraph Orchestration.
Điều phối toàn bộ quy trình chẩn đoán, lập kế hoạch, xử lý ảnh và đánh giá chất lượng.
"""

__all__ = ["build_doctor_graph", "run_pipeline", "DoctorState"]


def __getattr__(name: str):
    """Load graph dependencies only when the graph API is requested."""
    if name in {"build_doctor_graph", "run_pipeline"}:
        from .graph import build_doctor_graph, run_pipeline

        return {"build_doctor_graph": build_doctor_graph, "run_pipeline": run_pipeline}[name]
    if name == "DoctorState":
        from .state import DoctorState

        return DoctorState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
