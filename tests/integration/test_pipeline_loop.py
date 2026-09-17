"""
Integration test for the full LangGraph closed-loop pipeline.
"""

import numpy as np

from src.agent.graph import run_pipeline


def test_full_pipeline_synthetic():
    """Kiểm tra toàn bộ vòng lặp khép kín với dữ liệu synthetic mẫu."""
    clean_img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    # Tạo ảnh nhiễu nhân tạo
    noise = (np.random.randn(64, 64, 3) * 20).astype(np.int16)
    degraded_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    result_state = run_pipeline(
        image=degraded_img, ground_truth=clean_img, is_synthetic=True, max_iterations=2
    )

    assert result_state["current_image"].shape == degraded_img.shape
    assert result_state["iteration"] >= 2
    assert result_state["decision"] in ["SHIP", "STOP_BEST_EFFORT"]
    assert len(result_state["history"]) > 0
    # Khế ước serialization: evaluation_result trong state phải là dict thuần
    # (HistoryItem.eval_score, ProcessResponse.final_evaluation và json.dumps
    # ở frontend đều yêu cầu dict, không chấp nhận Pydantic model thô).
    assert isinstance(result_state["evaluation_result"], dict)
    assert result_state["evaluation_result"]["is_reference_eval"] is True
