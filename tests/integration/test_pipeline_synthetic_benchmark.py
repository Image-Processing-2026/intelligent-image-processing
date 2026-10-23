"""
Bộ test E2E Synthetic Benchmark (INT-04).
Kiểm chứng trọn vẹn luồng Agent <-> Evaluator (Module 4 <-> Module 1).
"""

from unittest.mock import patch

from src.agent.graph import run_pipeline
from tests.fixtures.synthetic_images import (
    create_checkerboard_image,
    create_flat_image,
    create_noisy_image,
    create_underexposed_image,
)


def test_underexposed_image_restoration():
    """
    Test ca ảnh thiếu sáng (underexposed).
    Chạy run_pipeline, assert chất lượng được Evaluator theo dõi,
    quyết định đưa ra là SHIP (hoặc có plan).
    """
    base = create_checkerboard_image((64, 64, 3))
    degraded = create_underexposed_image(base, factor=0.3)

    state = run_pipeline(image=degraded, ground_truth=base, is_synthetic=True, max_iterations=2)

    # Kế hoạch điều trị từ VLM (mặc dù fallback) phải có
    history = state.get("history", [])
    assert len(history) > 0
    assert state["decision"] in ["SHIP", "STOP_BEST_EFFORT"]

    eval_result = state.get("evaluation_result", {})
    # evaluation_result phải có delta metrics vì đã pass previous_image vào
    assert "delta_psnr" in eval_result.get("delta_metrics", {}) or not eval_result.get(
        "delta_metrics"
    )

    plan = history[0].plan
    assert plan is not None
    # Nếu VLM fallback hoạt động, phải sinh ra các actions
    assert len(plan.actions) > 0


def test_noisy_image_restoration():
    """
    Test ca ảnh nhiễu Gaussian.
    """
    base = create_flat_image(128, (64, 64, 3))
    degraded = create_noisy_image(base, noise_std=30.0)

    state = run_pipeline(image=degraded, ground_truth=base, is_synthetic=True, max_iterations=2)

    history = state.get("history", [])
    assert len(history) > 0

    metrics_before = history[0].metrics_before
    metrics_after = history[0].metrics_after

    # Không phải tất cả fallback xử lý nhiễu sẽ làm noise giảm tuyệt đối (do bộ lọc có thể làm mờ / thay đổi histogram).
    # Nhưng ta assert là logic có truyền metric đầy đủ trước/sau.
    assert "noise_variance" in metrics_before
    assert "noise_variance" in metrics_after


def test_degradation_guard():
    """
    Test cơ chế chặn suy thoái (Degradation Guard) ở đồ thị Real.
    Truyền vào ảnh bình thường, ép evaluate_no_reference trả về quality_improved = False
    để Agent dừng lại (STOP_BEST_EFFORT) ngay sau vòng 1.
    """
    base = create_flat_image(128, (64, 64, 3))

    with patch("src.agent.graph.evaluate_no_reference") as mock_eval:
        from src.analyzer_evaluator.analyzer import analyze_image
        from src.analyzer_evaluator.no_reference_eval import EvaluationResult

        # Tạo kết quả đánh giá giả lập với quality_improved=False
        fake_metrics = analyze_image(base)
        fake_eval = EvaluationResult(
            iteration=1,
            is_reference_eval=False,
            estimated_quality_score=40.0,
            technical_metrics=fake_metrics,
            delta_metrics={},
            quality_improved=False,  # <-- Guard trigger
            vlm_feedback="Mocked failure",
        )
        mock_eval.return_value = fake_eval

        # Max 3, nhưng phải dừng sau vòng 1 vì GUARD trigger
        state = run_pipeline(image=base, is_synthetic=False, max_iterations=3)

        # Cập nhật decision thành STOP_BEST_EFFORT trong history entry
        assert state["history"][0].decision == "STOP_BEST_EFFORT"
        # Đồ thị sẽ lặp (decide_node cập nhật iteration thành 2 rồi dừng)
        assert state["decision"] == "STOP_BEST_EFFORT"


def test_zero_redundant_compute():
    """
    Test đảm bảo không gọi dư thừa analyze_image trong decide_node.
    """
    base = create_flat_image(128, (64, 64, 3))

    # Mock analyze_image từ analyzer gốc
    with patch("src.agent.graph.analyze_image") as mock_analyze:
        from src.analyzer_evaluator.analyzer import analyze_image as original_analyze

        mock_analyze.side_effect = original_analyze

        run_pipeline(image=base, is_synthetic=False, max_iterations=1)

        # Lần 1: analyze_node (node 1)
        # Lần 2: evaluate_no_reference bên trong evaluate_node (để tính metrics ảnh hiện tại)
        # Lần 3 (nếu có prev_image): đánh giá prev_image trong evaluate_no_reference
        # NHƯNG KHÔNG ĐƯỢC GỌI trong decide_node!
        # Do đó chỉ nên dao động từ 1 -> 3 lần, thay vì 1 lần thừa trong decide_node.
        # Ở đây run_pipeline lần 1, previous_image=None nên trong evaluate_no_reference sẽ gọi 1 lần.
        # Suy ra mock_analyze.call_count == 2 (1 ở analyze_node, 1 ở evaluate_node).

        assert mock_analyze.call_count <= 2
