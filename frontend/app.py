"""
Gradio UI Demo for Intelligent Image Processing.
Giao diện người dùng trực quan để tải ảnh, chạy chẩn đoán, xử lý lặp và so sánh kết quả.
"""

import json
import gradio as gr
import numpy as np
from src.agent.graph import run_pipeline
from src.analyzer_evaluator.analyzer import analyze_image


def process_interface(input_image, ground_truth, max_iters):
    """Xử lý ảnh qua quy trình LangGraph khép kín."""
    if input_image is None:
        return None, "Vui lòng tải lên ảnh đầu vào.", "{}"

    is_synthetic = ground_truth is not None
    result_state = run_pipeline(
        image=input_image,
        ground_truth=ground_truth,
        is_synthetic=is_synthetic,
        max_iterations=int(max_iters)
    )

    output_img = result_state["current_image"]
    decision = result_state["decision"]
    iters = result_state["iteration"] - 1

    summary_text = f"### Trạng thái hoàn thành: {decision}\n- Số vòng lặp đã thực hiện: {iters}"
    details = {
        "technical_metrics_final": analyze_image(output_img).model_dump(),
        "evaluation_result": result_state.get("evaluation_result", {}),
        "history_count": len(result_state.get("history", []))
    }

    return output_img, summary_text, json.dumps(details, indent=2, ensure_ascii=False)


def create_app():
    with gr.Blocks(title="AI Image Doctor — Intelligent Image Processing") as demo:
        gr.Markdown(
            "# 🩺 AI Image Doctor (Intelligent Image Processing)\n"
            "*Hệ thống chẩn đoán và xử lý ảnh khép kín kết hợp thị giác máy tính và thuật toán kinh điển.*"
        )

        with gr.Row():
            with gr.Column():
                in_img = gr.Image(label="Ảnh đầu vào (Input Image)", type="numpy")
                gt_img = gr.Image(label="Ảnh gốc mẫu Ground-Truth (Tùy chọn - Dành cho Synthetic Test)", type="numpy")
                max_iter_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Số vòng lặp tối đa (Max Iterations)")
                btn_run = gr.Button("🚀 Khởi chạy chẩn đoán & xử lý", variant="primary")

            with gr.Column():
                out_img = gr.Image(label="Ảnh kết quả sau điều trị (Treated Image)", type="numpy")
                status_box = gr.Markdown(label="Kết quả chẩn đoán")
                details_box = gr.Code(label="Chi tiết kỹ thuật & Đánh giá (JSON)", language="json")

        btn_run.click(
            fn=process_interface,
            inputs=[in_img, gt_img, max_iter_slider],
            outputs=[out_img, status_box, details_box]
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
