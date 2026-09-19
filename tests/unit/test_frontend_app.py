"""
Unit tests for Gradio frontend app (frontend/app.py).
"""

import json

import gradio as gr
import numpy as np

from frontend.app import create_app, process_interface


def test_process_interface_none_input():
    """Kiểm tra khi chưa upload ảnh đầu vào."""
    out_img, status, gallery, reasoning, details = process_interface(None, None, 3)
    assert out_img is None
    assert "Vui lòng tải lên ảnh đầu vào" in status
    assert gallery == []
    assert reasoning == "*Chưa có dữ liệu.*"
    assert details == "{}"


def test_process_interface_valid_input_real():
    """Kiểm tra xử lý với ảnh đầu vào thực tế (không ground truth)."""
    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    out_img, status, gallery, reasoning, details = process_interface(img, None, 1)

    assert out_img is not None
    assert isinstance(out_img, np.ndarray)
    assert out_img.shape == (32, 32, 3)

    assert "Quyết định" in status
    assert "Số vòng lặp thực hiện" in status

    # Gallery phải có ít nhất ảnh gốc (Before) và kết quả
    assert isinstance(gallery, list)
    assert len(gallery) >= 1
    assert all(isinstance(item, tuple) and len(item) == 2 for item in gallery)

    # Reasoning và JSON details
    assert isinstance(reasoning, str)
    data = json.loads(details)
    assert "final_decision" in data
    assert "total_iterations" in data


def test_process_interface_synthetic_input():
    """Kiểm tra xử lý với ảnh nhân tạo (có ground truth)."""
    clean_img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    noisy_img = np.clip(clean_img.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    out_img, status, gallery, reasoning, details = process_interface(
        noisy_img, clean_img, 1
    )

    assert out_img is not None
    assert "Chỉ số tham chiếu" in status or "Quyết định" in status
    assert len(gallery) >= 1

    data = json.loads(details)
    assert "evaluation_result" in data


def test_create_app():
    """Kiểm tra hàm khởi tạo Gradio Blocks."""
    demo = create_app()
    assert isinstance(demo, gr.Blocks)
