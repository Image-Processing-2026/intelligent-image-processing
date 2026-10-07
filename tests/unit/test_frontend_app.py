"""
Unit tests for Gradio frontend app (frontend/app.py).
"""

import json

import gradio as gr
import numpy as np

from frontend.app import create_app, format_diagnosis, process_interface


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

    out_img, status, gallery, reasoning, details = process_interface(noisy_img, clean_img, 1)

    assert out_img is not None
    assert "Chỉ số tham chiếu" in status or "Quyết định" in status
    assert len(gallery) >= 1

    data = json.loads(details)
    assert "evaluation_result" in data


def test_create_app():
    """Kiểm tra hàm khởi tạo Gradio Blocks."""
    demo = create_app()
    assert isinstance(demo, gr.Blocks)


def test_format_diagnosis_lists_defects_preserve_and_regions():
    """Chẩn đoán Phase 1 hiển thị lỗi, điều cần giữ và số đo vùng; nhận cả dict (JSON)."""
    from src.agent.state import Defect, DiagnosisReport, PreserveItem, RegionMetrics

    diagnosis = DiagnosisReport(
        scene_type="portrait",
        lighting="ngược sáng",
        summary="Mặt tối.",
        defects=[Defect(type="backlit_subject", region="face", severity=2, origin="measured")],
        preserve=[PreserveItem(aspect="warm_tone", reason="nắng chiều")],
        region_metrics={
            "face": RegionMetrics(
                region="face",
                backend="mediapipe",
                area_ratio=0.1,
                brightness_mean=55.0,
                brightness_std=10.0,
                brightness_level="underexposed",
                highlight_clip_ratio=0.0,
                shadow_clip_ratio=0.0,
                brightness_vs_rest=-120.0,
            )
        },
    )
    for value in (diagnosis, diagnosis.model_dump()):
        text = format_diagnosis(value)
        assert "portrait" in text
        assert "backlit_subject" in text
        assert "warm_tone" in text
        assert "-120" in text

    assert format_diagnosis(None) == ""


def test_reasoning_lists_playbook_knowledge():
    """Phase 2: timeline hiển thị id tri thức (playbook) mà kế hoạch đã dựa vào."""
    img = np.ones((32, 32, 3), dtype=np.uint8) * 30  # ảnh tối → card global-underexposed
    _, _, _, reasoning, _ = process_interface(img, None, 1)
    assert "Tri thức tham khảo" in reasoning
    assert "global-underexposed" in reasoning


def _dark_image() -> np.ndarray:
    gradient = np.tile(np.linspace(10, 60, 48), (48, 1)).astype(np.uint8)
    return np.repeat(gradient[:, :, None], 3, axis=2)


def test_process_with_variants_and_render_choice():
    """Phase 3: một lần chạy cho 3 phiên bản; xuất bản 'balanced' khớp ảnh kết quả."""
    from frontend.app import process_with_variants, render_chosen_variant

    img = _dark_image()
    outputs = process_with_variants(img, None, 2, 3)
    assert len(outputs) == 9
    out_img, _, _, _, _, variants_gallery, variants_md, choice, actions = outputs

    assert len(variants_gallery) == 3
    assert "Xếp hạng theo" in variants_md
    assert choice["value"] in actions
    assert sorted(actions) == ["balanced", "natural", "vivid"]

    rendered, status = render_chosen_variant(img, "balanced", actions)
    assert np.array_equal(rendered, out_img)
    assert "48×48" in status


def test_process_with_single_variant_and_missing_choice():
    from frontend.app import process_with_variants, render_chosen_variant

    outputs = process_with_variants(_dark_image(), None, 1, 1)
    assert outputs[5] == [] and outputs[8] == {}
    rendered, status = render_chosen_variant(_dark_image(), None, {})
    assert rendered is None and "chọn một phiên bản" in status
    assert len(process_with_variants(None, None, 1, 3)) == 9
