"""
Unit tests for Gradio frontend app (frontend/app.py).
"""

import json
from unittest.mock import Mock

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
    assert len(outputs) == 10
    out_img, _, _, _, _, variants_gallery, variants_md, choice, actions, case_id = outputs
    assert case_id is None  # bộ nhớ tắt trong test

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
    assert len(process_with_variants(None, None, 1, 3)) == 10


def test_chat_flow_asks_answers_and_refines():
    """Phase 4: hỏi ý → trả lời → góp ý; mỗi bước cập nhật hội thoại và ảnh."""
    from frontend.app import QUESTION_SLOTS, chat_answer, chat_feedback, chat_start

    img = _dark_image()
    history, session_id, questions, *radios = chat_start(img, [])
    assert session_id and questions[-1]["id"] == "mood"
    assert len(radios) == QUESTION_SLOTS
    assert radios[len(questions) - 1]["visible"] is True
    assert "Chẩn đoán" in history[-1]["content"]

    answers = [q["default"] for q in questions] + [None] * (QUESTION_SLOTS - len(questions))
    history, out, gallery, choice, variant_actions, current, case_id = chat_answer(
        session_id, questions, *answers, "", history
    )
    assert case_id is None
    assert out.shape == img.shape and "Xong" in history[-1]["content"]
    assert choice["value"] in variant_actions

    history, refined, new_actions, cleared = chat_feedback(img, current, "tối quá", history)
    assert "Đã chỉnh" in history[-1]["content"] and cleared == ""
    assert len(new_actions) == len(current) + 1
    assert refined.mean() > out.mean()


def test_chat_without_image_or_session():
    from frontend.app import chat_answer, chat_feedback, chat_start

    history, session_id, *_ = chat_start(None, [])
    assert session_id is None and "tải một ảnh" in history[-1]["content"]
    history, out, *_ = chat_answer(None, [], None, None, None, "", [])
    assert out is None
    history, *_ = chat_feedback(_dark_image(), [], "", [])
    assert history == []


def test_ui_records_the_chosen_variant(tmp_path):
    """Phase 5: xuất phiên bản trong UI ghi lựa chọn vào bộ nhớ ca bệnh."""
    from frontend.app import process_with_variants, render_chosen_variant
    from src.agent.memory import configure_case_memory

    memory = configure_case_memory(tmp_path / "cases.sqlite")
    img = _dark_image()
    *_, actions, case_id = process_with_variants(img, None, 1, 3)
    assert case_id is not None
    render_chosen_variant(img, "vivid", actions, case_id)
    assert memory.get(case_id).chosen_variant == "vivid"


def test_resets_clear_image_dependent_state():
    from frontend.app import chat_reset, processing_reset

    result = processing_reset()
    assert len(result) == 10
    assert result[0] is None and result[2] == [] and result[4] == "{}"
    assert result[7]["value"] is None and result[8:] == ({}, None)
    chat = chat_reset()
    assert len(chat) == 14
    assert chat[:3] == ([], None, [])
    assert all(radio["visible"] is False for radio in chat[3:6])
    assert chat[6] is None and chat[9:12] == ({}, [], None)


def test_ui_job_success_calls_once_and_preserves_choice_updates():
    from frontend.app import WAITING, ui_job

    with gr.Blocks():
        image, choice, button, status = gr.Image(), gr.Radio(), gr.Button(), gr.Markdown()
    callback = Mock(return_value=(None, gr.update(choices=["balanced"], value="balanced")))
    job = ui_job(
        callback,
        ("input",),
        [image, choice],
        [choice, button],
        (None, gr.update(choices=[], value=None)),
        status,
        {button: lambda values: values[choice]["value"] == "balanced"},
    )
    waiting = next(job)
    assert waiting[status] == WAITING and not waiting[button]["interactive"]
    callback.assert_not_called()
    result = next(job)
    callback.assert_called_once_with("input")
    assert result[choice]["choices"] == ["balanced"]
    assert result[choice]["value"] == "balanced" and result[button]["interactive"]


def test_ui_job_failure_clears_results_unlocks_and_hides_exception(caplog):
    from frontend.app import ERROR, ui_job

    with gr.Blocks():
        image, state, run, render, status = (
            gr.Image(),
            gr.State(),
            gr.Button(),
            gr.Button(),
            gr.Markdown(),
        )
    callback = Mock(side_effect=RuntimeError("private payload"))
    updates = list(
        ui_job(
            callback,
            (),
            [image, state],
            [run, render],
            (None, {}),
            status,
            {render: lambda values: bool(values[state])},
        )
    )
    final = updates[-1]
    assert final[image] is None and final[state] == {}
    assert final[run]["interactive"] and not final[render]["interactive"]
    assert final[status] == ERROR and "private payload" not in final[status]
    assert "UI operation failed" in caplog.text
    callback.assert_called_once()


def test_processing_wrapper_invokes_pipeline_once(monkeypatch):
    import frontend.app as app

    image = _dark_image()
    pipeline = Mock(return_value={"current_image": image, "decision": "SHIP", "iteration": 2})
    monkeypatch.setattr(app, "run_pipeline", pipeline)
    monkeypatch.setattr(app, "remember_run", lambda state: None)
    result = app.process_with_variants(image, None, 1, 1)
    assert len(result) == 10 and result[7]["value"] is None and result[8] == {}
    pipeline.assert_called_once()


def test_chat_ui_consumes_session_and_keeps_variant_choices_during_refinement(monkeypatch):
    import frontend.app as app

    demo = app.create_app()
    handlers = {fn.fn.__name__: fn for fn in demo.fns.values()}
    image = _dark_image()
    answer = Mock(
        return_value=(
            [],
            image,
            [],
            gr.update(choices=["balanced"], value="balanced"),
            {"balanced": []},
            [],
            None,
        )
    )
    monkeypatch.setattr(app, "chat_answer", answer)
    handler = handlers["answer_job"]
    result = list(handler.fn("session-a", [], None, None, None, "", []))[-1]
    answer.assert_called_once()
    assert result[handler.outputs[1]] is None
    assert result[handler.outputs[8]]["choices"] == ["balanced"]
    feedback = Mock(return_value=([], image, [], ""))
    monkeypatch.setattr(app, "chat_feedback", feedback)
    snapshot = [
        [],
        None,
        [],
        None,
        None,
        None,
        image,
        [],
        "balanced",
        {"balanced": []},
        [],
        None,
        "",
        "",
    ]
    job = handlers["feedback_job"].fn(image, [], "tối quá", [], None, *snapshot)
    waiting = next(job)
    choice = handlers["feedback_job"].outputs[8]
    assert "choices" not in waiting[choice]
    final = next(job)
    assert final[choice]["value"] == "balanced"
    feedback.assert_called_once()
