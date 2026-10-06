"""
Gradio UI Demo for Intelligent Image Processing.
Giao diện người dùng trực quan: so sánh Before/After, Timeline, lý do chẩn đoán VLM.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from dotenv import load_dotenv

from src.agent.executor import execute_plan
from src.agent.graph import run_pipeline
from src.agent.planner import sanitize_actions
from src.agent.state import DiagnosisReport, RegionOperation, TreatmentPlan
from src.analyzer_evaluator.analyzer import analyze_image

_SEVERITY_LABELS = {0: "không đáng kể", 1: "nhẹ", 2: "rõ", 3: "nặng"}


def format_diagnosis(diagnosis: Any) -> str:
    """Định dạng chẩn đoán giai đoạn Perceive thành Markdown cho UI."""
    if diagnosis is None:
        return ""
    if isinstance(diagnosis, dict):
        diagnosis = DiagnosisReport.model_validate(diagnosis)

    lighting = f" · ánh sáng: {diagnosis.lighting}" if diagnosis.lighting else ""
    lines = [f"- **Cảnh:** `{diagnosis.scene_type}`{lighting} · nguồn: `{diagnosis.source}`"]
    if diagnosis.summary:
        lines.append(f"- **Chẩn đoán:** {diagnosis.summary}")
    for defect in diagnosis.defects:
        severity = _SEVERITY_LABELS.get(defect.severity, str(defect.severity))
        evidence = f" — {defect.evidence}" if defect.evidence else ""
        lines.append(
            f"  - Lỗi **`{defect.type}`** ở `{defect.region}` "
            f"({severity}, {defect.origin}){evidence}"
        )
    for item in diagnosis.preserve:
        reason = f" — {item.reason}" if item.reason else ""
        lines.append(f"  - Giữ **`{item.aspect}`** ở `{item.region}`{reason}")
    for region, measured in diagnosis.region_metrics.items():
        gap = (
            f", lệch {measured.brightness_vs_rest:+.0f} so với phần còn lại"
            if measured.brightness_vs_rest is not None
            else ""
        )
        lines.append(
            f"  - Đo vùng `{region}` ({measured.backend}): "
            f"sáng {measured.brightness_mean:.0f}/255{gap}"
        )
    return "\n".join(lines)


def process_interface(
    input_image: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray],
    max_iters: int,
) -> Tuple[Optional[np.ndarray], str, List[Tuple[np.ndarray, str]], str, str]:
    """
    Xử lý ảnh qua chu trình LangGraph khép kín và định dạng kết quả cho UI.

    Returns:
        output_img: Ảnh kết quả sau điều trị (After).
        status_md: Markdown trạng thái, quyết định, số vòng lặp.
        gallery_images: Danh sách (ảnh, chú thích) cho Timeline.
        reasoning_md: Diễn giải chẩn đoán lâm sàng của VLM qua các vòng.
        details_json: Chuỗi JSON chi tiết kỹ thuật.
    """
    if input_image is None:
        return (
            None,
            "⚠️ **Vui lòng tải lên ảnh đầu vào để bắt đầu quá trình chẩn đoán & xử lý.**",
            [],
            "*Chưa có dữ liệu.*",
            "{}",
        )

    is_synthetic = ground_truth is not None
    result_state = run_pipeline(
        image=input_image,
        ground_truth=ground_truth,
        is_synthetic=is_synthetic,
        max_iterations=int(max_iters),
    )
    return _format_result(input_image, is_synthetic, max_iters, result_state)


def _format_result(
    input_image: np.ndarray,
    is_synthetic: bool,
    max_iters: int,
    result_state: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str, List[Tuple[np.ndarray, str]], str, str]:
    """Định dạng trạng thái cuối của pipeline thành 5 đầu ra của tab Chẩn đoán & Lịch sử."""
    output_img = result_state.get("current_image")
    decision = result_state.get("decision", "UNKNOWN")
    iters = max(0, result_state.get("iteration", 1) - 1)
    history = result_state.get("history", [])
    intermediates = result_state.get("intermediate_images", [])
    eval_result = result_state.get("evaluation_result", {})

    # 1. Trạng thái & Quyết định cuối
    decision_map = {
        "SHIP": ("✅", "SHIP — Đạt chất lượng mục tiêu"),
        "STOP_BEST_EFFORT": ("⚠️", "STOP_BEST_EFFORT — Dừng ở mức tối ưu tốt nhất"),
        "RE_PROCESS": ("🔄", "RE_PROCESS — Tiếp tục lặp"),
    }
    emoji, label = decision_map.get(decision, ("❓", decision))

    status_lines = [
        f"### {emoji} Quyết định: **{label}**",
        f"- **Số vòng lặp thực hiện:** {iters} / {int(max_iters)}",
    ]

    if is_synthetic and eval_result:
        psnr = eval_result.get("psnr", "N/A")
        ssim = eval_result.get("ssim", "N/A")
        status_lines.append(f"- **Chỉ số tham chiếu:** PSNR = `{psnr} dB` | SSIM = `{ssim}`")
    elif eval_result:
        score = eval_result.get("estimated_quality_score", "N/A")
        status_lines.append(f"- **Điểm chất lượng ước lượng (No-Reference):** `{score} / 100`")

    status_md = "\n".join(status_lines)

    # 2. Timeline Gallery (Before → Iteration 1 → ... → Final)
    gallery_images: List[Tuple[np.ndarray, str]] = []
    gallery_images.append((input_image, "1. Ảnh gốc (Before)"))
    for i, thumb in enumerate(intermediates):
        gallery_images.append((thumb, f"Vòng {i + 1}"))
    if output_img is not None:
        gallery_images.append((output_img, f"Kết quả cuối ({decision})"))

    # 3. Diễn giải chẩn đoán (Clinical Reasoning)
    reasoning_sections: List[str] = []
    for item in history:
        it = getattr(item, "iteration", None) or (
            item.get("iteration") if isinstance(item, dict) else "?"
        )
        plan = getattr(item, "plan", None) or (item.get("plan") if isinstance(item, dict) else None)
        item_decision = getattr(item, "decision", "") or (
            item.get("decision", "") if isinstance(item, dict) else ""
        )

        reasoning = ""
        actions: List[Any] = []
        knowledge: List[str] = []
        if plan:
            reasoning = getattr(plan, "reasoning", "") or (
                plan.get("reasoning", "") if isinstance(plan, dict) else ""
            )
            actions = getattr(plan, "actions", []) or (
                plan.get("actions", []) if isinstance(plan, dict) else []
            )
            knowledge = getattr(plan, "knowledge", []) or (
                plan.get("knowledge", []) if isinstance(plan, dict) else []
            )

        actions_text = []
        for a in actions:
            op = getattr(a, "operation", "") or (
                a.get("operation", "") if isinstance(a, dict) else ""
            )
            reg = getattr(a, "region_id", "") or (
                a.get("region_id", "") if isinstance(a, dict) else ""
            )
            params = getattr(a, "parameters", {}) or (
                a.get("parameters", {}) if isinstance(a, dict) else {}
            )
            params_str = ", ".join(f"`{k}={v}`" for k, v in params.items())
            actions_text.append(f"  - Thao tác **`{op}`** trên vùng `{reg}` ({params_str})")

        actions_block = "\n".join(actions_text) if actions_text else "  *(Không có thao tác)*"

        diagnosis = getattr(item, "diagnosis", None) or (
            item.get("diagnosis") if isinstance(item, dict) else None
        )
        diagnosis_md = format_diagnosis(diagnosis)

        reasoning_sections.append(
            f"#### 🔄 Vòng lặp {it} (Quyết định: `{item_decision}`)\n"
            + (f"{diagnosis_md}\n" if diagnosis_md else "")
            + f"- **Lập kế hoạch:** {reasoning}\n"
            + (
                f"- **Tri thức tham khảo:** {', '.join(f'`{k}`' for k in knowledge)}\n"
                if knowledge
                else ""
            )
            + f"- **Phác đồ thực thi:**\n{actions_block}"
        )

    reasoning_md = (
        "\n\n---\n\n".join(reasoning_sections)
        if reasoning_sections
        else "*Chưa có dữ liệu lịch sử.*"
    )

    # 4. JSON chi tiết kỹ thuật
    history_serialized = [
        item.model_dump()
        if hasattr(item, "model_dump")
        else (item if isinstance(item, dict) else str(item))
        for item in history
    ]
    details: Dict[str, Any] = {
        "final_decision": decision,
        "total_iterations": iters,
        "evaluation_result": eval_result,
        "final_technical_metrics": (
            analyze_image(output_img).model_dump() if output_img is not None else {}
        ),
        "history": history_serialized,
    }
    details_json = json.dumps(details, indent=2, ensure_ascii=False)

    return output_img, status_md, gallery_images, reasoning_md, details_json


VariantActions = Dict[str, List[Dict[str, Any]]]


def _variant_outputs(
    result_state: Dict[str, Any],
) -> Tuple[List[Tuple[np.ndarray, str]], str, Any, VariantActions]:
    """
    Đầu ra phần "Chọn phiên bản": gallery preview, mô tả xếp hạng, radio chọn (mặc định là
    phiên bản đề xuất) và phác đồ của từng phiên bản (giữ trong gr.State để render full-res).
    """
    variants = result_state.get("variants") or []
    if not variants:
        return [], "*Chỉ có một kết quả (số phiên bản = 1).*", gr.update(choices=[], value=None), {}

    recommended = result_state.get("recommended_variant")
    source = result_state.get("variant_ranking_source")
    source_text = "VLM critic so sánh trực quan" if source == "critic" else "điểm Module 1"
    gallery: List[Tuple[np.ndarray, str]] = []
    lines = [f"**Xếp hạng theo:** {source_text}"]
    for variant in variants:
        star = " ⭐ đề xuất" if variant.id == recommended else ""
        gallery.append((variant.image, f"#{variant.rank} {variant.label}{star}"))
        note = f" — _{variant.critic_note}_" if variant.critic_note else ""
        lines.append(
            f"{variant.rank}. **{variant.label}**{star}: {variant.description} "
            f"(điểm `{variant.quality_score}`){note}"
        )
    choices = [(variant.label, variant.id) for variant in variants]
    actions = {
        variant.id: [action.model_dump(mode="json") for action in variant.actions]
        for variant in variants
    }
    return gallery, "\n".join(lines), gr.update(choices=choices, value=recommended), actions


def process_with_variants(
    input_image: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray],
    max_iters: int,
    num_variants: int,
) -> Tuple[Any, ...]:
    """
    Chạy pipeline một lần, trả về 5 đầu ra của process_interface cộng 4 đầu ra của phần
    chọn phiên bản (gallery, mô tả, radio, phác đồ).
    """
    if input_image is None:
        empty = process_interface(None, None, max_iters)
        return (*empty, [], "*Chưa có phiên bản.*", gr.update(choices=[], value=None), {})
    is_synthetic = ground_truth is not None
    result_state = run_pipeline(
        image=input_image,
        ground_truth=ground_truth,
        is_synthetic=is_synthetic,
        max_iterations=int(max_iters),
        num_variants=int(num_variants),
    )
    return (
        *_format_result(input_image, is_synthetic, max_iters, result_state),
        *_variant_outputs(result_state),
    )


def render_chosen_variant(
    input_image: Optional[np.ndarray],
    choice: Optional[str],
    variant_actions: Optional[VariantActions],
) -> Tuple[Optional[np.ndarray], str]:
    """
    Render phiên bản đã chọn ở độ phân giải gốc: áp lại đúng phác đồ của phiên bản (giữ
    nguyên thứ tự, đã kẹp tham số) lên ảnh đầu vào.
    """
    if input_image is None or not choice or not variant_actions or choice not in variant_actions:
        return None, "⚠️ **Hãy chạy xử lý với số phiên bản > 1 rồi chọn một phiên bản.**"
    actions = sanitize_actions(
        [RegionOperation.model_validate(raw) for raw in variant_actions[choice]]
    )
    rendered = execute_plan(input_image, TreatmentPlan(reasoning=choice, actions=actions))
    height, width = rendered.shape[:2]
    return rendered, f"✅ Đã xuất phiên bản **{choice}** ở độ phân giải gốc ({width}×{height})."


def create_app() -> gr.Blocks:
    """Tạo giao diện Gradio Blocks hoàn chỉnh."""
    with gr.Blocks(title="AI Image Doctor — Intelligent Image Processing") as demo:
        gr.Markdown(
            "# 🩺 AI Image Doctor\n"
            "### Hệ thống Chẩn đoán & Xử lý Ảnh Khép kín (Closed-Loop Agentic Processing)\n"
            "*Kết hợp Thị giác Máy tính Kinh điển và Mô hình Đa phương thức Gemini VLM.*"
        )

        with gr.Tabs():
            # ==================== TAB 1: Chẩn đoán & Xử lý ====================
            with gr.TabItem("🩺 Chẩn đoán & Xử lý"):
                with gr.Row():
                    # Cột trái: Đầu vào và Điều khiển
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Ảnh Đầu vào (Before)")
                        in_img = gr.Image(label="Ảnh đầu vào (Input Image)", type="numpy")
                        gt_img = gr.Image(
                            label="📎 Ảnh Ground-Truth (Tùy chọn — Dành cho Synthetic Test)",
                            type="numpy",
                        )
                        max_iter_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="⚙️ Số vòng lặp tối đa (Max Iterations)",
                        )
                        variants_slider = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=3,
                            step=1,
                            label="🎨 Số phiên bản để chọn (1 = chỉ một kết quả)",
                        )
                        btn_run = gr.Button(
                            "🚀 Khởi chạy Chẩn đoán & Xử lý",
                            variant="primary",
                            size="lg",
                        )

                    # Cột phải: Kết quả và So sánh After
                    with gr.Column(scale=1):
                        gr.Markdown("### ✅ Ảnh Kết quả sau Điều trị (After)")
                        out_img = gr.Image(
                            label="Ảnh kết quả sau điều trị (Treated Image)",
                            type="numpy",
                        )
                        status_box = gr.Markdown(
                            value="*Tải ảnh lên và nhấn 'Khởi chạy' để bắt đầu quy trình AI Image Doctor.*",
                            label="Trạng thái",
                        )

                gr.Markdown("### 🎨 Chọn phiên bản (preview)")
                variants_gallery = gr.Gallery(
                    label="Các phiên bản", columns=3, height="auto", object_fit="contain"
                )
                variants_box = gr.Markdown(value="*Chưa có phiên bản.*")
                with gr.Row():
                    variant_choice = gr.Radio(label="Phiên bản muốn xuất", choices=[])
                    btn_render = gr.Button("💾 Xuất phiên bản đã chọn (full-res)")
                variant_actions = gr.State({})

            # ==================== TAB 2: Lịch sử Chi tiết ====================
            with gr.TabItem("📊 Lịch sử Chi tiết"):
                gr.Markdown("### 🖼️ Timeline Ảnh qua Từng Vòng Lặp")
                gallery = gr.Gallery(
                    label="Timeline quá trình xử lý",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )
                gr.Markdown("### 🧠 Lý do Chẩn đoán & Phác đồ Điều trị qua các Vòng")
                reasoning_box = gr.Markdown(value="*Chưa có dữ liệu lịch sử.*")
                with gr.Accordion("📋 JSON Chi tiết Kỹ thuật (Raw Data)", open=False):
                    details_box = gr.Code(label="Chi tiết JSON", language="json")

        btn_run.click(
            fn=process_with_variants,
            inputs=[in_img, gt_img, max_iter_slider, variants_slider],
            outputs=[
                out_img,
                status_box,
                gallery,
                reasoning_box,
                details_box,
                variants_gallery,
                variants_box,
                variant_choice,
                variant_actions,
            ],
        )
        btn_render.click(
            fn=render_chosen_variant,
            inputs=[in_img, variant_choice, variant_actions],
            outputs=[out_img, status_box],
        )

    return demo


if __name__ == "__main__":
    # Chỉ nạp .env khi chạy UI, không nạp khi test import module này
    load_dotenv(encoding="utf-8-sig")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
