"""
Chẩn đoán bệnh lý ảnh bằng mô hình đa phương thức Gemini (VLM Diagnostician).
Kết hợp chỉ số kỹ thuật từ Module 1 và thị giác máy tính để xây dựng kế hoạch điều trị.
"""

import json
import os
from typing import Any, Dict
import numpy as np
from PIL import Image
from .state import TreatmentPlan


SYSTEM_PROMPT = """
Bạn là "AI Image Doctor" - một chuyên gia chẩn đoán và xử lý ảnh theo phương pháp kinh điển (Classical Image Processing).
Nhiệm vụ của bạn:
1. Nhìn ảnh và đọc các chỉ số kỹ thuật đo lường (độ sáng, tương phản, nhiễu, độ mờ).
2. Chẩn đoán vấn đề ở từng vùng cụ thể (ví dụ: bầu trời bị chói, khuôn mặt bị tối, nền bị nhiễu).
3. Đề xuất kế hoạch điều trị chỉ sử dụng các công cụ trong Toolbox hợp lệ:
   - denoise (parameters: method in ['gaussian', 'median', 'bilateral', 'nlm'], strength in [0.1, 2.0])
   - gamma_correct (parameters: gamma in [0.5, 2.5])
   - clahe (parameters: clip_limit in [1.0, 4.0])
   - sharpen (parameters: method in ['unsharp_mask', 'laplacian'], amount in [0.2, 2.0])
   - color_correct (parameters: saturation_scale in [0.5, 1.5], temperature_shift in [-1.0, 1.0])
4. Tuân thủ thứ tự ưu tiên: Luôn khử nhiễu TRƯỚC KHI làm nét (denoise before sharpen).

Trả về kết quả dưới định dạng JSON thuần túy theo cấu trúc:
{
  "iteration": 1,
  "reasoning": "Giải thích lý do lựa chọn thuật toán và tham số...",
  "actions": [
    {
      "region_id": "sky",
      "target_prompt": "sky",
      "detected_issue": "overexposed",
      "operation": "gamma_correct",
      "parameters": {"gamma": 0.8},
      "order": 1
    }
  ]
}
"""


def diagnose_and_plan(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int = 1
) -> TreatmentPlan:
    """
    Gọi mô hình Gemini VLM để phân tích và tạo kế hoạch điều trị.
    Có fallback rule-based nếu chưa cấu hình GEMINI_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback kế hoạch dự phòng dựa trên phân tích luật tĩnh
        actions = []
        order = 1
        if metrics.get("noise_level") in ["medium", "severe"]:
            actions.append({
                "region_id": "full_image",
                "target_prompt": "full",
                "detected_issue": "high_noise",
                "operation": "denoise",
                "parameters": {"method": "bilateral", "strength": 1.0},
                "order": order
            })
            order += 1

        if metrics.get("brightness_level") == "underexposed":
            actions.append({
                "region_id": "full_image",
                "target_prompt": "full",
                "detected_issue": "underexposed",
                "operation": "gamma_correct",
                "parameters": {"gamma": 1.3},
                "order": order
            })
            order += 1
        elif metrics.get("contrast_level") == "low":
            actions.append({
                "region_id": "full_image",
                "target_prompt": "full",
                "detected_issue": "low_contrast",
                "operation": "clahe",
                "parameters": {"clip_limit": 2.0},
                "order": order
            })
            order += 1

        return TreatmentPlan(
            iteration=iteration,
            reasoning="Chế độ Fallback Rule-Based: Điều chỉnh dựa trên ngưỡng thống kê kỹ thuật.",
            actions=actions
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        pil_img = Image.fromarray(image)
        prompt = f"{SYSTEM_PROMPT}\n\nChỉ số kỹ thuật hiện tại:\n{json.dumps(metrics, indent=2)}\nVòng lặp: {iteration}"
        response = model.generate_content([prompt, pil_img])

        # Trích xuất và phân tích chuỗi JSON trả về
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        return TreatmentPlan(**data)

    except Exception:
        # Fallback an toàn nếu có lỗi gọi mạng
        return TreatmentPlan(
            iteration=iteration,
            reasoning="Gặp sự cố khi gọi Gemini API, chuyển sang chế độ tự phục hồi.",
            actions=[]
        )
