import os
import sys

import cv2

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv

load_dotenv()

# LƯU Ý: Phải cấu hình GEMINI_API_KEY trong file .env trước khi chạy
if "GEMINI_API_KEY" not in os.environ:
    print("CẢNH BÁO: Không tìm thấy GEMINI_API_KEY. Pipeline sẽ chạy ở chế độ Fallback Rule-Based.")

from src.agent.graph import run_pipeline  # noqa: E402


def test_image(img_path):
    print("\n==================================================")
    print(f"Đang kiểm tra ảnh: {img_path}")
    print("==================================================")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Không thể đọc ảnh: {img_path}")
        return

    # Chuyển đổi BGR (OpenCV) sang RGB (Pipeline)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Khởi chạy Pipeline (is_synthetic=False)
    state = run_pipeline(image=img_rgb, is_synthetic=False, max_iterations=2)

    history = state.get("history", [])
    if not history:
        print("Pipeline không tạo ra lịch sử nào.")
        return

    for h in history:
        print(f"\n--- [VÒNG LẶP {h.iteration}] ---")

        # 1. Kết quả đo lường (Module 1)
        mb = h.metrics_before
        print("Bắt mạch (Module 1):")
        print(
            f"  - Ánh sáng: {mb.get('brightness_level')} (mean={mb.get('brightness_mean', 0):.1f})"
        )
        print(f"  - Tương phản: {mb.get('contrast_level')} (std={mb.get('contrast_std', 0):.1f})")
        print(f"  - Nhiễu: {mb.get('noise_level')} (var={mb.get('noise_variance', 0):.1f})")
        print(f"  - Ám màu: {mb.get('color_cast')}")

        # 2. Phán đoán của Gemini (Module 4)
        print("\nBác sĩ VLM (Gemini 3.5 Flash-Lite) chẩn đoán:")
        print(f'  > "{h.plan.reasoning}"')

        # 3. Kế hoạch điều trị
        print("\nKê đơn thuốc (Actions):")
        if not h.plan.actions:
            print("  (Không có hành động nào - Đã đẹp)")
        for a in h.plan.actions:
            print(f"  - {a.operation} on '{a.region_id}' ({a.region_type}): {a.parameters}")

        # 4. Quyết định của LangGraph
        print(f"\nQuyết định sau xử lý: {h.decision}")


def main():
    images = [
        "data/real/coffee_underexposed.jpg",
        "data/real/astro_noisy.jpg",
        "data/real/cat_warm_cast.jpg",
    ]
    for img in images:
        test_image(img)


if __name__ == "__main__":
    main()
