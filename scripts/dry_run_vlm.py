import json
import os
import sys

# Đảm bảo import đúng từ src
sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv

load_dotenv()

from src.agent.vlm_diagnostician import diagnose_and_plan  # noqa: E402
from src.analyzer_evaluator.analyzer import analyze_image  # noqa: E402
from tests.fixtures.synthetic_images import (  # noqa: E402
    create_checkerboard_image,
    create_noisy_image,
    create_underexposed_image,
)


def main():
    print(">>> 1. Creating sample real image (Underexposed + Noisy) ...")
    base = create_checkerboard_image((256, 256, 3), block_size=32)
    under = create_underexposed_image(base, factor=0.4)
    # Thêm chút nhiễu
    img = create_noisy_image(under, noise_std=15.0)

    print(">>> 2. Running Module 1 (Analyzer) ...")
    metrics = analyze_image(img)
    print("Technical Metrics:")
    print(json.dumps(metrics.model_dump(), indent=2))

    print("\n>>> 3. Sending to Gemini 3.5 Flash-Lite (Module 4 - VLM Diagnostician) ...")
    # Kiểm tra API KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("LỖI: Chưa có GEMINI_API_KEY trong biến môi trường.")
        return

    try:
        plan = diagnose_and_plan(image=img, metrics=metrics.model_dump(), iteration=1, history=[])
        print("\n=== VLM TREATMENT PLAN RESPONSE ===")
        print(f"Iteration: {plan.iteration}")
        print(f"Reasoning: {plan.reasoning}")
        print("Actions:")
        for a in plan.actions:
            print(
                f"  - [{a.order}] {a.operation} on '{a.region_id}' ({a.region_type}): {a.parameters}"
            )
    except Exception as e:
        print(f"Lỗi khi gọi VLM: {e}")


if __name__ == "__main__":
    main()
