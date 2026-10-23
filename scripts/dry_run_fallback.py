import json
import os
import sys

# Đảm bảo import đúng từ src
sys.path.insert(0, os.path.abspath("."))

from src.agent.vlm_diagnostician import diagnose_and_plan
from src.analyzer_evaluator.analyzer import analyze_image
from tests.fixtures.synthetic_images import (
    create_checkerboard_image,
    create_noisy_image,
    create_underexposed_image,
)


def main():
    print(">>> 1. Creating sample real image (Underexposed + Noisy) ...")
    base = create_checkerboard_image((256, 256, 3), block_size=32)
    under = create_underexposed_image(base, factor=0.4)
    img = create_noisy_image(under, noise_std=15.0)

    print(">>> 2. Running Module 1 (Analyzer) ...")
    metrics = analyze_image(img)
    print("Technical Metrics:")
    print(
        json.dumps(
            {
                "brightness_level": metrics.brightness_level,
                "contrast_level": metrics.contrast_level,
                "noise_level": metrics.noise_level,
            },
            indent=2,
        )
    )

    print("\n>>> 3. Forcing FALLBACK MODE (Removing GEMINI_API_KEY) ...")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        plan = diagnose_and_plan(image=img, metrics=metrics.model_dump(), iteration=1, history=[])
        print("\n=== FALLBACK TREATMENT PLAN RESPONSE ===")
        print(f"Iteration: {plan.iteration}")
        print(f"Reasoning: {plan.reasoning}")
        print("Actions:")
        for a in plan.actions:
            print(
                f"  - [{a.order}] {a.operation} on '{a.region_id}' ({a.region_type}): {a.parameters}"
            )
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()
