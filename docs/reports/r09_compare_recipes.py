"""R-09: so sánh cấu hình recipe backlit trên ảnh thật (offline, không gọi Gemini)."""

import os
import sys

import cv2
import numpy as np

os.environ.pop("GEMINI_API_KEY", None)
from src.agent import review  # noqa: E402
from src.agent.graph import run_pipeline  # noqa: E402
from src.agent.knowledge import engine  # noqa: E402

review.REVIEW_ENABLED = False
CONFIGS = [("off", False, 0.4), ("on_0.25", True, 0.25), ("on_0.4", True, 0.4)]


def luma(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].mean() * 100 / 255)


def main(path: str) -> None:
    raw = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    print(f"image {path} {image.shape} L*={luma(image):.1f}")
    for name, enabled, scale in CONFIGS:
        engine.BACKLIT_LIGHT_PERSON_ENABLED = enabled
        engine.BACKLIT_LIGHT_PERSON_SCALE = scale
        state = run_pipeline(image, max_iterations=2)
        out = state["current_image"]
        outcome = state.get("outcome")
        ops = [
            (a.region_id, a.operation, a.actual_mode, round(a.strength_scale, 2))
            for item in state["history"]
            for a in item.executed
        ]
        effects = [e["kind"] for item in state["history"] for e in item.side_effects]
        print(
            f"{name}: decision={state['decision']} outcome={outcome.status if outcome else None} "
            f"L*={luma(out):.1f} (+{luma(out) - luma(image):.1f}) effects={effects} ops={ops}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
