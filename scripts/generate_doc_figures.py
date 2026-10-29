"""Tạo hình gamma/mask tổng hợp, không dùng ảnh hoặc model bên ngoài."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing_engine.exposure_contrast import apply_gamma  # noqa: E402


def main() -> None:
    """Xuất ảnh so sánh xác định cho cả hai bản tài liệu."""
    width, height = 256, 192
    ramp = np.linspace(0, 120, width, dtype=np.uint8)
    original = np.repeat(np.tile(ramp, (height, 1))[..., None], 3, axis=2)
    yy, xx = np.mgrid[:height, :width]
    binary = (((xx - 128) ** 2 + (yy - 96) ** 2) < 65**2).astype(np.float32)
    mask = np.clip(cv2.GaussianBlur(binary, (0, 0), 10), 0, 1)
    processed = apply_gamma(original, mask=mask, gamma=2.0)
    canvas = Image.new("RGB", (width * 3, height + 32), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [original, np.repeat((mask * 255).astype(np.uint8)[..., None], 3, 2), processed]
    for index, (label, panel) in enumerate(zip(["Input / Goc", "Mask", "Gamma 2.0"], panels)):
        draw.text((index * width + 8, 8), label, fill="black")
        canvas.paste(Image.fromarray(panel), (index * width, 32))
    target = Path(__file__).resolve().parents[1] / "docs/assets/gamma-mask-example.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


if __name__ == "__main__":
    main()
