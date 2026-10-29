"""
Kiểm thử overlay vùng: mask khớp hình chữ nhật chủ thể (IoU) và bản vẽ đánh dấu đúng vùng.
Chạy offline, không cần API key.
"""

import cv2
import numpy as np

from src.agent.region_overlay import diagnosis_region_overlay, render_region_overlay
from src.agent.state import Defect, DiagnosisReport
from src.region_engine.subject import SubjectMask

HEIGHT, WIDTH = 256, 384
SUBJECT = (70, 200, 130, 260)  # y0, y1, x0, x1
BOX_2D = [
    round(SUBJECT[0] / HEIGHT * 1000),
    round(SUBJECT[2] / WIDTH * 1000),
    round(SUBJECT[1] / HEIGHT * 1000),
    round(SUBJECT[3] / WIDTH * 1000),
]


def _image() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), (150, 152, 158), dtype=np.uint8)
    y0, y1, x0, x1 = SUBJECT
    image[y0:y1, x0:x1] = (45, 43, 40)
    return image


def _truth() -> np.ndarray:
    truth = np.zeros((HEIGHT, WIDTH), dtype=bool)
    y0, y1, x0, x1 = SUBJECT
    truth[y0:y1, x0:x1] = True
    return truth


def test_subject_mask_matches_ground_truth_rectangle():
    """Mask resolver khớp hình chữ nhật chủ thể (IoU cao) trên ảnh tổng hợp."""
    diagnosis = DiagnosisReport(
        subjects=["person"],
        subject_boxes={"person": BOX_2D},
        defects=[Defect(type="backlit_subject", region="person", severity=3, evidence="t")],
    )
    _, masks = diagnosis_region_overlay(_image(), diagnosis)
    assert "person" in masks
    predicted = masks["person"].mask >= 0.5
    truth = _truth()
    iou = (predicted & truth).sum() / (predicted | truth).sum()
    assert iou > 0.8


def test_overlay_tints_region_and_leaves_outside_untouched():
    image = _image()
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    y0, y1, x0, x1 = SUBJECT
    mask[y0:y1, x0:x1] = 1.0
    subject = SubjectMask(mask=mask, source="grabcut", confidence=0.4, box=None)
    overlay = render_region_overlay(image, {"person": subject})
    assert overlay.shape == image.shape and overlay.dtype == np.uint8
    # Trong vùng đổi màu, ngoài vùng (xa viền và nhãn) giữ nguyên
    assert not np.array_equal(
        overlay[y0 + 20 : y1 - 20, x0 + 20 : x1 - 20], image[y0 + 20 : y1 - 20, x0 + 20 : x1 - 20]
    )
    assert np.array_equal(overlay[-30:, -30:], image[-30:, -30:])


def test_overlay_without_regions_returns_downscaled_image_and_handles_gray():
    gray = cv2.cvtColor(_image(), cv2.COLOR_RGB2GRAY)
    overlay = render_region_overlay(gray, {})
    assert overlay.shape == (HEIGHT, WIDTH, 3)
    big = np.zeros((2048, 1024, 3), dtype=np.uint8)
    assert max(render_region_overlay(big, {}).shape[:2]) == 1024
