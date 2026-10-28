"""
Vẽ các vùng (mask chủ thể) lên ảnh gốc để người dùng và nhóm kiểm tra vùng có khoanh đúng không.
Chỉ để NHÌN (ADR-001: không phải pixel đầu ra). Mask lấy từ cùng resolver mà Perceive và
executor dùng (cache theo ảnh gốc), nên bản vẽ chính là vùng hệ thống thực sự dùng.
"""

import logging
from typing import Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from src.region_engine.subject import SubjectMask

from .executor import executed_mask, image_cache_key, subject_cache_key
from .imaging import downscale
from .perception import resolve_subject_masks
from .regions import canonical_region
from .state import DiagnosisReport, ExecutedAction

logger = logging.getLogger(__name__)

OVERLAY_MAX_SIDE = 1024
FILL_ALPHA = 0.35
LOW_CONFIDENCE = 0.5  # dưới mức này nhãn có dấu "?" để cảnh báo vùng có thể chưa chuẩn
# Màu RGB cho từng vùng, xoay vòng
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (230, 57, 70),
    (29, 140, 255),
    (46, 196, 100),
    (255, 183, 3),
    (156, 89, 209),
    (0, 188, 212),
)


def _ascii(text: str) -> str:
    """cv2.putText chỉ vẽ được ASCII; bỏ ký tự có dấu và ký tự khác."""
    return text.encode("ascii", "ignore").decode("ascii").strip()


def _label(name: str, subject: SubjectMask) -> str:
    mark = "?" if subject.confidence < LOW_CONFIDENCE else ""
    return f"{_ascii(name) or 'region'} [{subject.source} {subject.confidence:.2f}]{mark}"


def render_region_overlay(
    image: np.ndarray,
    masks: Mapping[str, SubjectMask],
    max_side: int = OVERLAY_MAX_SIDE,
) -> np.ndarray:
    """
    Ảnh RGB uint8 (≤ max_side) với mỗi vùng tô màu bán trong suốt theo trọng số mask, viền ở
    mức 0.5 và nhãn "tên [nguồn độ-tin-cậy]". Không có vùng nào → trả ảnh thu nhỏ nguyên vẹn.
    """
    base = downscale(image, max_side)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    canvas = base.astype(np.float32)
    height, width = canvas.shape[:2]
    outlines: List[Tuple[List[np.ndarray], Tuple[int, int, int], str]] = []

    for index, (name, subject) in enumerate(masks.items()):
        mask = subject.mask[..., 0] if subject.mask.ndim == 3 else subject.mask
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height))
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        color = PALETTE[index % len(PALETTE)]
        weight = (mask * FILL_ALPHA)[..., None]
        canvas = canvas * (1.0 - weight) + np.array(color, dtype=np.float32) * weight
        contours, _ = cv2.findContours(
            (mask >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outlines.append((list(contours), color, _label(name, subject)))

    result = np.clip(canvas, 0, 255).astype(np.uint8)
    thickness = max(1, round(max(height, width) / 500))
    scale = max(0.4, max(height, width) / 1600)
    for contours, color, label in outlines:
        cv2.drawContours(result, contours, -1, color, thickness)
        if not contours:
            continue
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        origin = (max(2, x), max(round(14 * scale * 2), y - 4))
        cv2.putText(
            result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2
        )
        cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return result


def diagnosis_region_overlay(
    original_image: np.ndarray,
    diagnosis: DiagnosisReport,
    regions: Optional[List[str]] = None,
    max_side: int = OVERLAY_MAX_SIDE,
) -> Tuple[np.ndarray, Dict[str, SubjectMask]]:
    """
    Overlay các vùng của chẩn đoán trên ảnh gốc, dùng cùng cache mask với Perceive/executor.
    regions: giới hạn các vùng cần vẽ (mặc định: subjects của chẩn đoán và vùng của các lỗi,
    trừ 'full'). Trả (ảnh overlay, mask các vùng định vị được).
    """
    wanted = regions
    if wanted is None:
        wanted = list(diagnosis.subjects) + [defect.region for defect in diagnosis.defects]
    names = sorted({canonical_region(name) for name in wanted} - {"full"})
    boxes = {canonical_region(name): box for name, box in diagnosis.subject_boxes.items()}
    masks = resolve_subject_masks(
        original_image,
        names,
        boxes,
        subject_cache_key(image_cache_key(original_image), original_image.shape),
    )
    return render_region_overlay(original_image, masks, max_side), masks


def executed_region_overlay(
    image: np.ndarray,
    executed: List[ExecutedAction],
    max_side: int = OVERLAY_MAX_SIDE,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Overlay đúng các mask đã thực thi (tra theo ExecutedAction.mask_hash), không dựng lại bằng
    đường khác. Action bị bỏ qua, vùng toàn ảnh hoặc mask đã bị đẩy khỏi cache không được vẽ.
    Trả (ảnh overlay, {tên vùng: mask_hash đã vẽ}).
    """
    masks: Dict[str, SubjectMask] = {}
    drawn: Dict[str, str] = {}
    for item in executed:
        if item.skipped or item.mask_hash is None:
            continue
        mask = executed_mask(item.mask_hash)
        if mask is None:
            logger.info(
                "Executed mask %s for '%s' is no longer cached.", item.mask_hash, item.region_id
            )
            continue
        name = f"{item.region_id}:{item.operation}"
        masks[name] = SubjectMask(mask, item.actual_mode or item.backend or "executed", 1.0, None)  # type: ignore[arg-type]
        drawn[name] = item.mask_hash
    return render_region_overlay(image, masks, max_side), drawn
