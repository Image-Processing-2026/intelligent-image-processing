"""
Module Region Engine & Mask Synthesis.
Cung cấp các công cụ phân đoạn ngữ nghĩa, nhận diện khuôn mặt và tạo mặt nạ mềm.
"""

from .detector import segment_by_prompt
from .face_detector import detect_faces
from .mask_utils import blend_regions, create_soft_mask
from .spatial import create_bbox_mask, create_quadrant_mask

__all__ = [
    "segment_by_prompt",
    "detect_faces",
    "create_soft_mask",
    "blend_regions",
    "create_bbox_mask",
    "create_quadrant_mask",
]
