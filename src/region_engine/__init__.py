"""
Module Region Engine & Mask Synthesis.
Cung cấp các công cụ phân đoạn ngữ nghĩa, nhận diện khuôn mặt và tạo mặt nạ mềm.
"""

from .detector import segment_by_prompt
from .face_detector import (
    FaceDetectionError,
    FaceDetectionRecord,
    FaceDetectorUnavailableError,
    close_face_detector,
    detect_faces,
    reset_face_detector,
)
from .mask_utils import blend_regions, create_soft_mask
from .spatial import create_bbox_mask, create_quadrant_mask

__all__ = [
    "segment_by_prompt",
    "detect_faces",
    "FaceDetectionError",
    "FaceDetectionRecord",
    "FaceDetectorUnavailableError",
    "close_face_detector",
    "reset_face_detector",
    "create_soft_mask",
    "blend_regions",
    "create_bbox_mask",
    "create_quadrant_mask",
]
