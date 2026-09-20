"""
Module Region Engine & Mask Synthesis.
Cung cấp các công cụ phân đoạn ngữ nghĩa, nhận diện khuôn mặt và tạo mặt nạ mềm.
"""

from .detector import segment_by_prompt
from .controller import (
    InvalidRegionRequestError,
    RegionBackendUnavailableError,
    RegionEngineError,
    RegionInferenceError,
    RegionRequest,
    RegionResult,
    capabilities,
    resolve_region,
)
from .face_detector import (
    FaceDetectionError,
    FaceDetectionRecord,
    FaceDetectorUnavailableError,
    close_face_detector,
    detect_faces,
    reset_face_detector,
)
from .mask_utils import blend_regions, create_soft_mask
from .segmentation_backend import (
    SegmentationInferenceError,
    SegmentationUnavailableError,
    close_segmentation_backend,
    reset_segmentation_backend,
)
from .spatial import create_bbox_mask, create_quadrant_mask

__all__ = [
    "segment_by_prompt",
    "InvalidRegionRequestError",
    "RegionBackendUnavailableError",
    "RegionEngineError",
    "RegionInferenceError",
    "RegionRequest",
    "RegionResult",
    "capabilities",
    "resolve_region",
    "SegmentationInferenceError",
    "SegmentationUnavailableError",
    "close_segmentation_backend",
    "reset_segmentation_backend",
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
