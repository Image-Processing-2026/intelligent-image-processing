"""
Phân đoạn ngữ nghĩa dựa trên câu lệnh văn bản (Open-Vocabulary Semantic Segmentation).
Tích hợp GroundingDINO và MobileSAM/SAM2 để tạo mặt nạ vùng theo yêu cầu của Agent (ví dụ: 'sky', 'background').
"""

import numpy as np

from .spatial import create_quadrant_mask


def segment_by_prompt(image: np.ndarray, text_prompt: str, feather_radius: int = 15) -> np.ndarray:
    """
    Chuyển đổi mô tả văn bản thành mặt nạ vùng mềm (soft-mask).

    Args:
        image: Ảnh đầu vào RGB np.uint8.
        text_prompt: Từ khóa đối tượng (vd: 'sky', 'face', 'background', 'person').
        feather_radius: Độ mịn viền mặt nạ.

    Returns:
        np.ndarray: Soft mask float32 [0.0, 1.0].
    """
    h, w = image.shape[:2]
    prompt_lower = text_prompt.lower().strip()

    # Rule-based heuristics / Fallback thông minh khi chưa nạp SAM weights
    if "sky" in prompt_lower or "trời" in prompt_lower:
        return create_quadrant_mask((h, w), "top", feather_radius=feather_radius)
    elif "ground" in prompt_lower or "đất" in prompt_lower or "floor" in prompt_lower:
        return create_quadrant_mask((h, w), "bottom", feather_radius=feather_radius)
    elif "center" in prompt_lower or "giữa" in prompt_lower:
        return create_quadrant_mask((h, w), "center", feather_radius=feather_radius)
    elif "full" in prompt_lower or "toàn" in prompt_lower or "all" in prompt_lower:
        return np.ones((h, w), dtype=np.float32)

    # TODO (Person 2): Tích hợp pipeline MobileSAM / GroundingDINO inference tại đây
    # Ví dụ:
    # boxes = grounding_dino_predict(image, text_prompt)
    # masks = mobile_sam_predict(image, boxes)
    # return create_soft_mask(masks[0], feather_radius=feather_radius)

    # Mặc định trả về toàn bộ ảnh nếu không tìm thấy vùng đặc thù
    return np.ones((h, w), dtype=np.float32)
