"""
Nhận diện khuôn mặt và tạo mặt nạ bằng MediaPipe (Chạy tối ưu trên CPU).
"""

from typing import List
import numpy as np
from .mask_utils import create_soft_mask


def detect_faces(image: np.ndarray, feather_radius: int = 20) -> List[np.ndarray]:
    """
    Phát hiện các vùng khuôn mặt bằng MediaPipe và trả về danh sách các mặt nạ mềm.

    Args:
        image: Ảnh đầu vào RGB np.uint8.
        feather_radius: Bán kính làm mịn viền.

    Returns:
        List[np.ndarray]: Danh sách các soft-masks float32 [0.0, 1.0].
    """
    h, w = image.shape[:2]
    masks = []

    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    xmin = int(bboxC.xmin * w)
                    ymin = int(bboxC.ymin * h)
                    box_w = int(bboxC.width * w)
                    box_h = int(bboxC.height * h)

                    # Mở rộng vùng mặt thêm 15% để bao quát cả trán và cằm
                    pad_x = int(box_w * 0.15)
                    pad_y = int(box_h * 0.15)
                    x1 = max(0, xmin - pad_x)
                    y1 = max(0, ymin - pad_y)
                    x2 = min(w, xmin + box_w + pad_x)
                    y2 = min(h, ymin + box_h + pad_y)

                    bin_mask = np.zeros((h, w), dtype=np.uint8)
                    bin_mask[y1:y2, x1:x2] = 255
                    masks.append(create_soft_mask(bin_mask, feather_radius=feather_radius))
    except Exception:
        # Fallback nếu môi trường chưa nạp được mediapipe
        pass

    return masks
