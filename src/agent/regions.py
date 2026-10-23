"""
Từ vựng tên vùng dùng chung cho executor, planner và perception.
Một nơi duy nhất quyết định "khuôn mặt" và "face" là cùng một vùng, "toàn bộ" và "full"
là toàn ảnh, để preserve guard, chẩn đoán và executor không hiểu khác nhau.
"""

FACE_TARGETS = frozenset(("face", "faces", "khuôn mặt", "khuôn mặt người"))
FULL_TARGETS = frozenset(("full", "all", "toàn", "toàn bộ", "toàn ảnh", "full_image"))
SPATIAL_TARGETS = frozenset(("top", "bottom", "left", "right", "center", "giữa"))


def canonical_region(name: object) -> str:
    """
    Tên vùng chuẩn hóa: rỗng/đồng nghĩa toàn ảnh → 'full', đồng nghĩa khuôn mặt → 'face',
    còn lại giữ nguyên ở dạng chữ thường đã bỏ khoảng trắng hai đầu.
    """
    region = name.strip().casefold() if isinstance(name, str) else ""
    if not region or region in FULL_TARGETS:
        return "full"
    if region in FACE_TARGETS:
        return "face"
    return region
