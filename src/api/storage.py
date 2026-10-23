"""
Lưu trữ ảnh của web app trên đĩa: kiểm tra và chuẩn hóa ảnh upload (không tin cậy), ghi PNG /
thumbnail vào data/app/users/<user_id>/sessions/<session_id>/, xóa thư mục phiên an toàn.
Mọi đường dẫn được ghép từ id do server sinh; tên file của client không bao giờ thành đường dẫn.
"""

import logging
import re
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import AppSettings
from .errors import ApiError

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
THUMBNAIL_SIDE = 256
READ_CHUNK = 1024 * 1024
TITLE_MAX = 120
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class SavedImage:
    """Metadata của một file ảnh vừa ghi."""

    rel_path: str
    width: int
    height: int
    bytes: int


def read_limited(stream: BinaryIO, settings: AppSettings) -> bytes:
    """Đọc file upload theo khối, dừng ngay khi vượt MAX_UPLOAD_MB (→ 413)."""
    limit = settings.max_upload_bytes
    chunks = []
    total = 0
    while True:
        chunk = stream.read(READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ApiError(413, "file_too_large", f"Tệp vượt quá {settings.max_upload_mb} MB.")
        chunks.append(chunk)
    if total == 0:
        raise ApiError(422, "invalid_image")
    return b"".join(chunks)


def decode_image(data: bytes, settings: AppSettings) -> np.ndarray:
    """
    Giải mã ảnh không tin cậy thành RGB uint8 (H, W, 3):
    1. Định dạng xác định từ nội dung (không tin đuôi file/Content-Type): JPEG, PNG, WEBP.
    2. Kiểm tra số pixel TRƯỚC khi giải mã toàn bộ (chống decompression bomb).
    3. Xoay theo EXIF, chuyển RGB (nền trắng cho ảnh có kênh alpha).
    """
    try:
        image = Image.open(BytesIO(data))
        image_format = image.format
    except (UnidentifiedImageError, OSError, ValueError):
        raise ApiError(415, "unsupported_format")
    except Image.DecompressionBombError:
        raise ApiError(422, "invalid_image")
    if image_format not in ALLOWED_FORMATS:
        raise ApiError(415, "unsupported_format")
    width, height = image.size
    if width * height > settings.max_image_pixels:
        raise ApiError(422, "invalid_image")
    try:
        image.load()
        image = ImageOps.exif_transpose(image) or image
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError, SyntaxError):
        raise ApiError(422, "invalid_image")
    if min(image.size) < settings.min_image_side:
        raise ApiError(422, "image_too_small")
    return np.asarray(image, dtype=np.uint8).copy()


def session_dir(settings: AppSettings, user_id: str, session_id: str) -> Path:
    """Thư mục của một phiên xử lý (id do server sinh, dạng hex)."""
    for value in (user_id, session_id):
        if not re.fullmatch(r"[0-9a-f]{32}", value):
            raise ValueError("Invalid id for storage path")
    return settings.users_dir / user_id / "sessions" / session_id


def save_png(image: np.ndarray, base: Path, rel_path: str) -> SavedImage:
    """Ghi ảnh RGB uint8 dạng PNG vào base/rel_path."""
    target = base / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(target, format="PNG")
    return SavedImage(rel_path, int(image.shape[1]), int(image.shape[0]), target.stat().st_size)


def save_thumbnail(image: np.ndarray, base: Path, rel_path: str) -> SavedImage:
    """Ghi thumbnail JPEG (cạnh dài ≤ THUMBNAIL_SIDE) cho sidebar."""
    target = base / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    thumb = Image.fromarray(image)
    thumb.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
    thumb.save(target, format="JPEG", quality=82)
    return SavedImage(rel_path, thumb.width, thumb.height, target.stat().st_size)


def thumbnail_bytes(path: Path) -> bytes:
    """Thumbnail JPEG của một file ảnh có sẵn (ảnh mẫu), tạo trong bộ nhớ."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image) or image
        thumb = image.convert("RGB")
        thumb.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
        buffer = BytesIO()
        thumb.save(buffer, format="JPEG", quality=82)
        return buffer.getvalue()


def load_rgb(path: Path) -> np.ndarray:
    """Đọc lại một PNG do server đã ghi (ảnh gốc đã chuẩn hóa) thành RGB uint8."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def resolve_inside(base: Path, rel_path: str) -> Optional[Path]:
    """Đường dẫn tuyệt đối của rel_path nếu nó nằm trong base, ngược lại None."""
    root = base.resolve()
    candidate = (root / rel_path).resolve()
    return candidate if candidate.is_relative_to(root) else None


def delete_tree(settings: AppSettings, directory: Path) -> None:
    """Xóa thư mục phiên; chỉ xóa khi nó nằm trong users_dir (phòng lỗi ghép đường dẫn)."""
    root = settings.users_dir.resolve()
    target = directory.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Refusing to delete a path outside the users directory")
    if target.exists():
        shutil.rmtree(target)


def clean_title(raw: Optional[str]) -> str:
    """Tên phiên an toàn: bỏ ký tự điều khiển, gộp khoảng trắng, cắt TITLE_MAX ký tự."""
    text = _CONTROL_CHARS.sub(" ", raw or "")
    return " ".join(text.split())[:TITLE_MAX].strip()


def title_from_filename(filename: Optional[str]) -> str:
    """Tên phiên mặc định từ tên file client (chỉ để hiển thị, không dùng làm đường dẫn)."""
    name = Path((filename or "").replace("\\", "/")).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return clean_title(stem)
