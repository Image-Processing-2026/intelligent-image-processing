"""
Tải bộ ảnh thật dùng để đánh giá module brain (perceive + plan) từ Wikimedia Commons.

Danh sách ảnh, nguồn, giấy phép và đáp án mong đợi nằm trong data/real_benchmark_manifest.json.
Ảnh được lưu vào data/real/benchmark/<nhóm>/ (thư mục này bị .gitignore), cạnh dài tối đa
`max_side` px trong manifest. Ảnh đã có sẵn thì bỏ qua.

Chạy từ thư mục gốc repo:
    python data/download_real_benchmark.py
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict

from PIL import Image

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "real_benchmark_manifest.json"
OUT_DIR = ROOT / "real" / "benchmark"
API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia yêu cầu User-Agent mô tả rõ client
USER_AGENT = "AIImageDoctor-benchmark/0.1 (university course project; python urllib)"
MAX_RETRIES = 6


def _fetch(url: str) -> bytes:
    """Tải URL, tự chờ và thử lại khi bị giới hạn tốc độ (HTTP 429)."""
    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            delay = int(exc.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
            logger.warning("Rate limited by Commons; retrying in %ds.", delay)
            time.sleep(delay)
    raise RuntimeError(f"Still rate limited after {MAX_RETRIES} attempts: {url}")


def _scaled_url(entry: Dict[str, Any], max_side: int) -> str:
    """URL bản thu nhỏ của Commons sao cho cạnh dài <= max_side; ảnh nhỏ hơn thì lấy bản gốc."""
    width, height = entry["original_size"]
    if max(width, height) <= max_side:
        return entry["original_url"]
    target_width = max(1, round(width * max_side / max(width, height)))
    params = {
        "action": "query",
        "titles": entry["commons_title"],
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": target_width,
        "format": "json",
        "formatversion": "2",
    }
    data = json.loads(_fetch(API + "?" + urllib.parse.urlencode(params)))
    return data["query"]["pages"][0]["imageinfo"][0]["thumburl"]


def _fit_max_side(path: Path, max_side: int) -> None:
    """Thu nhỏ ảnh tại chỗ nếu cạnh dài > max_side (Commons chỉ trả các bậc kích thước chuẩn)."""
    with Image.open(path) as image:
        if max(image.size) <= max_side:
            return
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        image.save(path, "JPEG", quality=95)


def download_all() -> int:
    """Tải mọi ảnh trong manifest còn thiếu; trả về số ảnh tải mới."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    max_side = int(manifest.get("max_side", 1600))
    downloaded = 0
    for entry in manifest["images"]:
        path = OUT_DIR / entry["file"]
        if path.exists():
            _fit_max_side(path, max_side)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(_fetch(_scaled_url(entry, max_side)))
            _fit_max_side(path, max_side)
            downloaded += 1
            logger.info("Downloaded %s", entry["file"])
        except Exception as exc:
            logger.error("Failed to download %s (%s): %s", entry["id"], entry["source_page"], exc)
        # Giữ nhịp chậm để không bị Commons chặn
        time.sleep(1.5)
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = download_all()
    logger.info("Done: %d new image(s) in %s", count, OUT_DIR)
