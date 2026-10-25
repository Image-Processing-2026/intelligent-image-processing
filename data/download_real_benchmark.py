"""
Tải bộ ảnh thật dùng để đánh giá module brain (perceive + plan) từ Wikimedia Commons.

Danh sách ảnh, nguồn, giấy phép và đáp án mong đợi nằm trong data/real_benchmark_manifest.json.
Ảnh được lưu vào data/real/benchmark/<nhóm>/ (thư mục này bị .gitignore), cạnh dài tối đa
`max_side` px trong manifest. Ảnh đã có sẵn thì bỏ qua. Ảnh có trường `fullres_file` (nhóm noisy)
được tải thêm bản gốc độ phân giải đầy đủ vào data/real/benchmark_fullres/, không thu nhỏ.

Bộ hold-out (data/holdout_manifest.json, chỉ để chấm kết quả cuối, không chỉnh trên đó) tải vào
data/real/holdout/ bằng cờ --holdout.

Chạy từ thư mục gốc repo:
    python data/download_real_benchmark.py
    python data/download_real_benchmark.py --holdout
"""

import argparse
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
FULLRES_DIR = ROOT / "real" / "benchmark_fullres"
HOLDOUT_MANIFEST = ROOT / "holdout_manifest.json"
HOLDOUT_DIR = ROOT / "real" / "holdout"
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
    info = data["query"]["pages"][0].get("imageinfo")
    if not info:
        # Commons không tạo được bản thu nhỏ cho kích thước này: lấy bản gốc, _fit_max_side sẽ thu nhỏ
        logger.warning("No thumbnail for %s; using the original.", entry["commons_title"])
        return entry["original_url"]
    return info[0]["thumburl"]


def _fit_max_side(path: Path, max_side: int) -> None:
    """Thu nhỏ ảnh tại chỗ nếu cạnh dài > max_side (Commons chỉ trả các bậc kích thước chuẩn)."""
    with Image.open(path) as image:
        if max(image.size) <= max_side:
            return
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        image.save(path, "JPEG", quality=95)


def download_fullres() -> int:
    """Tải bản gốc full-res của các ảnh có `fullres_file` còn thiếu; trả về số ảnh tải mới."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    downloaded = 0
    for entry in manifest["images"]:
        if "fullres_file" not in entry:
            continue
        path = FULLRES_DIR / entry["fullres_file"].removeprefix("benchmark_fullres/")
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(_fetch(entry["original_url"]))
            downloaded += 1
            logger.info("Downloaded full-res %s", path.name)
        except Exception as exc:
            logger.error("Failed to download full-res of %s: %s", entry["id"], exc)
        time.sleep(1.5)
    return downloaded


def download_all(manifest_path: Path = MANIFEST, out_dir: Path = OUT_DIR) -> int:
    """Tải mọi ảnh trong manifest còn thiếu; trả về số ảnh tải mới."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_side = int(manifest.get("max_side", 1600))
    downloaded = 0
    for entry in manifest["images"]:
        path = out_dir / entry["file"]
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
    parser = argparse.ArgumentParser(description="Download benchmark images from Commons.")
    parser.add_argument("--holdout", action="store_true", help="Tải bộ hold-out thay bộ benchmark")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.holdout:
        count = download_all(HOLDOUT_MANIFEST, HOLDOUT_DIR)
        logger.info("Done: %d new image(s) in %s", count, HOLDOUT_DIR)
        raise SystemExit(0)
    count = download_all()
    logger.info("Done: %d new image(s) in %s", count, OUT_DIR)
    fullres = download_fullres()
    logger.info("Done: %d new full-res image(s) in %s", fullres, FULLRES_DIR)
