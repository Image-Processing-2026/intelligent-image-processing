"""Tải bộ ảnh thật basic đã chọn và tạo input, gallery cùng biên nhận kiểm chứng.

Chạy từ gốc repo: .venv/Scripts/python.exe data/download_basic_test_set.py
Không gọi Gemini, không chạy pipeline, không sinh ảnh suy giảm nhân tạo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/basic_test_manifest.json"
USER_AGENT = "AIImageDoctor-BasicTestSet/1.0 (university image processing project)"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    """Chỉ cho phép đường dẫn file nằm trong thư mục bộ ảnh."""
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return target


def _download(url: str, path: Path, expected_bytes: int | None, expected_sha256: str | None) -> None:
    """Giữ nguyên byte nguồn; chỉ ghi file hoàn chỉnh sau khi tải xong."""
    if (path.exists() and (expected_bytes is None or path.stat().st_size == expected_bytes)
            and (expected_sha256 is None or _sha256(path) == expected_sha256)):
        return
    if not url.startswith(("https://upload.wikimedia.org/", "https://thumb.wikimedia.org/")):
        raise ValueError(f"Unexpected download host: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=50) as response:
                payload = response.read()
            if expected_bytes is not None and len(payload) != expected_bytes:
                raise ValueError(f"Source size changed for {path.name}: {len(payload)} != {expected_bytes}")
            if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
                raise ValueError(f"Source checksum changed for {path.name}")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Không tiếp tục gửi hàng loạt request khi nguồn đang giới hạn tốc độ.
                raise RuntimeError("Source rate limited; pause before retrying") from exc
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def _prepare(root: Path, row: dict, verify_only: bool) -> dict:
    """Tách ảnh gốc khỏi input; crop nhiễu 1:1 để không làm mất hạt do resize."""
    original = _inside(root, row["source_file"])
    target = _inside(root, row["file"])
    preview = _inside(root, row["preview_file"])
    if not verify_only:
        _download(row["download_url"], original, row["download_bytes"], row.get("source_sha256"))
    if row["download_bytes"] is not None and original.stat().st_size != row["download_bytes"]:
        raise ValueError(f"Invalid original file size: {row['id']}")
    if row.get("source_sha256") and _sha256(original) != row["source_sha256"]:
        raise ValueError(f"Source checksum mismatch: {row['id']}")
    with Image.open(original) as source:
        source.load()
        if list(source.size) != row["download_size"]:
            raise ValueError(f"Source dimensions changed: {row['id']}")
        image = ImageOps.exif_transpose(source).convert("RGB")
    before = list(image.size)
    crop_box = None
    if row["input_policy"] == "native_crop_1600x1200":
        width, height = min(1600, image.width), min(1200, image.height)
        left, top = (image.width - width) // 2, (image.height - height) // 2
        if row.get("crop_anchor") == "bottom":
            top = image.height - height
        crop_box = [left, top, left + width, top + height]
        image = image.crop(crop_box)
    else:
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    if not verify_only:
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "PNG")
        preview.parent.mkdir(parents=True, exist_ok=True)
        small = image.copy()
        small.thumbnail((440, 320), Image.Resampling.LANCZOS)
        small.save(preview, "JPEG", quality=90)
    with Image.open(target) as actual:
        actual.load()
        if actual.mode != "RGB" or actual.size != image.size or actual.tobytes() != image.tobytes():
            raise ValueError(f"Input does not match declared transformation: {row['id']}")
    with Image.open(preview) as thumb:
        thumb.verify()
    return {"id": row["id"], "source_sha256": _sha256(original), "input_sha256": _sha256(target),
            "source_variant": row["source_variant"], "oriented_source_size": before, "input_size": list(image.size), "crop_box": crop_box,
            "resampled": row["input_policy"] != "native_crop_1600x1200" and before != list(image.size),
            "downloaded_source_bytes": original.stat().st_size, "input_bytes": target.stat().st_size,
            "decode_verified": True, "transformation_verified": True}


def _gallery(root: Path, rows: list[dict], receipts: list[dict]) -> None:
    """Gallery HTML xem ảnh, intent, giới hạn và attribution cho từng ca."""
    info = {item["id"]: item for item in receipts}
    cards = []
    for row in rows:
        esc = html.escape
        size = info[row["id"]]["input_size"]
        cards.append(
            f'<article data-tier="{esc(row["tier"])}" data-category="{esc(row["category"])}">'
            f'<a href="{esc(row["file"])}"><img loading="lazy" src="{esc(row["preview_file"])}" alt="{esc(row["id"])}"></a>'
            f'<h2>{esc(row["id"])}</h2><b>{esc(row["tier"])} · {size[0]}×{size[1]}</b>'
            f'<p>{esc(row["intent"])}</p><p class="note">{esc(row["note"])}</p>'
            f'<p><a href="{esc(row["file"])}">Input PNG</a> · <a href="{esc(row["source_file"])}">Downloaded source</a> · '
            f'<a href="{esc(row["source_page"])}">Source</a></p>'
            f'<small>{esc(row["author"].strip())} · {esc(row["license"])}</small></article>'
        )
    document = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Basic real-photo tests · 30 cases</title>
<style>body{font:16px/1.5 system-ui;margin:30px;background:#f4f6f8;color:#15232e}h1{margin-bottom:8px}
.intro{max-width:1000px}button{padding:9px 15px;margin:8px 5px 20px 0;cursor:pointer;border:1px solid #789;border-radius:6px;background:white}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}article{background:white;padding:15px;border-radius:12px}
img{width:100%;height:230px;object-fit:contain;background:#e9edf1}h2{font-size:17px}.note,small{color:#52616d}a{color:#126b92}</style>
<h1>30 ảnh thật · Basic test set v1</h1><p class="intro">16 basic · 4 secondary · 7 control · 3 boundary.
Nhãn do agent đề xuất, chưa được người dùng duyệt. Đây là bộ phát triển/test thủ công, không phải hold-out độc lập.
Ảnh nhiễu dùng crop 1:1; các input khác có cạnh dài tối đa 1600px. Bấm ảnh để xem input thực.</p>
<nav><button onclick="filter('all')">Tất cả</button><button onclick="filter('basic')">Basic</button>
<button onclick="filter('secondary')">Secondary</button><button onclick="filter('control')">Giữ nguyên</button>
<button onclick="filter('boundary')">Biết giới hạn</button></nav><main>""" + "\n".join(cards) + """</main>
<script>function filter(t){document.querySelectorAll('article').forEach(e=>e.hidden=t!=='all'&&e.dataset.tier!==t)}</script></html>"""
    document = document.replace("16 basic · 4 secondary · 7 control · 3 boundary.",
        " · ".join(f"{count} {tier}" for tier, count in Counter(row["tier"] for row in rows).items()) + ".")
    (root / "index.html").write_text(document, encoding="utf-8")
    for page in range(0, len(rows), 15):
        batch = rows[page:page + 15]
        sheet = Image.new("RGB", (1320, 380 * ((len(batch) + 2) // 3)), "#f4f6f8")
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(batch):
            x, y = idx % 3 * 440, idx // 3 * 380
            thumb = ImageOps.contain(Image.open(root / row["preview_file"]), (420, 310))
            sheet.paste(thumb, (x + (440 - thumb.width) // 2, y + 35))
            draw.text((x + 8, y + 5), row["id"], fill="#15232e")
            draw.text((x + 8, y + 350), row["tier"] + " | " + row["license"], fill="#15232e")
        sheet.save(root / f"contact_sheet_{page // 15 + 1:02d}.jpg", quality=92)
    checklist = root / "results_template.csv"
    if not checklist.exists():
        with checklist.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case_id", "tier", "category", "intent", "run_id", "target_improved", "preservation_passed", "artifacts", "accepted", "reviewer", "notes"])
            for row in rows:
                writer.writerow([row["id"], row["tier"], row["category"], row["intent"], "", "", "", "", "", "", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the curated basic real-photo test set")
    parser.add_argument("--verify-only", action="store_true", help="Verify local originals and inputs without network")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["images"]
    root = _inside(ROOT, manifest["dataset_root"])
    root.mkdir(parents=True, exist_ok=True)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case ID")
    existing_titles = set()
    for name in manifest["excluded_existing_manifests"]:
        prior = json.loads((ROOT / name).read_text(encoding="utf-8"))
        existing_titles.update(row.get("commons_title", "").replace("_", " ") for row in prior["images"])
    if any(row["commons_title"].replace("_", " ") in existing_titles for row in rows):
        raise ValueError("Selected source overlaps with an existing development/hold-out manifest")
    receipts, failures = [], []
    # Tải tuần tự để giảm áp lực lên Commons; bản gốc được giãn cách hơn thumbnail.
    last_original_request = 0.0
    work_rows = sorted(rows, key=lambda row: row["source_variant"] == "original" and not _inside(root, row["source_file"]).exists())
    for index, row in enumerate(work_rows):
        needs_download = not _inside(root, row["source_file"]).exists() and not args.verify_only
        if needs_download and row["source_variant"] == "original":
            time.sleep(max(0.0, 25.0 - (time.monotonic() - last_original_request)))
            last_original_request = time.monotonic()
        try:
            receipt = _prepare(root, row, args.verify_only)
            receipts.append(receipt)
            print(f"OK {row['id']} {receipt['input_size']}", flush=True)
        except Exception as exc:
            failures.append({"id": row["id"], "error": str(exc)})
            print(f"FAILED {row['id']}: {exc}", flush=True)
            if "rate limited" in str(exc):
                # Dừng lượt này ngay; có thể tiếp tục từ các file đã tải ở lượt sau.
                failures.extend({"id": rest["id"], "error": "Skipped after source rate limit"}
                                for rest in work_rows[index + 1:])
                break
        if needs_download:
            time.sleep(1.5)
    order = {row["id"]: idx for idx, row in enumerate(rows)}
    receipts.sort(key=lambda row: order[row["id"]])
    originals = [row["source_sha256"] for row in receipts]
    if len(set(originals)) != len(originals):
        raise ValueError("Byte-identical duplicate downloaded sources")
    report = {"dataset_id": manifest["dataset_id"], "manifest_sha256": _sha256(MANIFEST),
              "verified_at_local_date": "2026-10-27", "downloaded": len(receipts), "failed": failures,
              "counts_by_tier": dict(Counter(row["tier"] for row in rows)),
              "counts_by_category": dict(Counter(row["category"] for row in rows)),
              "pipeline_run": False, "human_labels_verified": False,
              "existing_source_title_overlap": 0, "exact_duplicate_sources": 0, "files": receipts}
    (root / "download_receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"Incomplete: {len(receipts)}/{len(rows)} verified; rerun to retry failed downloads")
    _gallery(root, rows, receipts)
    print(f"Complete: {len(receipts)} verified photos in {root}", flush=True)


if __name__ == "__main__":
    main()
