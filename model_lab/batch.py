"""Batch run of the free-vision protocol over the real-photo benchmark.

Reuses the lab's prepare_image/call_model so every model gets the same PNG bytes and the
same shared system prompt as in the UI. Only the output-token cap is raised (default
4000) so long answers are not cut off, and transient provider errors are retried.

From the intelligent-image-processing folder:
    .venv\\Scripts\\python.exe model_lab\\batch.py
    .venv\\Scripts\\python.exe model_lab\\batch.py --ids backlit_01 --providers google
    .venv\\Scripts\\python.exe model_lab\\batch.py --resume model_lab\\batch_runs\\<folder>

Output: model_lab/batch_runs/<timestamp>/<image_id>/{image.png,<provider>.json}
plus run.json (settings) and progress.log. Existing answers are skipped on resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from app import MODELS, call_model, prepare_image
from prompts import SYSTEM_PROMPT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = ROOT / "data" / "real_benchmark_manifest.json"
IMAGES = ROOT / "data" / "real" / "benchmark"
OUT = HERE / "batch_runs"
PROTOCOL_VERSION = "vision-free-v1-batch"
# Same spacing as the UI: Groq's output-token/minute quota is the bottleneck.
GAPS = {"groq": 65, "gemini": 13, "google": 3}
RETRY_WAITS = (20, 45, 90, 180)
RETRYABLE = re.compile(r"HTTP (429|5\d\d)\b|over capacity|overloaded|high demand", re.I)

log = logging.getLogger("batch")


def retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return bool(RETRYABLE.search(str(exc)))


def is_done(target: Path) -> bool:
    if not target.exists():
        return False
    return json.loads(target.read_text(encoding="utf-8"))["status"] == "done"


def prepared_image(entry: dict, folder: Path, max_edge: int) -> tuple[bytes, dict]:
    path = folder / "image.png"
    meta_path = folder / "image.json"
    if path.exists() and meta_path.exists():
        return path.read_bytes(), json.loads(meta_path.read_text(encoding="utf-8"))
    png, meta = prepare_image((IMAGES / entry["file"]).read_bytes(), max_edge)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return png, meta


def run_provider(
    provider: str, entries: list[dict], run_dir: Path, args: argparse.Namespace
) -> None:
    last_call = 0.0
    for index, entry in enumerate(entries, 1):
        folder = run_dir / entry["id"]
        target = folder / f"{provider}.json"
        if is_done(target):
            continue
        png, meta = prepared_image(entry, folder, args.max_edge)
        record = {
            "id": entry["id"],
            "provider": provider,
            "model": MODELS[provider]["model"],
            "image": meta,
            "attempts": [],
        }
        for attempt in range(len(RETRY_WAITS) + 1):
            wait = GAPS.get(provider, 3) - (time.monotonic() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.monotonic()
            try:
                result = call_model(provider, SYSTEM_PROMPT, None, png, args.max_tokens)
            except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
                last_call = time.monotonic()
                record["attempts"].append({"at": datetime.now(UTC).isoformat(), "error": str(exc)})
                if attempt < len(RETRY_WAITS) and retryable(exc):
                    log.warning(
                        "%s %s attempt %d failed (%s); retrying in %ds",
                        provider,
                        entry["id"],
                        attempt + 1,
                        str(exc)[:120],
                        RETRY_WAITS[attempt],
                    )
                    time.sleep(RETRY_WAITS[attempt])
                    continue
                record.update(status="error", error=str(exc))
                break
            last_call = time.monotonic()
            record.update(status="done", **result)
            break
        # Never replace a successful answer with a failure (e.g. one written by another run).
        if record["status"] != "done" and is_done(target):
            log.info("[%s] %s already has a successful answer; keeping it", provider, entry["id"])
            continue
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(
            "[%s %d/%d] %s %s%s",
            provider,
            index,
            len(entries),
            entry["id"],
            record["status"],
            " (TRUNCATED)" if record.get("truncated") else "",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--providers", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--ids", nargs="*", help="Only these manifest ids")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument("--resume", type=Path, help="Existing batch_runs/<folder> to continue")
    args = parser.parse_args()

    run_dir = args.resume or OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    # One run per folder: two runs double the provider quota use and race on the same files.
    lock = run_dir / ".lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"{lock} exists: another batch run is using this folder (delete it if stale)."
        )
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    try:
        run(run_dir, args)
    finally:
        lock.unlink(missing_ok=True)


def run(run_dir: Path, args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "progress.log", encoding="utf-8"),
        ],
    )
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]
    if args.ids:
        entries = [e for e in entries if e["id"] in set(args.ids)]
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "started_at": datetime.now(UTC).isoformat(),
                "system_prompt": SYSTEM_PROMPT,
                "max_output_tokens": args.max_tokens,
                "max_edge": args.max_edge,
                "models": {p: MODELS[p]["model"] for p in args.providers},
                "images": [e["id"] for e in entries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Prepare every PNG once up front so provider threads never write the same file.
    for entry in entries:
        prepared_image(entry, run_dir / entry["id"], args.max_edge)
    log.info("Batch %s: %d images x %s", run_dir.name, len(entries), args.providers)
    # Providers have independent quotas, so each walks the image list in its own thread.
    threads = [
        threading.Thread(target=run_provider, args=(p, entries, run_dir, args), name=p)
        for p in args.providers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.info("Batch finished: %s", run_dir)


if __name__ == "__main__":
    main()
