"""Local image + system-instruction comparison. Keys stay on the Python server."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from prompts import SYSTEM_PROMPT

HERE = Path(__file__).resolve().parent
DEFAULT_ENV = HERE.parent / ".env"
ENV_PATH = Path(os.environ.get("MODEL_LAB_ENV", str(DEFAULT_ENV)))
RUNS = HERE / "runs"
MODELS = {
    "google": {
        "label": "Gemma 4 31B",
        "model": "gemma-4-31b-it",
        "key": "GEMINI_API_KEY",
        "api": "google",
    },
    "groq": {
        "label": "Qwen 3.8 27B",
        "model": "qwen/qwen3.8-27b",
        "key": "GROQ_API_KEY",
        "api": "groq",
    },
    "gemini": {
        "label": "Gemini 3.5 Flash-Lite",
        "model": "gemini-3.5-flash-lite",
        "key": "GEMINI_API_KEY",
        "api": "google",
    },
}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 32_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
app = FastAPI(title="Image Doctor · Model Lab", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
CANCEL: dict[str, threading.Event] = {}
POOL = ThreadPoolExecutor(max_workers=len(MODELS), thread_name_prefix="vlm")
PROVIDER_LOCKS = {name: threading.Lock() for name in MODELS}
LAST_CALL = {name: 0.0 for name in MODELS}
PROTOCOL_VERSION = "vision-free-v1"


def stage_names(job: dict) -> tuple[str, ...]:
    return tuple(job.get("stages", ("analysis", "plan")))


def credentials() -> dict:
    values = dotenv_values(ENV_PATH, encoding="utf-8-sig")
    return {
        name: os.environ.get(spec["key"]) or values.get(spec["key"])
        for name, spec in MODELS.items()
    }


def safe_error(message: str) -> str:
    for key in credentials().values():
        if key:
            message = message.replace(key, "[REDACTED]")
    return message[:1800]


def run_path(run_id: str) -> Path:
    try:
        if str(uuid.UUID(run_id)) != run_id:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(404, "Không tìm thấy lượt thử.") from exc
    return RUNS / run_id


def persist(job: dict) -> None:
    directory = run_path(job["id"])
    temp = directory / "result.tmp"
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(directory / "result.json")


def update(run_id: str, provider: str, stage: str, **values) -> None:
    with LOCK:
        job = JOBS[run_id]
        job["results"][provider][stage].update(values)
        states = [
            v[s]["status"] for v in job["results"].values() for s in stage_names(job)
        ]
        job["status"] = (
            "running"
            if any(s in {"pending", "running", "waiting"} for s in states)
            else "done"
        )
        persist(job)


def prepare_image(data: bytes, max_edge: int) -> tuple[bytes, dict]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                if opened.width * opened.height > MAX_PIXELS:
                    raise ValueError("Ảnh vượt quá 32 megapixel.")
                image = ImageOps.exif_transpose(opened)
                original_size = image.size
                # Composite transparency on white; use identical bytes for every model.
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(
            "Không đọc được ảnh. Dùng PNG, JPEG hoặc WebP, tối đa 32 megapixel."
        ) from exc
    png = buffer.getvalue()
    return png, {
        "original_size": list(original_size),
        "sent_size": list(image.size),
        "sha256": hashlib.sha256(png).hexdigest(),
        "bytes": len(png),
    }


def call_model(
    provider: str, system: str, text: str | None, image: bytes | None, limit: int
) -> dict:
    key = credentials().get(provider)
    if not key:
        raise ValueError(
            f"Thiếu {MODELS[provider]['key']} trong .env hoặc biến môi trường."
        )
    model = MODELS[provider]["model"]
    encoded = base64.b64encode(image).decode() if image else None
    if MODELS[provider]["api"] == "google":
        parts = [{"text": text}] if text else []
        if encoded:
            parts.insert(0, {"inlineData": {"mimeType": "image/png", "data": encoded}})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": key}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": limit,
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        }
    else:
        content = text
        if encoded:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
            if text:
                content.append({"type": "text", "text": text})
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.4,
            "max_completion_tokens": limit,
            "reasoning_effort": "none",
            "reasoning_format": "hidden",
        }
    started = time.monotonic()
    response = httpx.post(
        url, headers=headers, json=payload, timeout=httpx.Timeout(150, connect=20)
    )
    try:
        value = response.json()
    except ValueError as exc:
        raise ValueError(
            f"HTTP {response.status_code}: nhà cung cấp trả dữ liệu không phải JSON."
        ) from exc
    if response.is_error:
        detail = value.get("error", {})
        message = (
            detail.get("message", str(detail))
            if isinstance(detail, dict)
            else str(detail)
        )
        raise ValueError(f"HTTP {response.status_code}: {safe_error(message)}")
    if MODELS[provider]["api"] == "google":
        candidate = (value.get("candidates") or [{}])[0]
        answer = "".join(
            p.get("text", "")
            for p in candidate.get("content", {}).get("parts", [])
            if not p.get("thought")
        )
        reason = candidate.get("finishReason", "unknown")
        raw_usage = value.get("usageMetadata", {})
        usage = {
            "input": raw_usage.get("promptTokenCount"),
            "output": raw_usage.get("candidatesTokenCount"),
            "total": raw_usage.get("totalTokenCount"),
        }
    else:
        candidate = (value.get("choices") or [{}])[0]
        answer = candidate.get("message", {}).get("content") or ""
        reason = candidate.get("finish_reason", "unknown")
        raw_usage = value.get("usage", {})
        usage = {
            "input": raw_usage.get("prompt_tokens"),
            "output": raw_usage.get("completion_tokens"),
            "total": raw_usage.get("total_tokens"),
        }
    if not answer.strip():
        raise ValueError(
            f"Model không trả nội dung phân tích (finish_reason={reason})."
        )
    return {
        "text": answer,
        "seconds": round(time.monotonic() - started, 2),
        "usage": usage,
        "model_returned": value.get("modelVersion", value.get("model", model)),
        "finish_reason": reason,
        "truncated": reason in {"MAX_TOKENS", "length"},
    }


def wait_for_slot(run_id: str, provider: str, stage: str) -> bool:
    # This account's Groq pilot exposed a 1000 output-token/minute limit.
    # Space requests rather than silently reducing output or rotating credentials.
    gap = {"groq": 65, "gemini": 13}.get(provider, 3)
    while (remaining := gap - (time.monotonic() - LAST_CALL[provider])) > 0:
        update(
            run_id, provider, stage, status="waiting", wait_seconds=int(remaining) + 1
        )
        if CANCEL[run_id].wait(min(1, remaining)):
            return False
    return not CANCEL[run_id].is_set()


def worker(run_id: str, provider: str) -> None:
    with PROVIDER_LOCKS[provider]:
        job = JOBS[run_id]
        stage = "analysis"
        if not wait_for_slot(run_id, provider, stage):
            update(run_id, provider, stage, status="cancelled", wait_seconds=0)
            return
        update(run_id, provider, stage, status="running", wait_seconds=0, error=None)
        try:
            image = (run_path(run_id) / "image.png").read_bytes()
            LAST_CALL[provider] = time.monotonic()
            result = call_model(
                provider, job["system_prompt"], None, image, job["max_output_tokens"]
            )
            LAST_CALL[provider] = time.monotonic()
            update(run_id, provider, stage, status="done", **result)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
            LAST_CALL[provider] = time.monotonic()
            update(run_id, provider, stage, status="error", error=safe_error(str(exc)))


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/config")
def config():
    keys = credentials()
    return {
        "models": {
            name: {"label": v["label"], "model": v["model"], "ready": bool(keys[name])}
            for name, v in MODELS.items()
        },
        "system_prompt": SYSTEM_PROMPT,
        "protocol_version": PROTOCOL_VERSION,
    }


@app.post("/api/runs")
async def create_run(
    image: UploadFile = File(...),
    system_prompt: str = Form(SYSTEM_PROMPT),
    max_edge: int = Form(1280),
    max_output_tokens: int = Form(1000),
):
    if max_edge not in (768, 1024, 1280, 1600) or max_output_tokens not in (
        512,
        768,
        1000,
    ):
        raise HTTPException(
            422, "Cấu hình độ phân giải hoặc giới hạn token không hợp lệ."
        )
    if not system_prompt.strip() or len(system_prompt) > 12000:
        raise HTTPException(
            422,
            "Cần system prompt, tối đa 12.000 ký tự.",
        )
    data = await image.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ảnh tối đa 20 MB.")
    try:
        png, metadata = prepare_image(data, max_edge)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise HTTPException(
                409, "Một lượt thử đang chạy. Chờ hoàn tất hoặc dừng lượt đó trước."
            )
        run_id = str(uuid.uuid4())
        directory = run_path(run_id)
        directory.mkdir(parents=True)
        (directory / "image.png").write_bytes(png)
        job = {
            "id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "filename": Path(image.filename or "image").name,
            "image": metadata,
            "system_prompt": system_prompt,
            "protocol_version": PROTOCOL_VERSION,
            "stages": ["analysis"],
            "max_output_tokens": max_output_tokens,
            "max_edge": max_edge,
            "protocol": "Same normalized PNG + exact shared system instruction; one image request per candidate; no extra user text, tool schema, tool bounds or planning call; temperature=0.4; thinking off/minimal.",
            "results": {
                name: {
                    "label": v["label"],
                    "model": v["model"],
                    "analysis": {"status": "pending"},
                }
                for name, v in MODELS.items()
            },
        }
        JOBS[run_id] = job
        CANCEL[run_id] = threading.Event()
        persist(job)
        for provider in MODELS:
            POOL.submit(worker, run_id, provider)
    return {"id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    directory = run_path(run_id)
    with LOCK:
        if run_id in JOBS:
            return copy.deepcopy(JOBS[run_id])
    try:
        value = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "Không tìm thấy lượt thử.") from exc
    if value["status"] == "running":
        value["status"] = "interrupted"
        for result in value["results"].values():
            for stage in stage_names(value):
                if result[stage]["status"] in {"pending", "waiting", "running"}:
                    result[stage].update(
                        status="error",
                        error="Server đã khởi động lại. Chạy lượt mới để thử lại.",
                    )
    with LOCK:
        if run_id not in JOBS:
            JOBS[run_id] = value
            CANCEL[run_id] = threading.Event()
            persist(value)
        return copy.deepcopy(JOBS[run_id])


@app.get("/api/runs/{run_id}/image")
def run_image(run_id: str):
    path = run_path(run_id) / "image.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/runs/{run_id}/download")
def download(run_id: str):
    path = run_path(run_id) / "result.json"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(
        path,
        filename=f"image-comparison-{run_id[:8]}.json",
        media_type="application/json",
    )


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str):
    with LOCK:
        if run_id not in CANCEL:
            raise HTTPException(404)
        CANCEL[run_id].set()
    return {"ok": True}


@app.post("/api/runs/{run_id}/retry/{provider}/{stage}")
def retry(run_id: str, provider: str, stage: str):
    with LOCK:
        if (
            provider not in MODELS
            or stage not in {"analysis", "plan"}
            or run_id not in JOBS
            or provider not in JOBS[run_id]["results"]
        ):
            raise HTTPException(404)
        job = JOBS[run_id]
        if job.get("protocol_version") != PROTOCOL_VERSION:
            raise HTTPException(
                409, "Lượt cũ dùng prompt ràng buộc. Chạy lượt mới để so sánh tự do."
            )
        if stage != "analysis":
            raise HTTPException(
                404, "Luồng hiện tại chỉ có một request ảnh cho mỗi model."
            )
        if job["status"] == "running" or any(
            j["status"] == "running" for j in JOBS.values()
        ):
            raise HTTPException(409, "Chờ lượt đang chạy hoàn tất trước khi thử lại.")
        if job["results"][provider][stage]["status"] not in {
            "error",
            "cancelled",
            "skipped",
        }:
            raise HTTPException(409, "Chỉ thử lại bước bị lỗi hoặc đã dừng.")
        CANCEL[run_id].clear()
        update(run_id, provider, stage, status="pending", error=None)
        POOL.submit(worker, run_id, provider)
    return {"id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    print(f"Model Lab: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
