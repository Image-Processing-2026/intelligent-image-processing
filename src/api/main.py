"""
FastAPI Server Entrypoint.
Khởi chạy dịch vụ backend cho dự án Intelligent Image Processing.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.memory import DEFAULT_MEMORY_PATH, MEMORY_ENV, configure_case_memory
from src.agent.perception import preload_region_models

from .auth import router as auth_router
from .config import settings_from_env
from .db import get_database
from .routes import router
from .web_static import mount_web
from .workspace import router as workspace_router

# utf-8-sig: .env lưu bằng Notepad/PowerShell 5.1 có BOM, làm tên khóa đầu tiên thành
# '﻿GEMINI_API_KEY' và server âm thầm chạy rule-based
load_dotenv(encoding="utf-8-sig")

# Bộ nhớ ca bệnh (Phase 5): CASE_MEMORY_PATH rỗng → tắt; mặc định data/memory/cases.sqlite.
# CSDL chỉ được mở ở lần ghi/đọc đầu tiên.
configure_case_memory(os.getenv(MEMORY_ENV, str(DEFAULT_MEMORY_PATH)))

settings = settings_from_env()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Mở CSDL web app khi khởi động (tạo bảng, đánh dấu phiên bị ngắt giữa chừng) và nạp sẵn
    model phân vùng ở luồng nền khi đã cài (không chặn khởi động).
    """
    get_database()
    preload_region_models()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Intelligent Image Processing API",
    description="Agentic, closed-loop image processing system with classical CV algorithms and VLM orchestration",
    version="0.1.0",
)

# CORS chỉ bật cho các origin khai báo trong CORS_ORIGINS. Web app dùng cùng origin (proxy
# của Vite hoặc bản build do FastAPI phục vụ) nên không cần; không dùng "*" kèm cookie.
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(router)
app.include_router(auth_router)
app.include_router(workspace_router)

# Bản build web (web/dist) đăng ký sau cùng để không che route API
mount_web(app, settings.web_dist_dir)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)
