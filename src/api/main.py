"""
FastAPI Server Entrypoint.
Khởi chạy dịch vụ backend cho dự án Intelligent Image Processing.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

load_dotenv()

app = FastAPI(
    title="Intelligent Image Processing API",
    description="Agentic, closed-loop image processing system with classical CV algorithms and VLM orchestration",
    version="0.1.0",
)

# Cấu hình CORS để frontend Gradio/React có thể kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)
