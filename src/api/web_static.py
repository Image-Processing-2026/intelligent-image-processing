"""
Phục vụ bản build tĩnh của web app (web/dist) từ FastAPI cho chế độ production-local.
Path không thuộc /api trả file tĩnh nếu có, ngược lại trả index.html (router phía client).
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

logger = logging.getLogger(__name__)


def mount_web(app: FastAPI, dist_dir: Path) -> bool:
    """
    Đăng ký route bắt mọi path GET cho bản build trong dist_dir. Phải gọi SAU khi đã đăng ký
    các router API để route API được khớp trước. Trả về False nếu chưa có bản build.
    """
    root = dist_dir.resolve()
    index = root / "index.html"
    if not index.is_file():
        logger.info("Web build not found at %s; static web app is not served.", dist_dir)
        return False

    @app.get("/{path:path}", include_in_schema=False)
    def serve_web(path: str) -> Response:
        """File tĩnh của bản build, hoặc index.html cho các route của SPA."""
        if path == "api" or path.startswith("api/"):
            # Giữ đúng body 404 mặc định của FastAPI cho route API không tồn tại
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            # Tên file trong assets/ có hash nội dung nên cache lâu được
            cache = (
                "public, max-age=31536000, immutable"
                if candidate.parent.name == "assets"
                else "no-cache"
            )
            return FileResponse(candidate, headers={"Cache-Control": cache})
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    logger.info("Serving web app from %s", root)
    return True
