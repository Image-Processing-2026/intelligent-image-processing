"""
Integration tests for FastAPI endpoints: /health, /diagnose, /process.
"""

import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

client = TestClient(app)


def _create_png_bytes(width: int = 64, height: int = 64, color: tuple = (128, 128, 128)) -> bytes:
    """Tạo file ảnh PNG giả lập trong bộ nhớ."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_noisy_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Tạo ảnh PNG có nhiễu để kích hoạt pipeline xử lý."""
    arr = np.random.randint(50, 200, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_check_endpoint():
    """Kiểm tra endpoint /api/v1/health hoạt động bình thường."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_diagnose_endpoint():
    """Kiểm tra endpoint /api/v1/diagnose trả về metrics và treatment_plan."""
    img_bytes = _create_png_bytes()
    response = client.post(
        "/api/v1/diagnose",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "technical_metrics" in data
    assert "treatment_plan" in data
    assert "brightness_mean" in data["technical_metrics"]
    assert "actions" in data["treatment_plan"]


def test_process_endpoint_real_image():
    """Kiểm tra endpoint /api/v1/process với ảnh thực (không ground truth)."""
    img_bytes = _create_noisy_png_bytes()
    response = client.post(
        "/api/v1/process",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"max_iterations": "2"},
    )
    assert response.status_code == 200
    data = response.json()

    # Kiểm tra các trường cơ bản
    assert "processed_image_base64" in data
    assert len(data["processed_image_base64"]) > 0
    # Base64 có thể decode được thành ảnh
    img_data = base64.b64decode(data["processed_image_base64"])
    decoded_img = Image.open(io.BytesIO(img_data))
    assert decoded_img.size == (64, 64)

    assert data["total_iterations"] >= 1
    assert data["final_decision"] in ["SHIP", "STOP_BEST_EFFORT"]
    assert isinstance(data["history"], list)
    assert len(data["history"]) >= 1

    # Kiểm tra trường intermediate_images_base64 (M4-07)
    assert "intermediate_images_base64" in data
    assert isinstance(data["intermediate_images_base64"], list)
    assert len(data["intermediate_images_base64"]) >= 1

    # Kiểm tra mỗi ảnh trung gian là base64 hợp lệ
    for item_b64 in data["intermediate_images_base64"]:
        assert len(item_b64) > 0
        thumb_bytes = base64.b64decode(item_b64)
        thumb_img = Image.open(io.BytesIO(thumb_bytes))
        assert max(thumb_img.size) <= 512


def test_process_endpoint_synthetic_image():
    """Kiểm tra endpoint /api/v1/process với ảnh nhân tạo (có ground truth)."""
    clean_bytes = _create_png_bytes(64, 64, (128, 128, 128))
    noisy_bytes = _create_noisy_png_bytes(64, 64)

    response = client.post(
        "/api/v1/process",
        files={
            "file": ("noisy.png", noisy_bytes, "image/png"),
            "ground_truth": ("clean.png", clean_bytes, "image/png"),
        },
        data={"max_iterations": "2"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["total_iterations"] >= 1
    assert data["final_decision"] in ["SHIP", "STOP_BEST_EFFORT"]
    assert "psnr" in data["final_evaluation"]
    assert "ssim" in data["final_evaluation"]

    # Ảnh trung gian phải có
    assert "intermediate_images_base64" in data
    assert len(data["intermediate_images_base64"]) >= 1


def test_process_endpoint_invalid_file():
    """Kiểm tra endpoint /api/v1/process với file ảnh hỏng → trả về 500."""
    response = client.post(
        "/api/v1/process",
        files={"file": ("corrupt.png", b"not_an_image", "image/png")},
    )
    assert response.status_code == 500
