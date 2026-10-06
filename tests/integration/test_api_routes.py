"""
Integration tests for FastAPI endpoints: /health, /diagnose, /process.
"""

import base64
import io
import json

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
    # Phase 1: chẩn đoán có cấu trúc đi kèm kế hoạch
    assert "defects" in data["diagnosis"]
    assert "preserve" in data["diagnosis"]


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


def _decode(image_base64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB"))


def _create_dark_png_bytes(width: int = 64, height: int = 64) -> bytes:
    gradient = np.tile(np.linspace(10, 60, width), (height, 1)).astype(np.uint8)
    img = Image.fromarray(np.repeat(gradient[:, :, None], 3, axis=2))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_process_endpoint_returns_ranked_variants():
    """Phase 3: num_variants=3 → các phiên bản đã xếp hạng, kèm preview và phác đồ."""
    response = client.post(
        "/api/v1/process",
        files={"file": ("dark.png", _create_dark_png_bytes(), "image/png")},
        data={"max_iterations": "2", "num_variants": "3"},
    )
    assert response.status_code == 200
    data = response.json()
    variants = data["variants"]
    assert sorted(v["id"] for v in variants) == ["balanced", "natural", "vivid"]
    assert [v["rank"] for v in variants] == [1, 2, 3]
    assert data["recommended_variant"] in {v["id"] for v in variants}
    assert data["variant_ranking_source"] == "score"
    assert all(_decode(v["preview_base64"]).shape == (64, 64, 3) for v in variants)


def test_render_reproduces_the_chosen_variant():
    """Render full-res phiên bản 'balanced' phải khớp ảnh kết quả của /process."""
    image_bytes = _create_dark_png_bytes()
    processed = client.post(
        "/api/v1/process",
        files={"file": ("dark.png", image_bytes, "image/png")},
        data={"max_iterations": "2", "num_variants": "2"},
    ).json()
    balanced = next(v for v in processed["variants"] if v["id"] == "balanced")

    response = client.post(
        "/api/v1/render",
        files={"file": ("dark.png", image_bytes, "image/png")},
        data={"actions": json.dumps(balanced["actions"])},
    )
    assert response.status_code == 200
    rendered = response.json()
    assert (rendered["width"], rendered["height"]) == (64, 64)
    assert np.array_equal(
        _decode(rendered["image_base64"]), _decode(processed["processed_image_base64"])
    )


def test_render_keeps_action_order_and_drops_unknown_operations():
    actions = [
        {
            "region_id": "full",
            "target_prompt": "full",
            "region_type": "full",
            "detected_issue": "x",
            "operation": op,
            "parameters": params,
        }
        for op, params in [
            ("gamma_correct", {"gamma": 9.0}),
            ("face_beautify", {}),
            ("denoise", {"method": "bilateral", "strength": 1.0}),
        ]
    ]
    response = client.post(
        "/api/v1/render",
        files={"file": ("dark.png", _create_dark_png_bytes(), "image/png")},
        data={"actions": json.dumps(actions)},
    )
    assert response.status_code == 200
    applied = response.json()["applied_actions"]
    assert [a["operation"] for a in applied] == ["gamma_correct", "denoise"]
    assert applied[0]["parameters"]["gamma"] == 2.5  # đã kẹp theo PARAMETER_BOUNDS


def test_render_rejects_malformed_actions():
    response = client.post(
        "/api/v1/render",
        files={"file": ("dark.png", _create_dark_png_bytes(), "image/png")},
        data={"actions": '{"not": "a list"}'},
    )
    assert response.status_code == 422
