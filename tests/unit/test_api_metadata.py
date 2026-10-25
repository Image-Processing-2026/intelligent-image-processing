"""
Unit test (offline) cho plumbing image_metadata: state → graph → API /diagnose và /process.
Không gọi Gemini (conftest xóa GEMINI_API_KEY), pipeline chạy luật dự phòng trên ảnh nhỏ.
"""

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from src.agent import graph as graph_module
from src.agent.graph import initial_doctor_state, run_pipeline
from src.analyzer_evaluator.metadata import ImageMetadata
from src.api import routes
from src.api.main import app

client = TestClient(app)


def _exif_jpeg() -> bytes:
    """JPEG 64x64 có EXIF giả: ISO 3200, phần mềm chỉnh sửa."""
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0110] = "Canon EOS R6"
    exif[0x0131] = "Lightroom 13.2"
    sub = exif.get_ifd(0x8769)
    sub[0x8827] = 3200
    sub[0x829D] = IFDRational(20, 10)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90, exif=exif.tobytes())
    return buf.getvalue()


def _image() -> np.ndarray:
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def test_initial_state_carries_image_metadata() -> None:
    meta = ImageMetadata(iso=3200, software="Lightroom 13.2")

    state = initial_doctor_state(_image(), image_metadata=meta)

    assert state["image_metadata"] == meta


def test_initial_state_defaults_metadata_to_none() -> None:
    state = initial_doctor_state(_image())

    assert state["image_metadata"] is None


def test_run_pipeline_stores_metadata_in_final_state() -> None:
    meta = ImageMetadata(iso=1600, camera="Test Cam")

    state = run_pipeline(_image(), max_iterations=1, image_metadata=meta)

    assert state["image_metadata"] == meta


def test_diagnose_returns_image_metadata() -> None:
    response = client.post(
        "/api/v1/diagnose",
        files={"file": ("photo.jpg", _exif_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    meta = response.json()["image_metadata"]
    assert meta["iso"] == 3200
    assert meta["software"] == "Lightroom 13.2"
    assert meta["camera"] == "Canon EOS R6"


def test_process_passes_metadata_into_pipeline_state(monkeypatch) -> None:
    captured: dict = {}
    real_run_pipeline = graph_module.run_pipeline

    def spy(**kwargs):
        captured.update(kwargs)
        return real_run_pipeline(**kwargs)

    monkeypatch.setattr(routes, "run_pipeline", spy)

    response = client.post(
        "/api/v1/process",
        files={"file": ("photo.jpg", _exif_jpeg(), "image/jpeg")},
        data={"max_iterations": "1"},
    )

    assert response.status_code == 200
    assert captured["image_metadata"].iso == 3200
    assert response.json()["image_metadata"]["software"] == "Lightroom 13.2"


def test_process_without_exif_still_succeeds() -> None:
    buf = BytesIO()
    Image.fromarray(_image()).save(buf, format="PNG")

    response = client.post(
        "/api/v1/process",
        files={"file": ("plain.png", buf.getvalue(), "image/png")},
        data={"max_iterations": "1"},
    )

    assert response.status_code == 200
    meta = response.json()["image_metadata"]
    assert meta["iso"] is None
    assert meta["software"] is None
