"""Unit contracts for the optional face-oval landmark adapter."""

from __future__ import annotations

import numpy as np
import pytest

import src.region_engine.face_landmarker as landmarker
from src.region_engine.face_landmarker import FaceLandmarkerError, detect_face_ovals

IMAGE = np.zeros((40, 60, 3), dtype=np.uint8)


class FakeLandmarker:
    def __init__(self, faces: object) -> None:
        self.faces = faces
        self.calls = 0
        self.closed = 0

    def detect(self, image: np.ndarray) -> object:
        self.calls += 1
        assert image.flags.c_contiguous
        return self.faces

    def close(self) -> None:
        self.closed += 1


def _face() -> list[dict[str, float]]:
    # The official oval uses indices through 454.  Make a non-degenerate
    # ellipse at every landmark so the adapter only depends on the published
    # oval ring, not arbitrary landmark ordering.
    points: list[dict[str, float]] = []
    for index in range(455):
        angle = 2 * np.pi * index / 455
        points.append({"x": float(0.5 + 0.2 * np.cos(angle)), "y": float(0.5 + 0.3 * np.sin(angle))})
    return points


def test_landmarker_rasterizes_face_oval_and_preserves_pixel_contour(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLandmarker([_face()])
    landmarker.reset_face_landmarker()
    monkeypatch.setattr(landmarker, "_LANDMARKER_FACTORY", lambda _path, _count: fake)

    ovals = detect_face_ovals(IMAGE, num_faces=2)

    assert len(ovals) == 1
    assert ovals[0].mask.dtype == np.bool_
    assert ovals[0].mask.shape == IMAGE.shape[:2]
    assert 0 < int(ovals[0].mask.sum()) < IMAGE.shape[0] * IMAGE.shape[1]
    assert ovals[0].contour.shape == (36, 2)
    assert np.all(ovals[0].contour[:, 0] >= 0)
    assert np.all(ovals[0].contour[:, 0] < IMAGE.shape[1])
    landmarker.reset_face_landmarker()
    assert fake.closed == 1


def test_landmarker_rejects_short_or_invalid_landmark_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLandmarker([[{"x": 0.5, "y": 0.5}]])
    landmarker.reset_face_landmarker()
    monkeypatch.setattr(landmarker, "_LANDMARKER_FACTORY", lambda _path, _count: fake)

    with pytest.raises(FaceLandmarkerError, match="needs index 454"):
        detect_face_ovals(IMAGE)

    landmarker.reset_face_landmarker()
