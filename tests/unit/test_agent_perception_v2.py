"""
Unit tests cho Perceive v2 (Analysis v2 P4b + P5): bbox subject → resolver chung, RegionStats,
cache mask giữa các vòng và executor, prompt bằng chứng (nhãn crop, prior EXIF),
self-consistency N = 3, skin locus, ngân sách ảnh inline, nạp sẵn model, phiên tương tác có
metadata. Chạy offline: Gemini được mock theo schema (như _route_gemini trong test_vlm_mock).
"""

import json
import threading
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from src.agent import perception, session
from src.agent import vlm_diagnostician as vlm
from src.agent.executor import execute_plan, image_cache_key, subject_cache_key
from src.agent.graph import run_pipeline
from src.agent.perception import (
    PERCEIVE_PROMPT,
    PERCEIVE_RESPONSE_SCHEMA,
    _parse_subjects,
    merge_samples,
    perceive,
    preload_region_models,
    skin_locus_stats,
)
from src.agent.state import Defect, DiagnosisReport, HistoryItem, RegionOperation, TreatmentPlan
from src.analyzer_evaluator.metadata import ImageMetadata
from src.api import workspace
from src.api.db import Database
from src.api.main import app
from src.region_engine import SubjectMask

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}
PERSON_BOX = [100, 500, 900, 900]  # (ymin, xmin, ymax, xmax) 0..1000


def _diagnosis(
    defects: Optional[List[Dict[str, Any]]] = None,
    preserve: Optional[List[Dict[str, Any]]] = None,
    scene: str = "portrait",
    subjects: Optional[List[Any]] = None,
    summary: str = "tóm tắt",
) -> str:
    return json.dumps(
        {
            "scene_type": scene,
            "lighting": "nắng",
            "subjects": subjects if subjects is not None else [],
            "defects": defects or [],
            "preserve": preserve or [],
            "summary": summary,
        }
    )


def _gemini(
    mock_genai: MagicMock, perceive_texts: List[str], plan_texts: Optional[List[str]] = None
):
    """Mock client.models theo schema (Perceive/Plan), an toàn khi gọi song song từ thread."""
    queues = {"perceive": list(perceive_texts), "plan": list(plan_texts or [])}
    lock = threading.Lock()
    calls: List[Dict[str, Any]] = []

    def generate_content(*, model: str, contents: list, config: Any) -> MagicMock:
        stage = "perceive" if config.response_json_schema == PERCEIVE_RESPONSE_SCHEMA else "plan"
        with lock:
            calls.append({"stage": stage, "contents": contents, "config": config})
            return MagicMock(text=queues[stage].pop(0))

    models = MagicMock()
    models.generate_content.side_effect = generate_content
    mock_genai.Client.return_value.models = models
    return calls


def _object_image(height: int = 200, width: int = 300) -> np.ndarray:
    """Nền xanh lá có texture nhẹ, vật thể đỏ đậm ở phải-giữa (cho GrabCut khi không có model)."""
    rng = np.random.default_rng(5)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 150 + rng.integers(0, 20, (height, width))
    image[..., 0] = 40
    image[30:170, 170:260] = (200, 30, 30)
    return image


def _image_size(part: Any) -> tuple:
    """(width, height) của một ảnh đã mã hóa thành Part gửi Gemini."""
    return Image.open(BytesIO(part.inline_data.data)).size


# ============================================================
# Self-consistency
# ============================================================
class TestMergeSamples:
    def test_three_different_samples_keep_majority(self) -> None:
        samples = [
            perception._report_from_vlm_json(json.loads(text), 1)
            for text in (
                _diagnosis(
                    defects=[
                        {"type": "noise", "region": "full", "severity": 2, "crops": ["A"]},
                        {"type": "blur", "region": "full", "severity": 1},
                    ],
                    preserve=[{"aspect": "warm_tone"}],
                    subjects=[{"name": "person", "box_2d": [100, 100, 900, 500]}],
                    summary="một",
                ),
                _diagnosis(
                    defects=[{"type": "noise", "region": "full", "severity": 3, "crops": ["B"]}],
                    preserve=[{"aspect": "warm_tone"}],
                    subjects=[{"name": "person", "box_2d": [120, 80, 880, 520]}],
                    summary="hai",
                ),
                _diagnosis(
                    defects=[{"type": "underexposed", "region": "person", "severity": 2}],
                    scene="landscape",
                    subjects=[{"name": "person", "box_2d": [110, 90, 890, 510]}],
                    summary="ba",
                ),
            )
        ]
        merged = merge_samples(samples, iteration=1)

        assert [(d.type, d.region, d.severity) for d in merged.defects] == [("noise", "full", 3)]
        noise = merged.defects[0]
        assert noise.confidence == pytest.approx(2 / 3, abs=1e-3)
        consistency = [i for i in noise.evidence_items if i.metric == "self_consistency"][0]
        assert consistency.text == "2/3 lần quan sát"
        assert [i.text for i in noise.evidence_items if i.metric == "crops"] == ["A, B"]
        assert sorted((d.type, d.region) for d in merged.rejected) == [
            ("blur", "full"),
            ("underexposed", "person"),
        ]
        assert all(d.confidence == pytest.approx(1 / 3, abs=1e-3) for d in merged.rejected)
        assert merged.scene_type == "portrait"
        assert [p.aspect for p in merged.preserve] == ["warm_tone"]
        assert merged.subjects == ["person"]
        assert merged.subject_boxes == {"person": [110, 90, 890, 510]}
        assert merged.summary in ("một", "hai")

    def test_single_sample_is_unchanged(self) -> None:
        sample = perception._report_from_vlm_json(
            json.loads(_diagnosis(defects=[{"type": "noise", "region": "full", "severity": 2}])), 1
        )
        merged = merge_samples([sample])
        assert [(d.type, d.severity, d.confidence) for d in merged.defects] == [("noise", 2, 1.0)]
        assert merged.rejected == []

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_perceive_runs_three_samples_at_temperature(
        self, mock_genai: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(perception, "PERCEIVE_SAMPLES", 3)
        noise = {"type": "noise", "region": "full", "severity": 2}
        calls = _gemini(
            mock_genai,
            [
                _diagnosis(defects=[noise]),
                _diagnosis(defects=[noise, {"type": "blur", "region": "full", "severity": 1}]),
                _diagnosis(defects=[]),
            ],
        )
        report = perceive(np.full((64, 64, 3), 128, dtype=np.uint8), {})

        assert len(calls) == 3
        assert all(call["config"].temperature == pytest.approx(0.7) for call in calls)
        assert [(d.type, d.confidence) for d in report.defects] == [("noise", pytest.approx(0.667))]
        assert [d.type for d in report.rejected] == ["blur"]
        assert report.source == "vlm"

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_failed_sample_is_skipped(
        self, mock_genai: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(perception, "PERCEIVE_SAMPLES", 3)
        noise = {"type": "noise", "region": "full", "severity": 2}
        _gemini(mock_genai, [_diagnosis(defects=[noise]), "không phải JSON", _diagnosis()])
        report = perceive(np.full((64, 64, 3), 128, dtype=np.uint8), {})
        # 2 lần hợp lệ → cần 2/2: noise chỉ có 1/2 → bị loại
        assert report.source == "vlm"
        assert report.defects == []
        assert [d.type for d in report.rejected] == ["noise"]

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_all_samples_failing_falls_back_to_rules(
        self, mock_genai: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(perception, "PERCEIVE_SAMPLES", 3)
        _gemini(mock_genai, ["x", "y", "z"])
        report = perceive(np.full((64, 64, 3), 128, dtype=np.uint8), {"noise_level": "severe"})
        assert report.source == "vlm_fallback"
        assert [d.type for d in report.defects] == ["noise"]

    def test_single_sample_uses_default_temperature(self) -> None:
        with (
            patch.dict("os.environ", FAKE_KEY),
            patch("src.agent.vlm_diagnostician.genai") as mock_genai,
        ):
            calls = _gemini(mock_genai, [_diagnosis()])
            perceive(np.full((64, 64, 3), 128, dtype=np.uint8), {})
        assert calls[0]["config"].temperature is None

    def test_env_overrides_sample_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mặc định D3 = 1 lần (người dùng chốt 2026-10-25); env PERCEIVE_SAMPLES ghi đè
        monkeypatch.setenv(perception.PERCEIVE_SAMPLES_ENV, "3")
        assert perception._perceive_samples() == 3
        monkeypatch.setenv(perception.PERCEIVE_SAMPLES_ENV, "0")
        assert perception._perceive_samples() == 1
        monkeypatch.setenv(perception.PERCEIVE_SAMPLES_ENV, "abc")
        assert perception._perceive_samples() == perception.PERCEIVE_SAMPLES


# ============================================================
# Schema và prompt
# ============================================================
class TestSchemaAndPrompt:
    def test_subjects_schema_has_box_2d(self) -> None:
        items = PERCEIVE_RESPONSE_SCHEMA["properties"]["subjects"]["items"]
        assert items["required"] == ["name"]
        assert items["properties"]["box_2d"]["maxItems"] == 4
        assert "box_2d" in PERCEIVE_PROMPT and "không có lỗi" in PERCEIVE_PROMPT

    def test_parse_subjects_accepts_legacy_names_and_drops_bad_boxes(self) -> None:
        names, boxes = _parse_subjects(
            [
                "Face",
                {"name": "sky", "box_2d": [0, 0, 400, 1000]},
                {"name": "dog", "box_2d": [500, 500, 400, 600]},  # ymin ≥ ymax
                {"name": "car", "box_2d": [0, 0, 10]},
                {"name": "full"},
                {"box_2d": [0, 0, 1, 1]},
            ]
        )
        assert names == ["face", "sky", "dog", "car"]
        assert boxes == {"sky": [0, 0, 400, 1000]}

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_prompt_lists_every_crop_label_and_exif_prior(self, mock_genai: MagicMock) -> None:
        rng = np.random.default_rng(1)
        image = np.clip(rng.normal(120, 6, (900, 1400, 3)), 0, 255).astype(np.uint8)
        image[100:300, 1000:1300] = 255
        calls = _gemini(mock_genai, [_diagnosis()])
        perceive(image, {}, metadata=ImageMetadata(iso=6400))

        contents = calls[0]["contents"]
        text = contents[0]
        crop_labels = [
            p.split(":")[0] for p in contents if isinstance(p, str) and p.startswith("Crop ")
        ]
        assert crop_labels, "large image must carry evidence crops"
        assert len(crop_labels) <= 6
        for label in crop_labels:
            assert f"- {label} [" in text
        assert "SỐ ĐO CLASSICAL TOÀN ẢNH" in text
        assert "GỢI Ý TỪ EXIF" in text and "noise" in text
        assert "Chỉ số kỹ thuật toàn cục" not in text  # không dump JSON khi đã có số đo
        images = [p for p in contents if not isinstance(p, str)]
        assert len(images) <= 1 + 6 + 2  # ngân sách: toàn khung + ≤ 6 crop + 2 overlay
        assert contents[1] == "ẢNH TOÀN KHUNG:"
        assert _image_size(images[0]) == (1400, 900)  # toàn khung ở độ phân giải gốc

    @patch.dict("os.environ", {**FAKE_KEY, "PERCEIVE_FULL_FRAME_MAX_SIDE": "512"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_full_frame_can_be_downscaled(self, mock_genai: MagicMock) -> None:
        calls = _gemini(mock_genai, [_diagnosis()])
        perceive(np.full((900, 1400, 3), 120, dtype=np.uint8), {})
        frame = [p for p in calls[0]["contents"] if not isinstance(p, str)][0]
        assert max(_image_size(frame)) == 512

    def test_plan_prompt_never_carries_subject_boxes(self) -> None:
        diagnosis = DiagnosisReport(
            subjects=["person"],
            subject_boxes={"person": PERSON_BOX},
            defects=[Defect(type="noise", region="person")],
        )
        contents = vlm._build_contents(np.zeros((8, 8, 3), np.uint8), {}, 1, None, None, diagnosis)
        assert "subject_boxes" not in contents[0] and "box_2d" not in contents[0]
        assert "region_stats" not in contents[0]


# ============================================================
# Đo theo vùng và cache mask
# ============================================================
class TestRegionsAndCache:
    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_resolver_receives_perceive_bbox(self, mock_genai: MagicMock) -> None:
        image = _object_image()
        _gemini(
            mock_genai,
            [
                _diagnosis(
                    subjects=[{"name": "Person", "box_2d": PERSON_BOX}],
                    defects=[{"type": "underexposed", "region": "person", "severity": 2}],
                )
            ],
        )
        seen: List[Any] = []

        def resolve(img: np.ndarray, name: str, box_2d=None, *, use_semantic=None, cache_key=None):
            seen.append((name, box_2d, cache_key))
            mask = np.zeros(img.shape[:2], np.float32)
            if name == "person":
                mask[30:170, 170:260] = 1.0
                return SubjectMask(mask, "gemini_grabcut", 0.4, None)
            return SubjectMask(np.zeros(img.shape[:2], np.float32), "face_mediapipe", 0.0, None)

        with patch("src.agent.perception.resolve_subject_mask", side_effect=resolve):
            report = perceive(image, {})

        expected_key = subject_cache_key(image_cache_key(image), image.shape)
        assert ("person", tuple(PERSON_BOX), expected_key) in seen
        assert report.subject_boxes == {"person": PERSON_BOX}
        assert set(report.region_stats) == {"person", "full", "background"}
        person = report.region_stats["person"]
        assert person.mask_source == "gemini_grabcut" and person.mask_confidence == 0.4
        assert report.region_stats["background"].mask_source == "complement"
        assert report.measurement is not None and report.measurement.width == 300
        assert "person" in report.region_metrics  # RegionMetrics cũ vẫn có

    def test_rule_based_pipeline_fills_measurement_and_hides_maps(self) -> None:
        state = run_pipeline(np.full((64, 64, 3), 60, dtype=np.uint8), max_iterations=1)
        diagnosis = state["history"][0].diagnosis
        assert diagnosis.source == "rule_based"
        assert diagnosis.measurement is not None
        assert {"full", "background"} <= set(diagnosis.region_stats)
        assert isinstance(diagnosis.measurement_maps, dict)
        dumped = state["history"][0].model_dump_json()
        assert "measurement_maps" not in dumped and "noise.sigma_luma" in dumped

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_masks_are_cached_across_iterations_and_executor(self, mock_genai: MagicMock) -> None:
        from src.region_engine import subject as subject_module

        original = _object_image()
        response = _diagnosis(subjects=[{"name": "person", "box_2d": [150, 567, 850, 867]}])
        _gemini(mock_genai, [response, response])

        with (
            patch.object(subject_module, "_resolve", wraps=subject_module._resolve) as spy,
            patch.object(
                subject_module,
                "capabilities",
                return_value={"face": False, "semantic_models": False},
            ),
        ):
            first = perceive(original, {}, iteration=1, original_image=original)
            calls_after_first = spy.call_count
            current = np.clip(original.astype(int) + 20, 0, 255).astype(np.uint8)
            history = [
                HistoryItem(
                    iteration=1,
                    plan=TreatmentPlan(reasoning="r"),
                    metrics_before={},
                    metrics_after={},
                    eval_score={},
                    decision="RE_PROCESS",
                    diagnosis=first,
                )
            ]
            second = perceive(current, {}, iteration=2, history=history, original_image=original)
            assert spy.call_count == calls_after_first  # vòng 2 trúng cache

            plan = TreatmentPlan(
                reasoning="r",
                actions=[
                    RegionOperation(
                        region_id="person",
                        target_prompt="person",
                        detected_issue="underexposed",
                        operation="gamma_correct",
                        parameters={"gamma": 1.5},
                    )
                ],
            )
            output = execute_plan(
                current, plan, second.subject_boxes, cache_key=image_cache_key(original)
            )
            assert spy.call_count == calls_after_first  # executor dùng lại mask

        assert first.region_stats["person"].mask_source == "gemini_grabcut"
        assert second.region_stats["person"].area_fraction == pytest.approx(
            first.region_stats["person"].area_fraction
        )  # cùng mask (ảnh khác nên số đo khác)
        changed = np.abs(output.astype(int) - current.astype(int)).sum(axis=2) > 0
        assert changed[100, 215] and not changed[100, 20]  # chỉ vật thể được xử lý


class TestExecutorSubjectResolver:
    def _plan(self, target: str = "person") -> TreatmentPlan:
        return TreatmentPlan(
            reasoning="r",
            actions=[
                RegionOperation(
                    region_id=target,
                    target_prompt=target,
                    detected_issue="underexposed",
                    operation="gamma_correct",
                    parameters={"gamma": 1.6},
                )
            ],
        )

    def test_semantic_action_with_box_uses_shared_resolver(self) -> None:
        image = np.full((40, 60, 3), 80, dtype=np.uint8)
        mask = np.zeros((40, 60), np.float32)
        mask[:, 30:] = 1.0
        with patch(
            "src.agent.executor.resolve_subject_mask",
            return_value=SubjectMask(mask, "gemini_sam", 0.6, None),
        ) as resolver:
            output = execute_plan(image, self._plan(), {"person": PERSON_BOX}, cache_key="base")

        args, kwargs = resolver.call_args
        assert args[1:] == ("person", tuple(PERSON_BOX))
        assert kwargs["cache_key"] == "base:40x60"
        assert output[0, 50, 0] > 80 and output[0, 5, 0] == 80

    def test_unlocated_subject_skips_action(self) -> None:
        image = np.full((40, 60, 3), 80, dtype=np.uint8)
        full = SubjectMask(np.ones((40, 60), np.float32), "full", 0.1, None)
        with patch("src.agent.executor.resolve_subject_mask", return_value=full):
            output = execute_plan(image, self._plan(), {"person": PERSON_BOX})
        assert np.array_equal(output, image)

    def test_actions_without_box_keep_old_path(self) -> None:
        image = np.full((40, 60, 3), 80, dtype=np.uint8)
        with patch("src.agent.executor.resolve_subject_mask") as resolver:
            execute_plan(image, self._plan("full"), {"person": PERSON_BOX})
            execute_plan(image, self._plan("sky"), {"person": PERSON_BOX})
        resolver.assert_not_called()


# ============================================================
# Skin locus, ngân sách ảnh, nạp model
# ============================================================
class TestSkinLocus:
    def _face(self, color: tuple) -> tuple:
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        image[10:40, 10:40] = color
        mask = np.zeros((50, 50), np.float32)
        mask[10:40, 10:40] = 1.0
        return image, mask

    def test_skin_colour_is_inside_locus(self) -> None:
        stats = skin_locus_stats(*self._face((224, 172, 140)), confidence=0.9)
        assert stats["color.skin_locus_fraction"].value == pytest.approx(1.0)
        assert 0.0 < stats["color.skin_locus_fraction"].confidence <= 0.9

    def test_blue_cast_face_is_outside_locus(self) -> None:
        stats = skin_locus_stats(*self._face((120, 150, 220)), confidence=0.9)
        assert stats["color.skin_locus_fraction"].value == pytest.approx(0.0)

    def test_grayscale_has_no_skin_check(self) -> None:
        image, mask = self._face((128, 128, 128))
        assert skin_locus_stats(image, mask, 0.9) == {}
        assert skin_locus_stats(image[..., 0], mask, 0.9) == {}


def test_small_images_are_encoded_before_the_full_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    budgets: Dict[tuple, int] = {}

    def fake_encode(image: Image.Image, budget: int):
        budgets[image.size] = budget
        return b"x" * (image.width * image.height // 100), "image/png"

    monkeypatch.setattr(vlm, "_encode_image", fake_encode)
    monkeypatch.setattr(vlm, "VLM_INLINE_IMAGE_BUDGET", 1_000_000)
    big = Image.new("RGB", (4000, 3000))
    crop = Image.new("RGB", (512, 512))
    parts = vlm._encode_contents(["text", big, crop])
    assert parts[0] == "text" and len(parts) == 3
    assert budgets[(512, 512)] == 500_000
    assert budgets[(4000, 3000)] == 1_000_000 - 512 * 512 // 100


def test_preload_skips_without_models() -> None:
    with (
        patch(
            "src.agent.perception.subject_capabilities",
            return_value={"face": True, "semantic_models": False},
        ),
        patch("src.agent.perception.resolve_subject_mask") as resolver,
    ):
        assert preload_region_models() is None
    resolver.assert_not_called()


def test_preload_warms_semantic_backend() -> None:
    with (
        patch(
            "src.agent.perception.subject_capabilities",
            return_value={"face": True, "semantic_models": True},
        ),
        patch("src.agent.perception.resolve_subject_mask") as resolver,
    ):
        thread = preload_region_models()
        assert thread is not None
        thread.join(timeout=5)
    assert resolver.call_args.kwargs["use_semantic"] is True


# ============================================================
# Phiên tương tác nhận metadata
# ============================================================
def test_start_session_puts_metadata_in_state() -> None:
    meta = ImageMetadata(iso=3200, software="Lightroom")
    result = session.start_session(np.full((32, 32, 3), 60, dtype=np.uint8), image_metadata=meta)
    try:
        values = session._APP.get_state(session._config(result.session_id)).values
        assert values["image_metadata"] == meta
    finally:
        session.end_session(result.session_id)


def _exif_jpeg() -> bytes:
    img = Image.fromarray(np.full((40, 40, 3), 90, dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0131] = "Lightroom 13.2"
    sub = exif.get_ifd(0x8769)
    sub[0x8827] = 3200
    sub[0x829D] = IFDRational(20, 10)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90, exif=exif.tobytes())
    return buffer.getvalue()


@pytest.fixture
def session_client(app_db: Database) -> Iterator[TestClient]:
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/register", json={"username": "meta", "password": "correct horse"}
    )
    assert response.status_code == 201
    yield client


def test_create_session_route_reads_upload_metadata(
    session_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: List[Dict[str, Any]] = []

    def fake_start(image: np.ndarray, **kwargs: Any) -> session.SessionResult:
        started.append(kwargs)
        return session.SessionResult(
            session_id="thread-1",
            status="needs_input",
            questions=[],
            diagnosis=DiagnosisReport(source="rule_based"),
        )

    monkeypatch.setattr(workspace, "start_session", fake_start)
    response = session_client.post(
        "/api/v1/workspace/sessions",
        files={"file": ("photo.jpg", _exif_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 201
    meta = started[0]["image_metadata"]
    assert isinstance(meta, ImageMetadata)
    assert meta.iso == 3200 and meta.software == "Lightroom 13.2"
