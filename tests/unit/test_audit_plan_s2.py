"""
Unit tests cho lane S2 của đợt sửa audit (R-05, R-06, R-07):
- R-05: action nhắm khuôn mặt mang defect_ref của lỗi ở chủ thể người; recipe dark-region chọn
  mặt/giảm một bậc; preserve 'person' phủ lên 'face'.
- R-06: ngược sáng ở người "hết" nhưng mặt không sáng lên tương ứng → unverified.
- R-07: black_lift ghi nhận (không rollback), hai chiều, lỗi đo → unavailable.
Chạy offline, không cần API key.
"""

from typing import Any, Dict

import numpy as np
import pytest

from src.agent import graph
from src.agent.fusion import (
    FACE_MIN_GAIN_L,
    backlit_progress,
    backlit_resolution_cause,
)
from src.agent.graph import (
    _decide_tracked,
    _needs_verification,
    _side_effects_check,
    _targeted_defects,
    _verify_backlit,
)
from src.agent.knowledge import format_context, plan_actions_from_knowledge, retrieve
from src.agent.planner import (
    apply_preserve_guard,
    normalize_defect_ref,
    sanitize_actions,
)
from src.agent.state import (
    Defect,
    DiagnosisReport,
    PreserveItem,
    RegionOperation,
    SideEffectCheck,
    TreatmentPlan,
)
from src.agent.vlm_diagnostician import _plan_from_vlm_json
from src.analyzer_evaluator.measurement.diff import (
    BLACK_LIFT_MAX_SEVERITY,
    DefectChange,
    black_lift,
)
from src.analyzer_evaluator.measurement.schemas import Estimate, RegionStats

BACKLIT_PERSON = Defect(type="backlit_subject", region="person", severity=3)


def _stats(region: str, mean_l: float) -> RegionStats:
    return RegionStats(
        region=region,
        area_fraction=0.2,
        values={"exposure.mean_l": Estimate(value=mean_l, confidence=0.9, n_samples=1000)},
    )


def _face_stats() -> Dict[str, RegionStats]:
    """region_stats của Perceive khi đã định vị được mặt."""
    return {"face": _stats("face", 30.0)}


def _action(**fields: Any) -> RegionOperation:
    base: Dict[str, Any] = {
        "region_id": "face",
        "target_prompt": "face",
        "detected_issue": "backlit_subject",
        "operation": "gamma_correct",
        "parameters": {"gamma": 1.5},
    }
    return RegionOperation(**{**base, **fields})


# ============================================================
# R-05: kế hoạch nhắm mặt và defect_ref
# ============================================================
class TestBacklitPlanning:
    def test_person_defect_with_face_targets_face(self):
        diagnosis = DiagnosisReport(defects=[BACKLIT_PERSON], region_stats=_face_stats())
        notes: list = []
        actions, used = plan_actions_from_knowledge(diagnosis, {}, notes=notes)
        assert used == ["dark-region"]
        assert len(actions) == 1
        action = actions[0]
        assert action["target_prompt"] == "face"
        assert action["region_type"] == "face"
        assert action["face_mode"] == "oval"
        assert action["defect_ref"] == "backlit_subject@person"
        assert action["parameters"] == {"gamma": 1.6}
        assert action["feather_radius"] == 30
        assert any("face" in note for note in notes)

    def test_person_defect_without_face_keeps_person_one_tier_lower(self):
        diagnosis = DiagnosisReport(defects=[BACKLIT_PERSON])
        notes: list = []
        actions, _ = plan_actions_from_knowledge(diagnosis, {}, notes=notes)
        assert [(a["target_prompt"], a["parameters"]) for a in actions] == [
            ("person", {"gamma": 1.4})
        ]
        assert actions[0]["defect_ref"] == "backlit_subject@person"
        assert "face_mode" not in actions[0]
        assert any("Không định vị được khuôn mặt" in note and "3 → 2" in note for note in notes)

    def test_severity_two_without_face_drops_to_lowest_tier(self):
        defect = Defect(type="backlit_subject", region="person", severity=2)
        actions, _ = plan_actions_from_knowledge(DiagnosisReport(defects=[defect]), {})
        assert actions[0]["parameters"] == {"gamma": 1.2}

    def test_non_person_region_is_unchanged(self):
        defect = Defect(type="backlit_subject", region="dog", severity=2)
        actions, _ = plan_actions_from_knowledge(
            DiagnosisReport(defects=[defect], region_stats=_face_stats()), {}
        )
        assert [(a["target_prompt"], a["parameters"]) for a in actions] == [("dog", {"gamma": 1.4})]

    def test_optional_light_person_action_is_off_by_default_and_toggleable(self, monkeypatch):
        from src.agent.knowledge import engine

        diagnosis = DiagnosisReport(defects=[BACKLIT_PERSON], region_stats=_face_stats())
        assert len(plan_actions_from_knowledge(diagnosis, {})[0]) == 1
        monkeypatch.setattr(engine, "BACKLIT_LIGHT_PERSON_ENABLED", True)
        actions, _ = plan_actions_from_knowledge(diagnosis, {})
        assert {a["target_prompt"] for a in actions} == {"face", "person"}
        light = next(a for a in actions if a["target_prompt"] == "person")
        assert 1.0 < light["parameters"]["gamma"] < 1.6
        assert light["defect_ref"] == "backlit_subject@person"

    def test_prompt_context_describes_face_target_and_defect_ref(self):
        diagnosis = DiagnosisReport(defects=[BACKLIT_PERSON], region_stats=_face_stats())
        text = format_context(retrieve(diagnosis))
        assert "vùng 'face'" in text
        assert "face_mode=oval" in text
        assert 'defect_ref="backlit_subject@person"' in text

    def test_normalize_defect_ref(self):
        assert normalize_defect_ref(" Backlit_Subject @ Person ") == "backlit_subject@person"
        assert normalize_defect_ref("backlit_subject@khuôn mặt") == "backlit_subject@face"
        assert normalize_defect_ref("not_a_defect@person") is None
        assert normalize_defect_ref("backlit_subject") is None
        assert normalize_defect_ref(7) is None

    def test_sanitize_normalizes_defect_ref(self):
        action = _action(defect_ref="Backlit_Subject@Person")
        assert sanitize_actions([action])[0].defect_ref == "backlit_subject@person"

    def test_vlm_json_keeps_defect_ref_and_tolerates_garbage(self):
        base = {
            "region_id": "face",
            "target_prompt": "face",
            "detected_issue": "backlit_subject",
            "operation": "gamma_correct",
            "parameters": {"gamma": 1.5},
        }
        plan = _plan_from_vlm_json(
            {
                "reasoning": "x",
                "actions": [
                    {**base, "defect_ref": "backlit_subject@person"},
                    {**base, "defect_ref": 42},
                ],
            },
            1,
        )
        assert [a.defect_ref for a in plan.actions] == ["backlit_subject@person", None]

    def test_preserve_person_guards_face_action(self):
        plan = TreatmentPlan(
            reasoning="x",
            actions=[
                _action(
                    operation="sharpen",
                    parameters={"method": "unsharp_mask", "amount": 1.0},
                )
            ],
        )
        guarded = apply_preserve_guard(plan, [PreserveItem(aspect="soft_focus", region="person")])
        assert guarded.actions == []


# ============================================================
# R-05: theo dõi lỗi qua defect_ref
# ============================================================
class TestDefectRefTracking:
    def _diagnosis(self, severity: int = 3) -> DiagnosisReport:
        return DiagnosisReport(
            defects=[Defect(type="backlit_subject", region="person", severity=severity)]
        )

    def test_face_action_without_ref_does_not_track_person_defect(self):
        plan = TreatmentPlan(reasoning="x", actions=[_action()])
        assert _targeted_defects(self._diagnosis(), plan) == []
        assert _needs_verification(self._diagnosis(), plan) is False

    def test_face_action_with_ref_tracks_person_defect(self):
        plan = TreatmentPlan(reasoning="x", actions=[_action(defect_ref="backlit_subject@person")])
        assert _targeted_defects(self._diagnosis(), plan) == [("backlit_subject", "person")]
        assert _needs_verification(self._diagnosis(), plan) is True

    def test_ref_must_match_type_and_region(self):
        plan = TreatmentPlan(reasoning="x", actions=[_action(defect_ref="underexposed@person")])
        assert _targeted_defects(self._diagnosis(), plan) == []

    def test_ref_is_normalized_before_matching(self):
        plan = TreatmentPlan(reasoning="x", actions=[_action(defect_ref="backlit_subject@Person")])
        assert _targeted_defects(self._diagnosis(), plan) == [("backlit_subject", "person")]

    def test_low_severity_defect_does_not_need_verification(self):
        plan = TreatmentPlan(reasoning="x", actions=[_action(defect_ref="backlit_subject@person")])
        assert _needs_verification(self._diagnosis(severity=1), plan) is False


# ============================================================
# R-06: kiểm chứng chéo vùng người - vùng mặt
# ============================================================
class _Mask:
    def __init__(self) -> None:
        self.mask = np.ones((4, 4), dtype=np.float32)


def _fake_stats(monkeypatch, before: Dict[str, float], after: Dict[str, float]) -> None:
    """region_stats của graph trả số đo dựng sẵn: lần gọi đầu là trước, lần hai là sau."""
    calls = iter([before, after])

    def fake(_result: Any, _regions: Any) -> Dict[str, RegionStats]:
        values = next(calls)
        return {name: _stats(name, level) for name, level in values.items()}

    monkeypatch.setattr(graph, "region_stats", fake)


def _resolved() -> DefectChange:
    return DefectChange(
        type="backlit_subject", region="person", before=3, after=0, status="resolved"
    )


class TestBacklitCrossCheck:
    def test_progress_contradiction_is_inconsistent(self):
        before = {"person": _stats("person", 30), "face": _stats("face", 32)}
        after = {"person": _stats("person", 55), "face": _stats("face", 33)}
        progress = backlit_progress("person", before, after)
        assert progress is not None
        assert progress.consistent is False
        assert progress.face_gain == pytest.approx(1.0)

    def test_progress_consistent_when_face_follows(self):
        before = {"person": _stats("person", 30), "face": _stats("face", 32)}
        after = {"person": _stats("person", 48), "face": _stats("face", 52)}
        progress = backlit_progress("person", before, after)
        assert progress is not None and progress.consistent is True

    def test_progress_needs_both_regions(self):
        stats = {"person": _stats("person", 30)}
        assert backlit_progress("person", stats, stats) is None

    def test_face_gain_floor_is_absolute_not_target_level(self):
        # Mặt đã rất sáng sẵn (L* 80) vẫn chỉ cần sáng thêm FACE_MIN_GAIN_L, không có đích tuyệt đối
        before = {"person": _stats("person", 78), "face": _stats("face", 80)}
        after = {
            "person": _stats("person", 78 + FACE_MIN_GAIN_L),
            "face": _stats("face", 80 + FACE_MIN_GAIN_L),
        }
        progress = backlit_progress("person", before, after)
        assert progress is not None and progress.consistent is True

    def test_resolution_cause(self):
        # Chủ thể sáng hẳn lên, nền vẫn sáng hơn nhiều → độ tối giảm
        assert (
            backlit_resolution_cause(
                "person", {"person": _stats("person", 60), "background": _stats("background", 90)}
            )
            == "darkness_reduced"
        )
        # Chủ thể vẫn tối nhưng nền chỉ sáng hơn chút → khoảng cách nhỏ hơn BACKLIT_MIN_GAP_L
        assert (
            backlit_resolution_cause(
                "person", {"person": _stats("person", 30), "background": _stats("background", 40)}
            )
            == "gap_below_min"
        )
        assert backlit_resolution_cause("person", {}) == ""

    def test_resolved_person_with_stagnant_face_becomes_unverified(self, monkeypatch):
        _fake_stats(
            monkeypatch,
            {"person": 30, "face": 32, "background": 90},
            {"person": 55, "face": 33, "background": 90},
        )
        changes = [_resolved()]
        _verify_backlit(changes, {"face": _Mask(), "person": _Mask()}, None, None, {"person"})
        assert changes[0].status == "unverified"
        assert changes[0].cause == "darkness_reduced"
        assert "mặt" in changes[0].detail

    def test_resolved_person_with_brighter_face_stays_resolved(self, monkeypatch):
        _fake_stats(
            monkeypatch,
            {"person": 30, "face": 32, "background": 90},
            {"person": 50, "face": 54, "background": 90},
        )
        changes = [_resolved()]
        _verify_backlit(changes, {"face": _Mask(), "person": _Mask()}, None, None, {"person"})
        assert changes[0].status == "resolved"

    def test_without_face_mask_nothing_changes(self, monkeypatch):
        monkeypatch.setattr(graph, "region_stats", lambda *_: pytest.fail("must not measure"))
        changes = [_resolved()]
        _verify_backlit(changes, {"person": _Mask()}, None, None, {"person"})
        assert changes[0].status == "resolved"

    def test_other_defects_are_not_touched(self, monkeypatch):
        _fake_stats(monkeypatch, {"person": 30, "face": 32}, {"person": 55, "face": 33})
        noise = DefectChange(type="noise", region="person", before=2, after=0, status="resolved")
        _verify_backlit([noise], {"face": _Mask()}, None, None, {"person"})
        assert noise.status == "resolved"

    @pytest.mark.parametrize(
        ("statuses", "expected"),
        [
            (["unverified"], "STOP_BEST_EFFORT"),
            (["resolved", "unverified"], "STOP_BEST_EFFORT"),
            (["improved", "unverified"], "STOP_BEST_EFFORT"),
            (["resolved"], "SHIP"),
        ],
    )
    def test_unverified_never_ships(self, statuses, expected):
        changes = [
            DefectChange(type="noise", region="full", before=2, after=0, status=status)
            for status in statuses
        ]
        assert _decide_tracked(changes) == expected


# ============================================================
# R-07: black_lift
# ============================================================
def _portrait() -> np.ndarray:
    """Nền sáng, 'tóc' đen ở nửa trên, 'mặt' tông trung ở giữa."""
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    image[0:80, 20:180] = 8  # tóc/áo đen
    image[90:170, 70:130] = 90  # mặt tối
    return image


def _face_mask() -> np.ndarray:
    mask = np.zeros((200, 200), dtype=np.float32)
    mask[90:170, 70:130] = 1.0
    return mask


class TestBlackLift:
    def test_detects_lifted_hair_outside_face(self):
        original = _portrait()
        lifted = original.copy()
        lifted[0:80, 20:180] = 70  # tóc đen bị nhấc thành xám
        lifted[90:170, 70:130] = 140  # mặt cũng sáng lên (được phép)
        effect = black_lift(original, lifted, _face_mask())
        assert effect is not None
        assert effect.kind == "black_lift"
        assert effect.value > 20
        assert effect.chroma_shift is not None
        assert effect.bbox is not None and effect.bbox[3] <= 80 + 1

    def test_never_reaches_rollback_severity(self):
        original = _portrait()
        lifted = original.copy()
        lifted[0:80, 20:180] = 120
        effect = black_lift(original, lifted, _face_mask())
        assert effect is not None
        assert effect.severity <= BLACK_LIFT_MAX_SEVERITY < graph.SIDE_EFFECT_ROLLBACK_SEVERITY

    def test_face_only_brightening_is_not_reported(self):
        original = _portrait()
        original[90:170, 70:130] = 12  # mặt gần như đen (ngược sáng nặng)
        rescued = original.copy()
        rescued[90:170, 70:130] = 120
        assert black_lift(original, rescued, _face_mask()) is None

    def test_no_dark_region_means_no_report(self):
        original = np.full((200, 200, 3), 150, dtype=np.uint8)
        brighter = np.clip(original.astype(int) + 40, 0, 255).astype(np.uint8)
        assert black_lift(original, brighter, None) is None

    def test_untouched_dark_region_is_not_reported(self):
        original = _portrait()
        changed = original.copy()
        changed[90:170, 70:130] = 140  # chỉ mặt đổi, tóc giữ nguyên
        assert black_lift(original, changed, _face_mask()) is None

    def test_identical_images(self):
        original = _portrait()
        assert black_lift(original, original.copy(), _face_mask()) is None


class TestSideEffectsCheck:
    def _state(self, previous, current, original=None) -> Dict[str, Any]:
        return {
            "previous_image": previous,
            "current_image": current,
            "original_image": original,
            "diagnosis": None,
        }

    def test_not_run_without_reference_images(self):
        assert _side_effects_check(self._state(None, _portrait())).status == "not_run"

    def test_both_directions_and_black_lift_recorded(self):
        original = _portrait()
        lifted = original.copy()
        lifted[0:80, 20:180] = 90
        check = _side_effects_check(self._state(original, lifted, original))
        assert check.status == "ok"
        assert [e["kind"] for e in check.previous if e["kind"] == "black_lift"] == ["black_lift"]
        assert check.original == check.previous

    def test_original_direction_differs_from_previous_in_later_iterations(self):
        original = _portrait()
        middle = original.copy()
        middle[0:80, 20:180] = 90  # vòng 1 đã làm xám tóc
        final = middle.copy()
        final[200 - 20 :, :] = 255  # vòng 2 chỉ đụng chân ảnh
        check = _side_effects_check(self._state(middle, final, original))
        previous_kinds = {e["kind"] for e in check.previous}
        original_kinds = {e["kind"] for e in check.original}
        assert "black_lift" not in previous_kinds
        assert "black_lift" in original_kinds

    def test_failure_is_unavailable_never_an_empty_ok(self, monkeypatch):
        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("measure failed")

        monkeypatch.setattr(graph, "side_effects", boom)
        image = _portrait()
        check = _side_effects_check(self._state(image, image.copy(), image))
        assert isinstance(check, SideEffectCheck)
        assert check.status == "unavailable"
        assert "measure failed" in (check.error or "")
        assert check.previous == [] and check.original == []
