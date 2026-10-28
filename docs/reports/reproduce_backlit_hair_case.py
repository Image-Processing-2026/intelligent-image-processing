"""Tái lập ca tóc bị nâng sáng để làm bằng chứng cho report, không sửa pipeline."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analyzer_evaluator.measurement.diff import side_effects
from src.processing_engine import apply_gamma
from src.region_engine.subject import resolve_subject_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session = json.loads(args.session.read_text(encoding="utf-8"))
    process = next(run for run in session["runs"] if run["kind"] == "process")
    actions = process["result"]["treatment"]
    diagnosis = next(run for run in session["runs"] if run["kind"] == "analyze")["result"]["diagnosis"]
    original = cv2.cvtColor(cv2.imread(str(args.original)), cv2.COLOR_BGR2RGB)
    expected = cv2.cvtColor(cv2.imread(str(args.result)), cv2.COLOR_BGR2RGB)
    cv2.setRNGSeed(0)
    subject = resolve_subject_mask(
        original, "person", diagnosis["subject_boxes"]["person"], use_semantic=False
    )
    current = original.copy()
    rounds = []
    for action in actions:
        assert action["operation"] == "gamma_correct" and action["target_prompt"] == "person"
        previous = current
        current = apply_gamma(current, mask=subject.mask, gamma=action["parameters"]["gamma"])
        rounds.append({
            "gamma": action["parameters"]["gamma"],
            "side_effects": [effect.model_dump() for effect in side_effects(previous, current)],
        })
    roi = (slice(65, 90), slice(435, 490))
    evidence = {
        "session_id": session["session"]["id"],
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (args.session, args.original, args.result)},
        "shape": list(original.shape),
        "mask_backend": subject.source,
        "mask_area": round(float(subject.mask.mean()), 6),
        "hair_roi_xyxy": [435, 65, 490, 90],
        "hair_rgb_original": original[roi].mean((0, 1)).round(3).tolist(),
        "hair_rgb_result": expected[roi].mean((0, 1)).round(3).tolist(),
        "replay_mae": float(np.abs(current.astype(float) - expected).mean()),
        "pixel_exact": bool(np.array_equal(current, expected)),
        "rounds": rounds,
        "cumulative_side_effects": [effect.model_dump() for effect in side_effects(original, expected)],
    }
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    assert evidence["pixel_exact"], "Replay differs from the supplied result"


if __name__ == "__main__":
    main()
