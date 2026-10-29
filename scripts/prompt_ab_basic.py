"""Frozen, paired development evaluation on visually screened real photographs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prompt_ab import dump

SELECTED = {
    "underexposed": [1, 3, 4], "local_dark": [1, 2],
    "overexposed": [1], "low_contrast": [1], "noise": [1, 2, 3],
    "oversaturated": [1, 3], "already_good": [1, 6],
    "intentional_style": [1], "highlight_clipping": [1], "dark_noisy": [1],
}


def prepare(out):
    import cv2
    import numpy as np
    from PIL import Image
    manifest = json.loads((ROOT / "data/basic_test_manifest.json").read_text(encoding="utf-8"))
    ids = {f"basic_{category}_{n:02d}" for category, ns in SELECTED.items() for n in ns}
    cases = [c for c in manifest["images"] if c["id"] in ids]
    assert len(cases) == 17
    prior = ROOT / "data/prompt_ab/2026-10-29-live/snapshot"
    snapshot = out / "snapshot"
    shutil.copytree(prior / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (snapshot / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/prompt_ab.py", snapshot / "scripts/prompt_ab.py")
    shutil.copyfile(prior / "scripts/benchmark_brain.py", snapshot / "scripts/benchmark_brain.py")
    for c in cases:
        c["group"] = ("control" if c["category"] in ("already_good", "intentional_style") else
                      "boundary" if c["category"] in ("highlight_clipping", "dark_noisy") else "basic")
        path = ROOT / manifest["dataset_root"] / c["file"]
        c["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        c["roi"] = {"basic_local_dark_01": [.32, .05, .52, .25],
                    "basic_local_dark_02": [.30, .28, .65, .80]}.get(c["id"], [.25, .25, .75, .75])
        image = np.asarray(Image.open(path).convert("RGB"))
        category = c["category"]
        # Fixed references are illustrative algorithms, never clean ground truth.
        if category in ("underexposed", "local_dark", "dark_noisy"):
            baseline = np.uint8(np.clip(255 * (image.astype(float) / 255) ** .72, 0, 255))
            recipe = "Global gamma 0.72 (does not localize the requested subject)"
        elif category in ("overexposed", "highlight_clipping"):
            baseline = np.uint8(255 * (image.astype(float) / 255) ** 1.25)
            recipe = "Global gamma 1.25; cannot reconstruct clipped detail"
        elif category == "noise":
            baseline = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
            recipe = "OpenCV NLM h=7 hColor=7 template=7 search=21"
        elif category == "low_contrast":
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            baseline = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            recipe = "CLAHE LAB L clip=2 grid=8x8"
        elif category == "oversaturated":
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.uint8(hsv[:, :, 1].astype(float) * .8)
            baseline = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            recipe = "HSV saturation x0.8"
        else:
            baseline, recipe = image, "Identity (keep original)"
        c["baseline_recipe"] = recipe
        dest = out / "references" / c["id"]
        dest.mkdir(parents=True)
        Image.fromarray(image).save(dest / "input.png")
        Image.fromarray(baseline).save(dest / "baseline.png")
    files = [p for p in snapshot.rglob("*.py")]
    dump(out / "protocol.json", {
        "cases": cases, "repeats": 1, "dataset_root": manifest["dataset_root"],
        "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "note": "Development screen, 17 independent images, one run/version. No statistical superiority claim. Core src frozen from previous pilot; V2 unchanged. Natural-language intent passed as notes to both, no expected labels injected. Existing no-action gate can prevent notes reaching Plan; count that as pipeline behavior. Clarify/Refine conversation not evaluated. Fixed references are not ground truth. Controls/boundaries reported separately.",
        "hashes": {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/prompt_ab/2026-10-29-basic")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "protocol.json").exists():
        prepare(out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if args.prepare_only:
        print(f"Prepared {len(protocol['cases'])} cases: {out}")
        return
    from scripts.benchmark_brain import _load_api_key
    _load_api_key()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY unavailable")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", GEMINI_MODEL=protocol["model"])
    for name, file in {"REGION_FACE_MODEL_PATH": "mediapipe/face_detection_full_range.tflite",
                       "REGION_FACE_LANDMARKER_MODEL_PATH": "mediapipe/face_landmarker.task",
                       "REGION_DINO_MODEL_PATH": "segmentation/grounding-dino-tiny",
                       "REGION_MOBILE_SAM_CHECKPOINT": "segmentation/mobile_sam.pt"}.items():
        env[name] = str(ROOT / "models" / file)
    def run_pair(index, case):
        source = ROOT / protocol["dataset_root"] / case["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["input_sha256"]
        for ver in (["v1", "v2"] if index % 2 == 0 else ["v2", "v1"]):
            folder = out / case["id"] / "0" / ver
            if (folder / "result.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            print(f"RUN {case['id']} {ver}", flush=True)
            with (folder / "process.log").open("w", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, str(out / "snapshot/scripts/prompt_ab.py"),
                    "--version", ver, "--image", str(source), "--intent-text", case["intent"],
                    "--output", str(folder)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                dump(folder / "error.json", {"exit_code": process.returncode})
            print(f"FINISHED {case['id']} {ver} exit={process.returncode}", flush=True)
        return case["id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_pair, i, c) for i, c in enumerate(protocol["cases"])]
        for future in as_completed(futures):
            print("PAIR " + future.result(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
