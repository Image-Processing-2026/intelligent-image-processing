"""Thử nghiệm prompt V1/V2 biệt lập; lưu bằng chứng và không đổi mặc định sản phẩm."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def install_v2():
    """Chỉ cài các override trong process benchmark V2."""
    from src.agent import fusion as f
    from src.agent import intent, refine
    from src.agent import perception as p
    from src.agent import review as r
    from src.agent import vlm_diagnostician as v
    p.PERCEIVE_PROMPT += """
Phân biệt quan sát trực tiếp với suy đoán phong cách. Chỉ gán preserve khi có dấu hiệu
thẩm mỹ rõ; thiếu sáng trên mặt không tự chứng minh chủ ý silhouette. Khi chưa chắc,
nêu sự mơ hồ trong summary; không tự khẳng định ý định người chụp.
"""
    v.SYSTEM_PROMPT += """
QUY ƯỚC LẬP KẾ HOẠCH: Khi đã có chẩn đoán, nhiệm vụ là chuyển lỗi đã hợp nhất thành action,
không tự thêm lỗi. Giới hạn toolbox và chống chỉ định luôn bắt buộc. Yêu cầu rõ của người
dùng quyết định mục tiêu; preserve do AI suy đoán không được ghi đè yêu cầu đó.
Trong giới hạn này, dùng công thức phù hợp; nếu lệch để đáp ứng mục tiêu hãy nêu lý do.
Phản hồi reviewer là bằng chứng cần đối chiếu lại với ảnh, không phải chân lý.
Action bị bỏ qua không có nghĩa đã sửa lỗi; không lặp cùng cách nếu không tác động được vùng.
"""
    def history_text(history):
        if not history:
            return ""
        rows = []
        for h in history[-2:]:
            rows.append({"iteration": h.iteration, "executed": [x.model_dump() for x in h.executed],
                         "planned": h.plan.model_dump() if h.plan else None,
                         "metrics_before": h.metrics_before, "metrics_after": h.metrics_after,
                         "review": h.review.model_dump() if h.review else None,
                         "defect_status": h.defect_status, "decision": h.decision,
                         "rolled_back": h.rolled_back})
        return "\nPHẢN HỒI CÁC VÒNG TRƯỚC (planned là dự kiến, executed mới là thực thi):\n" + json.dumps(rows, ensure_ascii=False, default=str)
    v._build_history_feedback = history_text
    p._build_history_feedback = history_text
    f.VERIFY_PROMPT += '\nNếu ảnh/crop không đủ để kết luận, trả verdict "uncertain"; không dùng not_present thay cho thiếu bằng chứng.'
    p.VERIFY_PROMPT = f.VERIFY_PROMPT
    schema = copy.deepcopy(f.VERIFY_RESPONSE_SCHEMA)
    schema["properties"]["answers"]["items"]["properties"]["verdict"]["enum"].append("uncertain")
    p.VERIFY_RESPONSE_SCHEMA = schema
    def parse_answers(data):
        # Không có câu trả lời = đi theo nhánh dự phòng chưa kiểm chứng hiện hữu.
        return {a.hypothesis_id: a for x in data.get("answers", [])
                if x.get("verdict") != "uncertain" for a in [f.VerifyAnswer(**x)]}
    p.parse_verify_answers = parse_answers
    def question(h, crops=()):
        phenomenon, intentional, texture = f.VERIFY_PHRASES.get(h.type, (h.type, "chủ ý", "texture"))
        text = (f"Quan sát vùng '{h.region}': {phenomenon} có nhìn thấy rõ không? "
                f"Xem crop {', '.join(crops)}. Phân biệt lỗi cần sửa với {intentional}, {texture}, "
                "không có hiện tượng, hoặc chưa đủ bằng chứng. " + f._measured_text(h))
        return f.VerifyQuestion(hypothesis_id=h.id, defect_type=h.type, region=h.region, question=text, crops=list(crops))
    f.verify_question = question
    r.REVIEW_PROMPT += "\nĐánh giá mức đạt mục tiêu và bảo toàn phong cách riêng biệt. Không coi ảnh sáng hơn là tự động tốt hơn. Nếu vùng mục tiêu quá nhỏ không đủ soi, trả UNCERTAIN."
    refine.FEEDBACK_PROMPT += "\nYêu cầu sở thích (ấm hơn, tối hơn) không cần xác nhận là lỗi. Phân biệt với nhận định kỹ thuật. 'Vẫn hơi' không mặc nhiên cần strength 2; xét mức độ người dùng mô tả."
    old_questions = intent.build_questions
    def questions(d):
        result = old_questions(d)
        if result or d is None or not d.ask_user:
            return result
        return [intent.Question(id="approach", text="Có hiện tượng chưa chắc là lỗi: " + intent.recommendation_text(d) + " Bạn muốn giữ hay mô tả cách sửa?", choices=[intent.Choice(value="recommend", label="Giữ nguyên"), intent.Choice(value="custom", label="Mô tả cách sửa")], default="recommend")]
    intent.build_questions = questions
    from src.agent import graph
    graph.build_questions = questions


def worker(args):
    import numpy as np
    from PIL import Image, ImageOps

    from scripts.benchmark_brain import _load_api_key
    _load_api_key()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY unavailable; real A/B requires API access")
    from src.agent import perception as p
    from src.agent import refine, user_analysis
    from src.agent import review as r
    from src.agent import vlm_diagnostician as v
    from src.agent.graph import run_pipeline
    from src.agent.memory import configure_case_memory
    from src.analyzer_evaluator.metadata import read_metadata
    configure_case_memory(None)
    if args.version == "v2":
        install_v2()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    calls = []
    original_call = v.call_gemini_json
    def capture(api_key, system_prompt, contents, schema, **kwargs):
        index = len(calls)
        item = {"system": system_prompt, "schema": schema, "contents": [], "kwargs": kwargs}
        calls.append(item)
        for n, part in enumerate(contents):
            if isinstance(part, Image.Image):
                name = f"call_{index:02d}_image_{n:02d}.png"
                part.save(out / name)
                item["contents"].append({"image": name})
            else:
                item["contents"].append(str(part))
        start = time.monotonic()
        try:
            answer = original_call(api_key, system_prompt, contents, schema, **kwargs)
            item["response"] = answer
            return answer
        except Exception as exc:
            item["error"] = str(exc)
            raise
        finally:
            item["seconds"] = time.monotonic() - start
            dump(out / "calls.json", calls)
    for module in (v, p, r, refine, user_analysis):
        module.call_gemini_json = capture
    start = time.monotonic()
    image = np.asarray(ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")).copy()
    Image.fromarray(image).save(out / "input.png")
    from src.agent.state import IntentProfile
    intent = IntentProfile(notes=args.intent_text) if args.intent_text else None
    state = run_pipeline(image, max_iterations=3, intent=intent, image_metadata=read_metadata(Path(args.image).read_bytes()))
    Image.fromarray(state["current_image"]).save(out / "output.png")
    def serial(value):
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        if hasattr(value, "model_dump"):
            return serial(value.model_dump())
        if isinstance(value, dict):
            return {k: serial(x) for k, x in value.items()}
        if isinstance(value, (list, tuple)):
            return [serial(x) for x in value]
        return value
    dump(out / "state.json", serial(state))
    history = state.get("history", [])
    dump(out / "result.json", {"version": args.version, "seconds": time.monotonic()-start,
         "calls": len(calls), "decision": state.get("decision"),
         "outcome": serial(state.get("outcome")), "iterations": len(history),
         "changed": not np.array_equal(image, state["current_image"]),
         "fallbacks": sum(h.plan.source != "vlm" for h in history if h.plan),
         "review": serial(state.get("review"))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"])
    parser.add_argument("--image")
    parser.add_argument("--intent-text", default="")
    parser.add_argument("--output", default="data/prompt_ab/2026-10-29")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ids", nargs="+", default=["backlit_01", "backlit_02", "noisy_01", "already_good_01"])
    args = parser.parse_args()
    if args.version:
        worker(args)
        return
    import subprocess
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for key, relative in {
        "REGION_FACE_MODEL_PATH": "models/mediapipe/face_detection_full_range.tflite",
        "REGION_FACE_LANDMARKER_MODEL_PATH": "models/mediapipe/face_landmarker.task",
        "REGION_DINO_MODEL_PATH": "models/segmentation/grounding-dino-tiny",
        "REGION_MOBILE_SAM_CHECKPOINT": "models/segmentation/mobile_sam.pt",
    }.items():
        os.environ.setdefault(key, str(ROOT / relative))
    from scripts.benchmark_brain import _load_api_key
    _load_api_key()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY unavailable")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/real_benchmark_manifest.json").read_text(encoding="utf-8-sig"))
    cases = [c for c in manifest["images"] if c["id"] in args.ids]
    if len(cases) != len(args.ids):
        raise ValueError("Unknown case IDs")
    files = [f for f in (ROOT / "src").rglob("*") if f.is_file() and "__pycache__" not in f.parts] + [Path(__file__), ROOT / "scripts/benchmark_brain.py"]
    snapshot = out / "snapshot"
    if not (out / "protocol.json").exists():
        for file in files:
            target = snapshot / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
        dump(out / "protocol.json", {"cases": cases, "repeats": args.repeats,
             "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
             "note": "Exploratory pilot; draft labels; no human blind ratings yet. V2 is experimental. Holdout untouched.",
             "hashes": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}})
    rows = []
    for repeat in range(args.repeats):
        for case in cases:
            versions = ["v1", "v2"] if repeat % 2 == 0 else ["v2", "v1"]
            for version in versions:
                folder = out / case["id"] / str(repeat) / version
                result = folder / "result.json"
                if not result.exists():
                    folder.mkdir(parents=True, exist_ok=True)
                    print(f"RUN {case['id']} {repeat} {version}", flush=True)
                    with (folder / "process.log").open("w", encoding="utf-8") as log:
                        process = subprocess.run([sys.executable, str(snapshot / "scripts/prompt_ab.py"), "--version", version,
                            "--image", str(ROOT / "data/real/benchmark" / case["file"]),
                            "--output", str(folder.resolve())], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode:
                        dump(folder / "error.json", {"exit_code": process.returncode})
                row = {"id": case["id"], "repeat": repeat, "version": version}
                row.update(json.loads(result.read_text(encoding="utf-8")) if result.exists() else {"error": True})
                rows.append(row)
                dump(out / "results.json", rows)
                lines = ["# Exploratory prompt A/B", "", "No independent quality verdict yet. Pipeline verdicts are not quality scores.", "", "| Image | Repeat | Version | Calls | Seconds | Changed | Fallbacks | Pipeline decision |", "|---|---|---|---|---|---|---|---|"]
                for x in rows:
                    lines.append(f"| {x['id']} | {x['repeat']} | {x['version']} | {x.get('calls','ERR')} | {round(x.get('seconds',0),1)} | {x.get('changed')} | {x.get('fallbacks')} | {x.get('decision')} |")
                (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
