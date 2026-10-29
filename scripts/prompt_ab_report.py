"""Xuất bảng kỹ thuật và gallery chấm mù từ bằng chứng A/B đã lưu."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import statistics
from pathlib import Path

ROIS = {"backlit_01": (0.28, 0.33, 0.70, 0.67), "backlit_02": (0.10, 0.27, 0.34, 0.44),
        "noisy_01": (0.30, 0.23, 0.56, 0.59), "already_good_01": (0.37, 0.28, 0.65, 0.63)}


def pixel_metrics(folder, case_id):
    """Số đo mô tả trên ROI cố định; không chuyển thành điểm đẹp/xấu."""
    import cv2
    import numpy as np
    from PIL import Image
    before, after = [np.asarray(Image.open(folder / name).convert("RGB"), dtype=np.float32) for name in ("input.png", "output.png")]
    h, w = before.shape[:2]
    roi = ROIS.get(case_id, (0.25, 0.25, 0.75, 0.75))
    x1, y1, x2, y2 = [round(v*(w if n % 2 == 0 else h)) for n, v in enumerate(roi)]
    mask = np.zeros((h,w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    delta = after-before
    weights = np.array([0.2126, 0.7152, 0.0722])
    result = {"roi_luma_delta": float((delta @ weights)[mask].mean()),
              "outside_roi_mae": float(np.abs(delta)[~mask].mean()),
              "whole_mae": float(np.abs(delta).mean())}
    if case_id == "noisy_01":
        # Patch tường bên trái thập giá, chọn theo ảnh gốc trước so sánh đầu ra.
        for label, im in [("before",before),("after",after)]:
            patch = (im @ weights)[int(.33*h):int(.48*h),int(.35*w):int(.42*w)].astype(np.float32)
            residual = patch-cv2.GaussianBlur(patch, (0,0), 1)
            result[f"flat_residual_{label}"] = float(np.median(np.abs(residual-np.median(residual))))
    return result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    protocol = read(root / "protocol.json")
    ROIS.update({c["id"]: c["roi"] for c in protocol["cases"] if "roi" in c})
    rows, mapping, cards = [], {}, []
    for case in protocol["cases"]:
        for repeat in range(protocol["repeats"]):
            pair = {}
            for version in ("v1", "v2"):
                folder = root / case["id"] / str(repeat) / version
                if not (folder / "result.json").exists():
                    continue
                result, state = read(folder / "result.json"), read(folder / "state.json")
                calls = read(folder / "calls.json") if (folder / "calls.json").exists() else []
                errors = sum("error" in c for c in calls)
                row = {"image": case["id"], "repeat": repeat, "version": version,
                       "calls": len(calls), "api_errors": errors, "seconds": result["seconds"],
                       "changed": result["changed"], "decision": result["decision"],
                       "outcome": (result.get("outcome") or {}).get("status"),
                       "iterations": result["iterations"], "plan_fallbacks": result["fallbacks"],
                       "output_sha256": state["current_image"]["sha256"],
                       "actions_executed": sum(not e.get("skipped", False) for h in state.get("history", []) for e in h.get("executed", [])),
                       "rolled_back": sum(bool(h.get("rolled_back")) for h in state.get("history", [])),
                       "no_change_expected_draft": case["expected"].get("empty_plan"),
                       "initial_defects": [(d["type"], d["region"], d["severity"]) for d in ((state.get("history") or [{}])[0].get("diagnosis") or {}).get("defects", [])]}
                row["skips"] = [e.get("skip_reason") for h in state.get("history", []) for e in h.get("executed", []) if e.get("skipped")]
                row["phase_calls"] = {
                    "perceive": sum("GIAI ĐOẠN 1:" in c["system"] for c in calls),
                    "verify": sum("BƯỚC KIỂM CHỨNG" in c["system"] for c in calls),
                    "review": sum("người duyệt ảnh độc lập" in c["system"] for c in calls),
                    "plan": sum("Toolbox hợp lệ" in c["system"] for c in calls),
                }
                row["request_text_chars"] = sum(len(c["system"])+sum(len(x) for x in c["contents"] if isinstance(x,str)) for c in calls)
                row["image_parts"] = sum(isinstance(x,dict) and "image" in x for c in calls for x in c["contents"])
                rows.append(row)
                row.update(pixel_metrics(folder, case["id"]))
                pair[version] = folder
            if len(pair) != 2:
                continue
            key = f"{case['id']}-{repeat}"
            order = ["v1", "v2"] if hashlib.sha256(key.encode()).digest()[0] % 2 else ["v2", "v1"]
            mapping[key] = dict(zip(["X", "Y"], order))
            blind_dir = root / "blind" / key
            blind_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pair["v1"] / "input.png", blind_dir / "input.png")
            for label, version in zip(["X", "Y"], order):
                shutil.copyfile(pair[version] / "output.png", blind_dir / f"{label}.png")
            # Sheet dùng nhãn X/Y, thêm crop cùng tọa độ để soi trực quan.
            from PIL import Image, ImageDraw, ImageOps
            images = [Image.open(pair["v1"] / "input.png").convert("RGB")] + [Image.open(pair[v] / "output.png").convert("RGB") for v in order]
            sheet = Image.new("RGB", (1200, 1050), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            for col, (label, im) in enumerate(zip(["Original", "X", "Y"], images)):
                draw.text((col*400+8, 8), label, fill="black")
                preview = ImageOps.contain(im, (395, 620))
                sheet.paste(preview, (col*400+(400-preview.width)//2, 30))
                roi = ROIS.get(case["id"], (0.25, 0.25, 0.75, 0.75))
                crop = im.crop(tuple(round(v*(im.width if n % 2 == 0 else im.height)) for n,v in enumerate(roi)))
                # Giữ crop ở tỷ lệ pixel 1:1; lấy tâm ROI nếu vùng lớn hơn ô.
                if crop.width > 395 or crop.height > 360:
                    cw, ch = min(395, crop.width), min(360, crop.height)
                    left, top = (crop.width-cw)//2, (crop.height-ch)//2
                    crop = crop.crop((left, top, left+cw, top+ch))
                crop.save(blind_dir / f"{label}-crop.png")
                sheet.paste(crop, (col*400+(400-crop.width)//2, 680))
            sheets = root / "sheets"
            sheets.mkdir(exist_ok=True)
            sheet.save(sheets / f"{key}.png")
            def url(path):
                return html.escape(path.relative_to(root).as_posix(), quote=True)
            credit = f"{case['author']} · {case['license']}"
            goals = {"backlit_01": "Đối chứng: khuôn mặt vốn đủ rõ; tránh nâng sáng không cần thiết.",
                     "backlit_02": "Theo nhãn dự thảo: làm rõ mặt tối vừa phải, giữ nền và tông ảnh. Phong cách gốc có thể gây bất đồng.",
                     "noisy_01": "Giảm hạt nhiễu, giữ cạnh và chi tiết kiến trúc, tránh bệt màu.",
                     "already_good_01": "Đối chứng: giữ nguyên ảnh tốt, màu da, hình xăm và texture."}
            cards.append(f'''<section><h2>{html.escape(key)}</h2><p>{html.escape(case.get('intent', goals.get(case['id'],'')))}</p><p><a href="{html.escape(case['source_page'], quote=True)}">{html.escape(credit)}</a></p>
<div class="images"><figure><figcaption>Gốc</figcaption><img src="{url(blind_dir / 'input.png')}"></figure>
<figure><figcaption>X</figcaption><img src="{url(blind_dir / 'X.png')}"></figure>
<figure><figcaption>Y</figcaption><img src="{url(blind_dir / 'Y.png')}"></figure></div>
<label>Đánh giá: <select data-key="{key}"><option value="">Chưa chấm</option><option>X tốt hơn</option><option>Y tốt hơn</option><option>Ngang nhau</option><option>Cả hai không đạt</option><option>Chưa rõ</option></select></label>
<input data-key="{key}-notes" placeholder="Vùng, mục tiêu, tác dụng phụ quan sát được"></section>''')
    (root / "blind-key.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (root / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = list(root.glob("*/*/*/error.json"))
    expected = len(protocol["cases"])*protocol["repeats"]*2
    lines = ["# Pilot V1/V2 — kết quả vận hành", "", "Chưa có chấm chất lượng độc lập. ACCEPT/NO_CHANGE của pipeline không phải điểm chất lượng.",
             f"Hoàn tất {len(rows)}/{expected} lượt; process lỗi {len(failed)}. Thời gian là pipeline, không gồm khởi động Python/import.",
             "", "| Ảnh | Lượt | Ver | Calls | Lỗi API | Giây | Đổi pixel | Action thực thi | Rollback | Kết cục pipeline |", "|---|---|---|---:|---:|---:|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['image']} | {r['repeat']+1} | {r['version']} | {r['calls']} | {r['api_errors']} | {r['seconds']:.1f} | {r['changed']} | {r['actions_executed']} | {r['rolled_back']} | {r['outcome']} |")
    lines += ["", "## Tổng hợp", "", "| Ver | Số lượt | Lỗi API | Calls tổng | Thời gian trung vị | Số lượt đổi pixel |", "|---|---:|---:|---:|---:|---:|"]
    for version in ("v1", "v2"):
        subset = [r for r in rows if r["version"] == version]
        if subset:
            lines.append(f"| {version} | {len(subset)} | {sum(r['api_errors'] for r in subset)} | {sum(r['calls'] for r in subset)} | {statistics.median(r['seconds'] for r in subset):.1f} | {sum(r['changed'] for r in subset)} |")
    lines += ["", "## Theo ảnh (số đo mô tả, không phải điểm chất lượng)", "", "| Ảnh | Ver | Đổi ảnh / lượt | Output khác nhau | Calls TB | Giây trung vị | Δ luma ROI TB |", "|---|---|---:|---:|---:|---:|---:|"]
    for case in protocol["cases"]:
        for version in ("v1", "v2"):
            subset = [r for r in rows if r["image"] == case["id"] and r["version"] == version]
            if subset:
                lines.append(f"| {case['id']} | {version} | {sum(r['changed'] for r in subset)}/{len(subset)} | {len(set(r['output_sha256'] for r in subset))} | {statistics.mean(r['calls'] for r in subset):.1f} | {statistics.median(r['seconds'] for r in subset):.1f} | {statistics.mean(r['roi_luma_delta'] for r in subset):.2f} |")
    lines += ["", "Số output khác nhau chỉ phản ánh ổn định pixel. ROI cố định chọn theo ảnh gốc; tăng sáng không tự chứng minh cải thiện.", "Nhãn manifest là dự thảo. Chưa kiểm thử Clarify/Refine qua hội thoại trong pilot này; holdout chưa dùng."]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    page = '''<!doctype html><meta charset="utf-8"><title>Prompt A/B — chấm mù</title>
<style>body{font:16px system-ui;margin:24px;background:#eee;color:#222}section{background:white;padding:20px;margin:20px 0}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;max-height:650px;object-fit:contain;background:#ddd}input{width:65%;padding:8px}select{padding:8px}figcaption{font-weight:bold}a{color:#345}</style>
<h1>So sánh mù ảnh đầu ra</h1><p>Ảnh gốc / X / Y. Click ảnh để mở ở độ phân giải gốc. Chấm đạt mục tiêu, giữ chi tiết và tác dụng phụ. Chưa mở blind-key.json trước khi chấm.</p>
<button onclick="exportRatings()">Xuất đánh giá JSON</button>''' + "\n".join(cards) + '''<script>
const fields=[...document.querySelectorAll('[data-key]')];
for(const f of fields){f.value=localStorage.getItem('prompt-ab:'+f.dataset.key)||'';f.onchange=()=>localStorage.setItem('prompt-ab:'+f.dataset.key,f.value)}
for(const img of document.images){img.onclick=()=>window.open(img.src,'_blank')}
function exportRatings(){const data=Object.fromEntries(fields.map(f=>[f.dataset.key,f.value]));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='ratings.json';a.click()}
</script>'''
    (root / "gallery.html").write_text(page, encoding="utf-8")
    print(f"Exported {len(rows)} runs, {len(cards)} paired comparisons")


if __name__ == "__main__":
    main()
