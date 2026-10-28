"""Aggregate judged scores of a batch run into a comparison table.

scores.json (written by the judge, one entry per image and provider):
{
  "judge": "...", "rubric_version": 1,
  "scores": {
    "<image_id>": {
      "<provider>": {
        "defects":  {"<type>@<region>": 1 | 0.5 | 0, ...},   # expected defects found
        "preserve": {"<aspect>": 1 | 0.5 | 0, ...},           # expected styles recognised
        "plan_ok": 1 | 0.5 | 0,         # fix-or-leave decision matches expected.empty_plan
        "false_defects": ["..."],       # problems claimed that the image does not have
        "hallucinations": ["..."],      # content described that is not in the image
        "harmful": ["..."],             # advice that would damage the image or its style
        "degenerate": true,             # optional: answer unusable (repetition loop, empty)
        "note": "..."
      }
    }
  }
}

Usage: .venv\\Scripts\\python.exe model_lab\\score_report.py model_lab\\batch_runs\\<folder>
Writes <folder>/score_report.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from app import MODELS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "real_benchmark_manifest.json"
COUNTS = ("false_defects", "hallucinations", "harmful")


def avg(values: list[float]) -> float | None:
    return round(mean(values), 2) if values else None


def block(items: list[dict]) -> dict:
    defects = [v for s in items for v in s["defects"].values()]
    preserve = [v for s in items for v in s["preserve"].values()]
    return {
        "defect_recall": avg(defects),
        "preserve_recall": avg(preserve),
        "plan_ok": avg([s["plan_ok"] for s in items]),
        **{c: sum(len(s[c]) for s in items) for c in COUNTS},
        "degenerate": sum(1 for s in items if s.get("degenerate")),
        "images": len(items),
    }


def fmt(value: object) -> str:
    if value is None:
        return "–"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def main() -> None:
    run_dir = Path(sys.argv[1])
    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]
    scored = [
        e for e in entries if not e["expected"].get("ambiguous") and e["id"] in scores["scores"]
    ]
    providers = [p for p in MODELS if any(p in scores["scores"][e["id"]] for e in scored)]
    categories = sorted({e["category"] for e in scored})
    cols = ["defect_recall", "preserve_recall", "plan_ok", *COUNTS, "degenerate", "images"]

    def per(provider: str, category: str | None = None) -> dict:
        return block(
            [
                scores["scores"][e["id"]][provider]
                for e in scored
                if provider in scores["scores"][e["id"]]
                and (category is None or e["category"] == category)
            ]
        )

    lines = [
        "# Free-vision benchmark: judged scores",
        "",
        f"Judge: {scores.get('judge', '?')} · rubric v{scores.get('rubric_version', '?')} · "
        f"{len(scored)} scored images (ambiguous excluded)",
        "",
        "## Overall",
        "",
        "| model | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for p in providers:
        b = per(p)
        lines.append(f"| {MODELS[p]['label']} | " + " | ".join(fmt(b[c]) for c in cols) + " |")
    for metric in ("defect_recall", "preserve_recall", "plan_ok"):
        lines += [
            "",
            f"## {metric} by category",
            "",
            "| category | " + " | ".join(MODELS[p]["label"] for p in providers) + " |",
            "|---" * (len(providers) + 1) + "|",
        ]
        for c in categories:
            lines.append(f"| {c} | " + " | ".join(fmt(per(p, c)[metric]) for p in providers) + " |")
    lines += ["", "## Per image", ""]
    for e in entries:
        if e["id"] not in scores["scores"]:
            continue
        flag = " (ambiguous, not scored)" if e["expected"].get("ambiguous") else ""
        lines.append(f"### {e['id']}{flag}")
        for p in providers:
            s = scores["scores"][e["id"]].get(p)
            if not s:
                continue
            issues = "; ".join(f"{c}: {', '.join(s[c])}" for c in COUNTS if s[c])
            lines.append(
                f"- **{MODELS[p]['label']}**: defects {s['defects'] or '–'} · preserve "
                f"{s['preserve'] or '–'} · plan_ok {s['plan_ok']}"
                + (f" · {issues}" if issues else "")
                + (f" — {s['note']}" if s.get("note") else "")
            )
        lines.append("")
    (run_dir / "score_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(run_dir / "score_report.md")


if __name__ == "__main__":
    main()
