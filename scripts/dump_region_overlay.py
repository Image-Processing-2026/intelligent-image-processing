"""
Lưu overlay vùng (mask chủ thể) lên ảnh gốc để kiểm tra bằng mắt vùng có khoanh đúng không.

Dùng: python scripts/dump_region_overlay.py <ảnh> <vùng[=ymin,xmin,ymax,xmax]>... [--out overlay.png]
Ví dụ: python scripts/dump_region_overlay.py original.png person=210,340,780,700 face
Box theo quy ước Gemini (ymin, xmin, ymax, xmax) trên thang 0..1000; bỏ box thì resolver tự tìm.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from src.agent.region_overlay import diagnosis_region_overlay
from src.agent.state import DiagnosisReport


def parse_regions(specs: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """'name' hoặc 'name=ymin,xmin,ymax,xmax' → (danh sách vùng, box theo vùng)."""
    names: List[str] = []
    boxes: Dict[str, List[int]] = {}
    for spec in specs:
        name, _, box = spec.partition("=")
        names.append(name)
        if box:
            boxes[name] = [int(value) for value in box.split(",")]
    return names, boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("regions", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("region_overlay.png"))
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    names, boxes = parse_regions(args.regions)
    diagnosis = DiagnosisReport(subjects=names, subject_boxes=boxes)
    overlay, masks = diagnosis_region_overlay(image, diagnosis)
    cv2.imwrite(str(args.out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    for name in names:
        subject = masks.get(name)
        status = (
            f"{subject.source} confidence={subject.confidence:.2f}" if subject else "NOT LOCATED"
        )
        print(f"{name}: {status}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
