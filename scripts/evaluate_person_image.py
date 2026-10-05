"""Visualise Module 2 Region Engine output tren anh nguoi thuc.

Dau vao  : data/image/human-pic.jpg
Dau ra   : artifacts/region_engine_person_output.png

Cach chay (tu thu muc goc repo):
    python scripts/evaluate_person_image.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import src.region_engine.face_detector as face_detector_module  # noqa: E402
from src.region_engine.controller import RegionRequest, resolve_region  # noqa: E402
from src.region_engine.face_detector import FaceDetectionRecord, reset_face_detector  # noqa: E402
from src.region_engine.mask_utils import blend_regions  # noqa: E402
from src.region_engine.controller import capabilities  # noqa: E402

# ---------------------------------------------------------------------------
# Duong dan
# ---------------------------------------------------------------------------

IMAGE_PATH = REPO_ROOT / "data" / "image" / "human-pic.jpg"
OUT_PATH = REPO_ROOT / "artifacts" / "region_engine_person_output.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nap anh goc
# ---------------------------------------------------------------------------

with Image.open(IMAGE_PATH) as _img:
    ORIGINAL: np.ndarray = np.asarray(_img.convert("RGB"), dtype=np.uint8).copy()

H, W = ORIGINAL.shape[:2]
print(f"[info] Image: {IMAGE_PATH.name}  shape={ORIGINAL.shape}")

# ---------------------------------------------------------------------------
# Kiem tra backend MediaPipe thuc su
# ---------------------------------------------------------------------------

cap = capabilities()
FACE_READY = cap["face"]["ready"]
FACE_MODE = "real" if FACE_READY else "mock"
print(f"[info] Face backend: {FACE_MODE}")
if not FACE_READY:
    missing = [
        k for k, v in cap["face"]["assets_present"].items() if not v
    ] + [
        k for k, v in cap["face"]["dependencies_available"].items() if not v
    ]
    print(f"[warn] Face detection unavailable — missing: {missing}")
    print(f"[warn] Expected model: {cap['face']['model_path']}")

# Fallback fake backend neu chua co model/package
_FACE_RECORD = FaceDetectionRecord(x=700, y=290, width=200, height=210, score=0.95)


class _FakeBackend:
    def detect(self, _image: np.ndarray) -> list[FaceDetectionRecord]:
        return [_FACE_RECORD]

    def close(self) -> None:
        pass


reset_face_detector()
if not FACE_READY:
    face_detector_module._DETECTOR_FACTORY = lambda _path: _FakeBackend()

# ---------------------------------------------------------------------------
# Chay tung loai region qua controller
# ---------------------------------------------------------------------------

results: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}  # name -> (mode, mask, blended)

# Anh 'da xu ly' gia lap: brightened
PROCESSED = np.clip((ORIGINAL.astype(np.float32) ** 0.5) * 16, 0, 255).astype(np.uint8)


def _run(label: str, request: dict | RegionRequest) -> None:
    result = resolve_region(ORIGINAL, request)
    blended = blend_regions(ORIGINAL, PROCESSED, result.mask)
    mode = FACE_MODE if "face" in label else "geometry"
    results[label] = (mode, result.mask, blended)
    area_pct = 100.0 * float(np.mean(result.mask > 0.05))
    print(f"  [{result.status:5s}] {label:22s}  mask_area={area_pct:.1f}%  "
          f"backend={result.metadata.get('backend', '?')}")


print("\n[Module 2] Resolving regions ...")
_run("full",           {"kind": "full"})
_run("bbox (person)",  {"kind": "bbox", "bbox": [180, 200, 1000, 853], "feather_radius": 20})
_run("spatial: top",   RegionRequest(kind="spatial", quadrant="top",    feather_radius=25))
_run("spatial: bottom",RegionRequest(kind="spatial", quadrant="bottom", feather_radius=25))
_run("spatial: left",  RegionRequest(kind="spatial", quadrant="left",   feather_radius=25))
_run("spatial: right", RegionRequest(kind="spatial", quadrant="right",  feather_radius=25))
_run("spatial: center",RegionRequest(kind="spatial", quadrant="center", feather_radius=25))
face_label = f"face ({FACE_MODE})"
_run(face_label,       RegionRequest(kind="face",    feather_radius=20, expand_ratio=0.20))

reset_face_detector()

# ---------------------------------------------------------------------------
# Vẽ mosaic: 4 cột × N hàng
# cột 1: ảnh gốc overlay contour mask
# cột 2: soft-mask (grayscale)
# cột 3: ảnh đã xử lý (PROCESSED)
# cột 4: kết quả blend
# ---------------------------------------------------------------------------

LABELS = list(results.keys())
N = len(LABELS)
COLS = 4
fig, axes = plt.subplots(N, COLS, figsize=(COLS * 5, N * 3.2), dpi=100)
fig.patch.set_facecolor("#1a1a2e")

COL_TITLES = ["Original + region", "Soft-mask", "Processed (brightened)", "Blended result"]
for col_idx, title in enumerate(COL_TITLES):
    axes[0, col_idx].set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)

for row, label in enumerate(LABELS):
    mode, mask, blended = results[label]

    # --- cột 0: ảnh gốc với vùng highlight ---
    ax0 = axes[row, 0]
    overlay = ORIGINAL.copy().astype(np.float32)
    highlight = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    overlay = np.clip(overlay + highlight * 60, 0, 255).astype(np.uint8)
    ax0.imshow(overlay)
    ax0.set_ylabel(label, color="white", fontsize=9, rotation=0, labelpad=80, va="center")

    # --- cột 1: soft-mask grayscale ---
    ax1 = axes[row, 1]
    im = ax1.imshow(mask, cmap="hot", vmin=0, vmax=1)

    # --- cột 2: ảnh xử lý gốc ---
    ax2 = axes[row, 2]
    ax2.imshow(PROCESSED)

    # --- cột 3: blend result ---
    ax3 = axes[row, 3]
    ax3.imshow(blended)

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

plt.suptitle(
    f"Module 2 Region Engine  |  {IMAGE_PATH.name}  ({H}x{W})  |  face={FACE_MODE}",
    color="white", fontsize=13, fontweight="bold", y=1.002,
)
plt.tight_layout(pad=0.5)
fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\n[done] Đã lưu → {OUT_PATH}")
