# Analysis v2 — tiến độ (lead)

Nhánh tích hợp `feat/analysis-v2`. Chưa push, chưa mở PR.

## Lane

| Wave | Lane | Phase | Merge | Tiêu chí | Ghi chú |
|---|---|---|---|---|---|
| 1 | bench | P0.2–0.4 | merge bench lane (2026-10-25) | đạt (trừ sai số synthetic → lead đo, xem dưới) | ngoài phạm vi nhưng chấp nhận: `.gitignore`, `data/calibration_sources.json`, `data/download_real_benchmark.py` (tải full-res), `test_build_calibration_set.py`. Chú thích `annotated_by: "agent"` trong manifest chưa được người duyệt. |
| 1 | measure | P1 | merge measure lane (2026-10-25) | đạt | lệch khỏi plan: quyết định trong `measurement_synthetic.md`; numpy 1.26.4: 70/70 xanh |
| 1 | region | P4a | merge region lane (2026-10-25) | đạt | contract-changes 1 (A), 2 (chấp nhận), 3 (chấp nhận). Cần Person 2 review. |
| 1 | meta | P3 | merge meta lane (2026-10-25) | đạt | phiên tương tác → lane `agent`; Gradio hoãn |
| 2 | calib | P2 | – | – | mở sau wave 1 |
| 2 | agent | P4b + P5 | – | – | mở sau wave 1 |
| 3 | fusion | P6 + P7 | – | – | chờ checkpoint sau `calib` |

Sau mỗi merge: `ruff check src/ tests/` sạch; `pytest tests/unit/ -q`: 1173 → 1243 → 1267 → 1296
passed (sau bench / measure / region / meta), thêm 5 test của `benchmark_measurement`.

## Số đo

| Bộ | Config | Recall | FP rate ảnh sạch | MAE severity | Nguồn |
|---|---|---|---|---|---|
| real (30 ảnh chấm, 13 sạch/chủ ý) | baseline | 0.47 | 0.54 | 1.00 | `baseline.md` |
| real | vlm_only | 0.53 | 0.62 | 0.88 | `baseline.md` |
| real | measure_only | – | – | – | chờ `calib` |
| real | fused | – | – | – | chờ `fusion` |

Mục tiêu cuối: FP ≤ 0.10, recall ≥ 0.85, MAE ≤ 0.6.

Sai số ước lượng P1 trên synthetic (chi tiết `measurement_synthetic.md`): σ nhiễu 5.8 % MAE;
σ defocus 13.1 % (lệch thấp); độ rộng cạnh tương đối 7.4 %; hướng mờ chuyển động 10.9° (sai trên
texture có hướng); nguồn sáng 5.5° (ám mạnh bị đo thiếu).

## Checkpoint (người dùng đặt)

Sau khi merge `calib`: chạy lại benchmark. Nếu FP rate ảnh sạch (already_good + intentional_style)
không giảm rõ từ 0.54 về phía ≤ 0.10 → dừng, báo người dùng trước khi mở wave 3 (`fusion`).

## Hoãn

- Gradio (`frontend/app.py`) chưa nhận metadata: `gr.Image(type="numpy")` bỏ bytes file. Không
  đổi sang `type="filepath"` lúc này (người dùng quyết định, 2026-10-25).
- `measure`: zone histogram 11 vùng, vị trí blob cháy, Tenengrad (xem `measurement_synthetic.md`).

## Review cần người

- P1/P2 (`measure`, `calib`): Person 1.
- P4a (`region`): Person 2.
