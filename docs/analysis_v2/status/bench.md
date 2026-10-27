# Lane bench — trạng thái

Nhánh: `feat/analysis-v2-bench` · Brief: `docs/analysis_v2/agents/bench.md` · Phase: P0.2–0.4

## Đã làm

- P0.2 `scripts/build_calibration_set.py`: 30 biến thể/ảnh sạch (nhiễu Gaussian/Poisson-Gaussian/chroma,
  mờ Gaussian/chuyển động, gamma, co tương phản, nguồn sáng CCT + tint xanh, bão hòa, JPEG qua Pillow,
  haze `I=J·t+A(1−t)`, 2 tổ hợp, 1 ảnh đối chứng), seed 7, `params.json` cạnh ảnh theo C5.
  5 ảnh sạch (3 already_good + 2 tải từ Commons, giấy phép trong `data/calibration_sources.json`)
  → 150 ảnh trong `data/synthetic/calibration/` (gitignored).
- P0.3 manifest schema_version 2: `severity` cho mọi defect mong đợi (`annotated_by: "agent"`),
  `intentional_style` từ 4 lên 10 ảnh (6 ảnh mới từ Commons, `annotated_by: "agent"`), trường `fullres_file`
  cho nhóm noisy, `download_real_benchmark.py` tải thêm full-res vào `data/real/benchmark_fullres/`.
- P0.4 `scripts/benchmark_scoring.py` (hàm thuần) + `scripts/benchmark_analysis.py` (CLI C5; configs
  `baseline`, `vlm_only`; `measure_only`/`fused` báo NotImplementedError; cờ `--offline`).
- Unit test: `test_benchmark_scoring.py` (13), `test_build_calibration_set.py` (9). Cả bộ
  `pytest tests/unit/` xanh: 1173 passed. `ruff check` sạch trên các file của lane.
- Baseline thật đã chạy và ghi vào `docs/analysis_v2/baseline.md`: real baseline + vlm_only
  (32 ảnh, 32 + 32 = 64 lần gọi Gemini, 0 fallback) và synthetic baseline offline (150 ca, 0 lần gọi).

## Tiêu chí

- [x] P0.2 bộ sinh synthetic
- [x] P0.3 manifest v2 (nhóm film grain/muted và `measurable_truth` chưa làm, ngoài tiêu chí ≥ 10 ảnh)
- [x] P0.4 scoring + CLI
- [x] Unit test offline xanh
- [x] Baseline thật + `baseline.md`
- [ ] Sai số ước lượng synthetic (σ nhiễu, σ mờ, góc sáng): chờ config `measure_only` (lane measure/agent)

## Số đo chính (real, 30 ảnh chấm)

| | baseline | vlm_only |
|---|---|---|
| Recall | 0.47 | 0.53 |
| FP rate (13 ảnh sạch/chủ ý) | 0.54 | 0.62 |
| MAE severity | 1.00 | 0.88 |

## Việc còn lại / cần lead lưu ý

- Cần review chú thích `annotated_by: "agent"` trong manifest và bảng severity ban đầu của bộ synthetic.
- Lane `calib` dùng `data/synthetic/calibration/` (tạo bằng `build_calibration_set.py`; format `params.json` đúng C5,
  thêm tên degradation `tint_green`).
- Chưa push, chưa mở PR.

Commit cuối: xem `git log` (commit docs baseline là commit cuối của lane).

LANE DONE
