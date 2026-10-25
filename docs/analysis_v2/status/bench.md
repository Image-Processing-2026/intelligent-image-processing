# Lane bench — trạng thái

Nhánh: `feat/analysis-v2-bench` · Brief: `docs/analysis_v2/agents/bench.md` · Phase: P0.2–0.4

## Đang làm

- Khởi động lane. Đã đọc brief, `orchestration.md`, `contracts.md` §C5, `plan_analysis_v2.md` §Phase 0.
- Kế hoạch: (1) scoring thuần + unit test, (2) bộ synthetic + test, (3) manifest schema v2 + ảnh
  intentional_style + fullres nhóm noisy, (4) CLI `benchmark_analysis.py`, (5) chạy baseline thật
  và commit `docs/analysis_v2/baseline.md`.

## Tiêu chí

- [ ] P0.2 `scripts/build_calibration_set.py`
- [ ] P0.3 manifest schema v2 (`severity`, `region`, nhóm `intentional_style` ≥ 10 ảnh, fullres `noisy`)
- [ ] P0.4 `scripts/benchmark_scoring.py` + `scripts/benchmark_analysis.py`
- [ ] Unit test offline (`test_benchmark_scoring.py`, test bộ synthetic)
- [ ] Baseline thật: `--set real --config baseline`, `--config vlm_only`, `--set synthetic --config baseline`
- [ ] `docs/analysis_v2/baseline.md` đã commit

## Blocked

(không có)
