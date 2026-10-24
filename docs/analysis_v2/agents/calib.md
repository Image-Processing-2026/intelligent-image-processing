# Lane `calib` — P2: hiệu chỉnh severity (wave 2)

Nhánh `feat/analysis-v2-calib`, tạo từ `feat/analysis-v2` **sau khi** `bench` và `measure` đã merge.
Đọc trước: `CLAUDE.md`, `docs/plan_analysis_v2.md` §Phase 2 và §7 D4, `contracts.md` §C1 §C4 §C5,
`docs/analysis_v2/baseline.md`, `docs/analysis_v2/status/measure.md`.

## Việc
1. Sinh bộ synthetic bằng `scripts/build_calibration_set.py`; chạy `measure_image` trên toàn bộ,
   lưu bảng đặc trưng (gitignored).
2. `scripts/calibrate_severity.py`: khớp số đo → severity 0–3 cho từng defect đo được (ngưỡng
   theo phân vị hoặc hồi quy thứ tự đơn giản; σ nhiễu vùng tối nhẹ hơn vùng sáng). Xuất
   `src/analyzer_evaluator/measurement/calibration.yaml` (version, ngày, bộ dữ liệu, ngưỡng).
3. `calibration.py` theo C4.
4. D4: `TechnicalMetrics.*_level` trong `analyzer.py` suy ra từ calibration — một nguồn ngưỡng duy
   nhất; `knowledge/engine._metric_defects` dùng lại. Giữ coverage `analyzer_evaluator` ≥ 90 %.
5. Kiểm định trên ảnh thật (không hiệu chỉnh trên ảnh thật). Chạy lại benchmark pipeline
   `--config baseline` để kiểm tra SHIP/RE_PROCESS không xấu đi; ghi so sánh vào status.

## Tiêu chí xong
MAE severity trên synthetic ≤ 0.4; FP rate trên `already_good` không tăng so với baseline;
`docs/analysis_v2/status/calib.md` kết thúc bằng `LANE DONE`.
