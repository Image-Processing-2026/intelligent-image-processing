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

## Thêm từ review wave 1 (lead, 2026-10-25)
Đọc `docs/analysis_v2/measurement_synthetic.md` trước. Bộ synthetic đã sinh sẵn ở
`data/synthetic/calibration/` trong worktree tích hợp; sai số thô: `python scripts/benchmark_measurement.py`.
Config `measure_only` trong `scripts/benchmark_analysis.py` đã nối sẵn, tự chạy khi có `calibration.py`.

6. Severity mờ dùng `defocus_sigma` (lệch thấp 13–21 % có hệ thống, hiệu chỉnh ở đây);
   `edge_width_*` chỉ dùng tương đối, không làm ngưỡng tuyệt đối.
7. σ trong `params.json` là σ mỗi kênh RGB; σ luma = 0.669·σ. Quy đổi trước khi khớp.
8. Mờ chuyển động: phạm vi mở rộng sang phần motion của `measurement/sharpness.py`. Thêm xác nhận
   hướng mờ (dị hướng defocus theo hướng pháp tuyến cạnh, hoặc FFT như plan §1.3). Ảnh sạch có
   texture (cỏ, phong cảnh) có `motion_coherence` 0.75 nên coherence một mình không đủ. Chưa có
   xác nhận thì `severity_from_measurement("motion_blur", …)` trả `(0, 0.0)`.
9. Ám màu mạnh bị đo thiếu (3000 K: kỳ vọng ~21°, đo 5–13°). Phạm vi mở rộng sang
   `measurement/color.py`: chọn pixel trung tính lặp (ước lượng sơ bộ, bù, rồi mới lọc `C*`).
   Giữ test `test_measurement_color.py` xanh.

## Tiêu chí xong
MAE severity trên synthetic ≤ 0.4; FP rate trên `already_good` không tăng so với baseline;
`docs/analysis_v2/status/calib.md` kết thúc bằng `LANE DONE`.
