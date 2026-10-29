# Lane `color` — đo ám màu trên ảnh thật (sau checkpoint fusion)

Nhánh `feat/analysis-v2-color`, tạo từ `feat/analysis-v2` (sau `562217e`). Một lane, chạy song song
với `fusion` (chưa merge). Người dùng duyệt mở lane qua reporter (2026-10-26).
Đọc trước: `CLAUDE.md`, `contracts.md` §C1 §C4, `docs/analysis_v2/measurement_synthetic.md`,
`docs/analysis_v2/status/calib.md`, `docs/analysis_v2/checkpoint_calib.md`, và mục "Ám màu đo ≈ 0
trên ảnh thật" trong `git show feat/analysis-v2-fusion:docs/analysis_v2/status/fusion.md`.

## Luật cứng
- **Không đọc, chạy, chấm hay chỉnh trên bộ hold-out.** Không mở `data/holdout_manifest.json`,
  `docs/analysis_v2/holdout.md` hay `data/real/holdout/`. Hold-out chỉ dùng để chấm kết quả cuối,
  do lead chạy.
- Chỉ hiệu chỉnh trên bộ synthetic (`data/synthetic/calibration/`, sinh bằng
  `scripts/build_calibration_set.py`) và bộ dev thật (`data/real/benchmark/`,
  `data/real_benchmark_manifest.json`). Bộ dev có ít nhãn ám màu (color_cast_01 cool 1,
  color_cast_02 ambiguous, color_cast_03 warm 3, multi_defect_01 cool 1). Báo số ảnh dương tính
  trong mọi kết luận. Cần thêm nhãn ám màu thật thì hỏi reporter (`herdr agent prompt wA:p1 "..."`),
  người dùng sẽ duyệt ảnh qua trang review; không tự đặt nhãn làm đáp án.
- Không gọi Gemini (không chạy config `vlm_*`, `fused*`, `baseline`). `measure_only` và
  `scripts/benchmark_measurement.py` chạy offline, được phép trên synthetic và dev.
- Ảnh hoàng hôn chủ ý (`intentional_style_05` ln(b/r) −1.00, `intentional_style_09` −0.72) và mọi
  ảnh `already_good` / `intentional_style` không được thành FP ám màu ở `measure_only`. Ghi FP sạch
  trước/sau (số đếm / 13).
- Không đổi `numpy<2`. Docstring/comment tiếng Việt; log, exception, commit tiếng Anh.

## Việc
1. **Không có pixel trung tính** (`color_cast_03`: 0 % pixel C* < 12, 1.1 % C* < 30): `_illuminant`
   trả không đo được. Thêm đường dự phòng trong `measurement/color.py`, theo thứ tự:
   (a) nới cổng C* từng bậc (ví dụ 12 → 18 → 24 → 30) tới khi đủ `MIN_NEUTRAL_PIXELS`;
   (b) không được thì ước lượng từ mọi pixel không cháy (Shades-of-Gray / Gray-Edge đã có trong
   `_illuminant_estimates`).
   Confidence phải phản ánh đường đã dùng (thấp hơn rõ với (b)) để fusion biết cần verify. Ghi
   đường đã dùng vào `Section` (ví dụ ước lượng `illuminant_gate_chroma` hoặc `illuminant_method`);
   khóa mới thì thêm vào `contract-changes.md` (mục mới) cho lead duyệt.
2. **Ngưỡng ám nhẹ**: cool mức 1 `ln(b/r) ≥ 0.49` khớp trên 5 ca synthetic (reliability 0.16); ảnh
   thật mức 1 đo 2–3°. Trước khi hạ ngưỡng, đo phân bố `ln(b/r)`, `ln(r/b)`, trục green và góc
   nguồn sáng trên **mọi** ảnh sạch của bộ dev và synthetic `__clean`, so với các ca ám nhẹ. Hạ
   ngưỡng chỉ khi tách được (ghi bảng phân bố). Không tách được thì ghi rõ "mức 1 không đo được
   bằng nguồn sáng, để VLM + verify" và giữ ngưỡng: đây là kết quả hợp lệ.
   Được thêm mức ám nhẹ vào `build_calibration_set.py` (ví dụ 5000 K, 6500 K→7000 K, tint 1.05)
   nhưng không đổi các mức đang có (giữ so sánh với số cũ). Chạy lại `scripts/calibrate_severity.py`,
   tăng `version` của `calibration.yaml`.
3. Test offline: không có pixel trung tính → có ước lượng với confidence thấp; hoàng hôn ảnh tổng
   hợp không thành ám lạnh; giữ `test_measurement_color.py`, test calibration xanh; coverage
   `analyzer_evaluator` ≥ 90 %.

## Phạm vi file
`src/analyzer_evaluator/measurement/color.py`, `measurement/calibration.py` (chỉ hàm `_score_cast_*`
và `_illuminant`), `measurement/calibration.yaml` (mục `color_cast_*`, `version`, ngày),
`scripts/calibrate_severity.py`, `scripts/build_calibration_set.py` (chỉ thêm mức ám), test tương
ứng trong `tests/unit/`. Sửa ngoài phạm vi: ghi `contract-changes.md` trước.

## Tiêu chí xong
- `color_cast_03` có ước lượng (hướng ấm) và severity ≥ 1 ở `measure_only`.
- FP ám màu trên 13 ảnh sạch dev không tăng; hai ảnh hoàng hôn không báo ám lạnh.
- Bảng phân bố và quyết định ngưỡng ở mục 2, kèm số ca dương tính.
- `ruff check src/ tests/`, `pytest tests/unit/` xanh.
- `docs/analysis_v2/status/color.md` kết thúc bằng `LANE DONE` (hoặc `BLOCKED: <lý do>`).
