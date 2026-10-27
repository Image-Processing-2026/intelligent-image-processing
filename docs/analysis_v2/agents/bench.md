# Lane `bench` — P0.2–0.4: benchmark độ chính xác chẩn đoán

Nhánh `feat/analysis-v2-bench`. Đọc trước: `CLAUDE.md`, `docs/plan_analysis_v2.md` §Phase 0,
`docs/analysis_v2/contracts.md` §C5, `docs/analysis_v2/orchestration.md` (quy tắc chung).

## Mục tiêu
Có thước đo cho mọi phase sau và **baseline của luồng hiện tại** trước khi ai sửa logic chẩn đoán.

## Việc
1. **0.2** `scripts/build_calibration_set.py` theo C5: ảnh sạch từ `data/real/benchmark/already_good/`
   (thêm ảnh sạch khác nếu cần đa dạng; tải từ Wikimedia Commons với giấy phép ghi trong manifest
   riêng `data/calibration_sources.json`). Mỗi loại degradation trong plan, tham số biết trước,
   seed cố định, `params.json` cạnh ảnh. Nhiễu Poisson-Gaussian phụ thuộc độ sáng; JPEG qua Pillow;
   haze theo `I = J·t + A(1−t)`. Có cả tổ hợp hai lỗi.
2. **0.3** Mở rộng `data/real_benchmark_manifest.json` (tăng `schema_version`): `severity` (0–3) và
   `region` cho mỗi defect mong đợi. Bổ sung nhóm `intentional_style` (đêm, low-key, silhouette,
   bokeh, hoàng hôn, film grain, muted) để ≥ 10 ảnh; cập nhật `data/download_real_benchmark.py`.
   Với nhóm `noisy`, thêm trường cho bản gốc full-res (tải thêm vào `data/real/benchmark_fullres/`).
   Đánh dấu rõ chú thích nào bạn tự đặt (`"annotated_by": "agent"`) để nhóm review.
3. **0.4** `scripts/benchmark_scoring.py` (hàm thuần) + `scripts/benchmark_analysis.py` (CLI theo C5).
   Tái sử dụng phần chấm của `scripts/benchmark_brain.py` nếu có. Config `baseline` = luồng
   `perception.perceive` hiện tại; `vlm_only` = perceive bỏ measured defects.
4. Unit test offline `tests/unit/test_benchmark_scoring.py` (precision/recall, FP rate, MAE,
   region match, preserve recall, sai số synthetic) và test cho bộ sinh synthetic (tham số → ảnh
   tất định, đúng shape/dtype).
5. Chạy **baseline thật**: `--set real --config baseline` và `--config vlm_only` (gọi Gemini thật;
   26+ ảnh, ghi số lần gọi), `--set synthetic --config baseline` (offline được). Ghi bảng số vào
   `docs/analysis_v2/baseline.md` (commit), kết quả thô để trong thư mục gitignored.

## Tiêu chí xong
Script chạy lại được bằng một lệnh; baseline commit trong `docs/analysis_v2/baseline.md`; unit test
xanh; `docs/analysis_v2/status/bench.md` kết thúc bằng `LANE DONE`.

## Lưu ý
- Lane `calib` (wave 2) dùng bộ synthetic của bạn: giữ đúng format `params.json` trong C5.
- `measure_only`/`fused` để `NotImplementedError` có thông báo rõ.
- Ảnh và kết quả benchmark không commit (`data/real/*` đã gitignored; thêm ignore cho
  `data/synthetic/calibration/` nếu cần).
