# Lane `measure` — P1: lõi đo cục bộ (Module 1)

Nhánh `feat/analysis-v2-measure`. Đọc trước: `CLAUDE.md`, `docs/plan_analysis_v2.md` §Phase 1,
`docs/analysis_v2/contracts.md` §C1, `docs/analysis_v2/orchestration.md` (quy tắc chung).

## Mục tiêu
Package `src/analyzer_evaluator/measurement/` đúng hợp đồng C1: đo cục bộ, có độ tin cậy, trên ảnh
độ phân giải gốc. Không sửa `analyze_image`/`TechnicalMetrics` (lane `calib` làm ở wave 2).

## Thứ tự (commit sau mỗi bước, test xanh)
1. `schemas.py` + `tiles.py` (`aggregate`, `MAP_KEYS`) + `__init__.py` với `measure_image` gọi các
   module con. **Commit bước này sớm** — lane `region`/`agent` code theo nó.
2. `noise.py`: Liu–Tanaka–Okutomi 2013 (patch 7×7, stride 3, chọn lặp patch yếu texture theo
   ngưỡng trace hiệp phương sai gradient, σ² = trị riêng nhỏ nhất); σ(I) theo 8 bin; luma + Cb + Cr;
   giữ Immerkær. Chỉ NumPy/OpenCV/scikit-image.
3. `exposure.py`: clip theo kênh, connected components (blob ≥ 0.5 % = vùng cháy), L\*, key.
4. `sharpness.py`: khử nhiễu nhẹ theo σ đo được (chỉ để đo) → độ rộng cạnh (Marziliano) →
   defocus map Zhuo–Sim (σ₀ = 1) → motion coherence (structure tensor) + góc.
5. `color.py`: pixel tin được (không clip, L\* > 15, chroma thấp); Gray-World, White-Patch p99,
   Shades-of-Gray p=6, Gray-Edge; góc lệch, đồng thuận; C\* trên CIELAB float.
6. `compression.py` (blockiness 8×8), `haze.py` (dark channel ngoài vùng trời ước lượng).

## Test (`tests/unit/test_measurement_*.py`, offline, tạo ảnh synthetic ngay trong test)
- σ nhiễu biết trước ∈ [2, 30] trên ảnh phẳng + texture → sai số ≤ 15 %; texture sạch → σ thấp.
- Mờ Gaussian tăng → độ rộng cạnh tăng đơn điệu; nhiễu không làm "nét hơn".
- Nhân kênh biết trước trên patch xám → sai số góc ≤ 3°; cảnh ấm có patch trung tính → không ám.
- Đốm phản xạ nhỏ không bị báo cháy; mảng cháy lớn thì có.
- Grayscale, (H, W, 1), ảnh nhỏ (32×32), ảnh toàn đen/toàn trắng không crash, không NaN trong report.
- `aggregate` với mask mềm, mask toàn 0 (confidence 0), mask khác shape (raise ValueError).
- Coverage: `pytest --cov=src/analyzer_evaluator tests/unit/` ≥ 90 %.

## Tiêu chí xong
Ngưỡng sai số trên đạt; nếu lane `bench` đã có bộ synthetic trên nhánh tích hợp, chạy thêm và ghi
sai số vào `docs/analysis_v2/status/measure.md`. Kết thúc file status bằng `LANE DONE`.
