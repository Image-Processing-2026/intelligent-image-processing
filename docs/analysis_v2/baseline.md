# Analysis v2: baseline của luồng chẩn đoán hiện tại

Chạy ngày 2026-10-25 bằng lane `bench` (commit code: `cc43bb0`). Model Gemini: `gemini-3.5-flash-lite`
(mặc định của `vlm_diagnostician`). Kết quả thô (`results.json`, `report.md` từng ca) nằm trong
`data/real/benchmark_results/<timestamp>-<config>-<set>/` (gitignored).

Chạy lại:

```bash
python scripts/benchmark_analysis.py --set real --config baseline
python scripts/benchmark_analysis.py --set real --config vlm_only
python scripts/benchmark_analysis.py --set synthetic --config baseline --offline
# bộ synthetic: python scripts/build_calibration_set.py --fetch-sources --out data/synthetic/calibration/
```

- `baseline` = `analyze_image` → `perceive` (VLM + defect đo được).
- `vlm_only` = baseline bỏ defect `origin="measured"`.
- `measure_only`, `fused`: chưa có (NotImplementedError có thông báo rõ).

## Số lần gọi Gemini

| Lần chạy | Ca | Gọi Gemini | Fallback | Lỗi | Giây/ca (TB) |
|---|---|---|---|---|---|
| real / baseline | 32 | 32 | 0 | 0 | 20.0 |
| real / vlm_only | 32 | 32 | 0 | 0 | 14.4 |
| synthetic / baseline (`--offline`) | 150 | 0 | – | 0 | 0.1 |

Tổng 64 lần gọi thật. Ảnh gửi VLM là ảnh benchmark đã thu về cạnh dài 1600 px (không phải bản full-res).
Chấm 30/32 ảnh thật: `backlit_04` và `color_cast_02` bị đánh dấu `ambiguous` nên không tính điểm.

## Bộ ảnh thật (32 ảnh, 30 chấm)

| Chỉ số | baseline | vlm_only | Mục tiêu khởi điểm |
|---|---|---|---|
| Precision | 0.175 | 0.235 | – |
| Recall defect | 0.467 | 0.533 | ≥ 0.85 |
| FP rate ảnh (already_good + intentional_style, n=13) | 0.538 | 0.615 | ≤ 0.10 |
| MAE severity (trên cặp đã ghép) | 1.00 | 0.88 | ≤ 0.6 |
| Độ đúng vùng (trên cặp đã ghép) | 1.00 | 1.00 | – |
| Preserve recall (21 đặc điểm) | 0.762 | 0.714 | – |
| Defect mong đợi / dự đoán | 15 / 40 | 15 / 34 | – |

Theo loại defect (tp / fp / fn, precision, recall):

| Loại | baseline | vlm_only |
|---|---|---|
| underexposed | 3 / 5 / 2, 0.38, 0.60 | 3 / 4 / 2, 0.43, 0.60 |
| backlit_subject | 0 / 10 / 1, 0.00, 0.00 | 0 / 2 / 1, 0.00, 0.00 |
| noise | 3 / 1 / 1, 0.75, 0.75 | 3 / 0 / 1, 1.00, 0.75 |
| blur | 1 / 9 / 0, 0.10, 1.00 | 1 / 12 / 0, 0.08, 1.00 |
| low_contrast | 0 / 6 / 1, 0.00, 0.00 | 0 / 6 / 1, 0.00, 0.00 |
| color_cast_cool | 0 / 1 / 2, 0.00, 0.00 | 0 / 1 / 2, 0.00, 0.00 |
| color_cast_warm | 0 / 1 / 1, 0.00, 0.00 | 1 / 1 / 0, 0.50, 1.00 |

Nhận xét:
- Phần đo được (`measured`) thêm 10 `backlit_subject` báo sai nhưng không bắt được ca ngược sáng thật,
  nên baseline có FP cao hơn `vlm_only` mà recall không tốt hơn.
- Cả hai cấu hình báo `blur` rất nhiều (9–12 FP) trên ảnh sạch hoặc ảnh có chủ ý; `low_contrast`
  cũng gần như chỉ ra FP.
- FP theo nhóm (tỉ lệ ảnh có ít nhất một FP): already_good 1.00, group_small_faces 1.00,
  intentional_style 0.40 (baseline) / 0.50 (vlm_only), multi_defect 0.50.
- Mẫu nhỏ (15 defect mong đợi): số trên dùng để so cấu hình, chưa đủ để hiệu chỉnh ngưỡng.

## Bộ synthetic (150 ảnh, baseline offline: luật dự phòng, `source=rule_based`)

Đây là đường luật (không VLM), nên chỉ phản ánh lớp luật + `measured`, không phải VLM.

| Chỉ số | Giá trị |
|---|---|
| Precision / Recall | 0.325 / 0.650 |
| FP rate ảnh trên 20 ca không có đáp án (5 clean + 15 jpeg) | 0.90 |
| MAE severity | 0.67 |
| Độ đúng vùng | 1.00 |

Recall theo họ degradation: gaussian_noise 0.96, poisson_gaussian_noise 1.00, chroma_noise 0.80,
gaussian_blur 0.87, motion_blur 0.60, contrast_compress 1.00, haze 0.80, illuminant_shift 0.47,
tint_green 0.20, gamma 0.30 (overexposed gần như không bắt được: 1/10), saturation_scale 0.00.

Sai số ước lượng (σ nhiễu, σ mờ, góc nguồn sáng): chưa tính được vì cần số đo từ lane `measure`;
hàm chấm (`estimation_summary`, `angle_error_deg`) đã có và có test, sẽ nối vào config `measure_only`.

## Hạn chế và việc nhờ review

- Severity ở bộ ảnh thật (`annotated_by: "agent"`) và ở bộ synthetic (bảng ngưỡng `_sev` trong
  `build_calibration_set.py`) là ước đoán ban đầu; P2 sẽ hiệu chỉnh, nhóm cần review.
- Ở bộ synthetic, JPEG được coi là "không có lỗi trong từ vựng" nên tính vào FP rate.
- `intentional_style` đạt 10 ảnh nhưng chưa có film grain và muted; chưa thêm `measurable_truth`
  (không có EXIF ISO đáng tin trong các ảnh hiện có). Bản full-res nhóm `noisy` đã tải về
  `data/real/benchmark_fullres/` nhưng baseline vẫn dùng bản 1600 px.
