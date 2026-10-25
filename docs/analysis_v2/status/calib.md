# Lane `calib` — trạng thái (P2: hiệu chỉnh severity)

Nhánh `feat/analysis-v2-calib`, worktree `iip-wt/calib`. Brief: `docs/analysis_v2/agents/calib.md`.
Hợp đồng: `docs/analysis_v2/contracts.md` §C4. Không gọi Gemini (lane `calib` không nằm trong danh
sách lane được gọi thật, `orchestration.md` quy tắc 4).

## Đã làm

| Việc (brief) | Kết quả |
|---|---|
| 1. Bộ synthetic + bảng đặc trưng | 150 ảnh copy từ `../integration/data/synthetic/calibration/` (cùng seed 7). `calibrate_severity.py` đo `measure_image` toàn bộ, cache `data/synthetic/calibration/features.json` (gitignored) kèm dấu vân tay mã đo (đổi mã đo → tự đo lại). |
| 2. `scripts/calibrate_severity.py` → `calibration.yaml` | Hồi quy thứ tự đơn giản: 3 ngưỡng cực tiểu Σ\|severity − nhãn\| trên mọi ca (vét cạn, hòa → khe rộng nhất). Kiểm chéo bỏ-một-nguồn (5 ảnh nguồn): chỉ giữ lỗi có MAE cân bằng kiểm chéo thấp hơn bộ dự đoán rỗng; `reliability` = 1 − MAE_cv/MAE_null hạ confidence lỗi khớp yếu. YAML có version `p2-20261025`, ngày, bộ dữ liệu, phương pháp, ngưỡng, chỉ số khớp. |
| 3. `measurement/calibration.py` (C4) | `severity_from_measurement`, `calibration_version`, thêm `defect_severities`, `severity_from_calibration` (script dùng chung hàm điểm). Lỗi không đo được / loại lạ → `(0, 0.0)`. |
| 4. D4 | `analyze_image` đo cục bộ rồi suy `*_level` từ severity hiệu chỉnh; `TechnicalMetrics` thêm `defect_severity`, `calibration_version` (có default); `knowledge/engine._metric_defects` dùng `defect_severity` (giữ nhánh level cho dict cũ). Bỏ ngưỡng tay, `_classify_noise`, `_detect_color_cast`. Cache severity theo băm nội dung ảnh (pipeline gọi `analyze_image` 2–3 lần/vòng trên cùng ảnh). |
| 5. Kiểm định ảnh thật, benchmark pipeline | Xem "Số đo". Không hiệu chỉnh trên ảnh thật. |
| 6. Mờ: `defocus_sigma` | Bù lệch thấp khớp trên ca mờ Gaussian: `defocus_bias` = 0.1415 (đo ≈ 0.86·thật). Quy về cạnh dài 1024 px (không phóng to ảnh nhỏ). `edge_width_*` không dùng làm ngưỡng. |
| 7. σ RGB → luma | Điểm nhiễu ở thang σ luma; nhãn của bench theo σ RGB (4, 12) ≙ σ luma (2.68, 8.03). Ngưỡng khớp 2.86 / 8.996 khớp với quy đổi lý thuyết. Nhiễu chroma quy về thang luma ×0.669. Nhiễu vùng tối: chia 1 + 0.5·(σ_tối/σ_sáng − 1) (prior cảm nhận, synthetic không có nhãn để khớp). |
| 8. Mờ chuyển động | `sharpness.py`: σ defocus theo 12 hướng pháp tuyến cạnh → `motion_anisotropy`, `motion_extent_sigma`, `motion_blur_angle_deg` (key mới). Cổng xác nhận khớp biên lớn nhất (anisotropy ≥ 1.282, extent ≥ 0.726 px; ảnh nhiễu σ ≥ 3 cần thêm 0.2): 10/10 ca mờ chuyển động qua, 0/140 ca khác lọt. Ảnh sạch cỏ/phong cảnh (coherence 0.75) không qua. `blur` = max(defocus, motion_blur đã xác nhận). |
| 9. Ám màu mạnh | `color.py`: chọn pixel trung tính lặp (ước lượng trên C* < 12, bù von Kries, lọc lại C* < 12 trên ảnh đã bù, tối đa 6 vòng, dừng khi đổi < 0.25°), chỉ nhận lại pixel C* gốc < 30 để màu bão hòa (cỏ) không trôi vào. `test_measurement_color.py` vẫn xanh (cảnh ấm có patch xám vẫn < 1°). |

Commit: `dd9c6c8` (màu), `0b225bf` (mờ chuyển động + calibration C4), `198bc7c` (D4), commit docs cuối.

## Số đo

### Lõi đo (`scripts/benchmark_measurement.py`, synthetic)

| Đại lượng | Trước (`measurement_synthetic.md`) | Sau |
|---|---|---|
| Vector nguồn sáng, MAE | 5.5° (39 % ≤ 3°) | **2.0°** (72 % ≤ 3°); 3000 K đo 13–22° (kỳ vọng 18–25°) |
| Hướng mờ chuyển động (`motion_blur_angle_deg`) | – (structure tensor: 10.9°, sai trên texture có hướng) | 8/10 ≤ 15.4°; sai ở `clean_grass_01` (40°, 63°: cạnh cỏ thiếu hướng) |
| σ nhiễu, σ defocus, độ rộng cạnh | không đổi (không sửa) | |

### Hiệu chỉnh (`calibration.yaml` p2-20261025)

| Lỗi | Ngưỡng (điểm) | reliability | Ghi chú |
|---|---|---|---|
| noise | σ luma 2.86 / 2.86 / 9.00 | 0.72 | không có dải mức 1 (xem Hạn chế 1) |
| blur | σ defocus (bù, 1024 px) 1.22 / 1.22 / 2.32 | 0.60 | + combine_max motion_blur |
| motion_blur | extent sau cổng: −∞ / 1.48 / 1.48 | 0.70 | qua cổng ⇒ ≥ 1 |
| underexposed | 50 − L\* ≥ 16.3 (mức 2) / 28.9 (mức 3) | 0.11 | synthetic không có nhãn mức 1 |
| overexposed | L\* − 50 ≥ 17.4 / 21.0 / 21.0 | 0.34 | |
| low_contrast | 100 − (p95 − p5) ≥ 74.7 (mức 2) | 0.03 | yếu: chỉ dải động rất hẹp |
| color_cast_warm | ln(r/b) 0.35 / 0.80 / 0.80 | 0.49 | |
| color_cast_cool | ln(b/r) 0.49 | 0.16 | |
| oversaturated | C\* p50 ≥ 65.9 | 0.04 | gần như chỉ ảnh rất bão hòa |
| haze (P6) | dark channel ≥ 0.449 (mức 2) | 0.16 | |
| jpeg_artifacts (P6) | blockiness 1.45 / 1.53 | 0.52 | nhãn q ≤ 25 → 2, ≤ 50 → 1 |
| highlight_clipping (P6) | blob 0.5 % / 3 % / 10 % | prior | không có nhãn; confidence × 0.5 |
| color_cast_green, undersaturated, backlit_subject | – | – | không đo được → `(0, 0.0)` |

### Tiêu chí

| Tiêu chí | Kết quả |
|---|---|
| MAE severity synthetic ≤ 0.4 | **đạt: 0.14** (`benchmark_analysis --set synthetic --config measure_only`, cặp đã ghép); 0.17 trong báo cáo của script (tính cả P6) |
| FP rate `already_good` không tăng | **đạt**: thật `measure_only` 0.00 (baseline 1.00); synthetic ảnh sạch + JPEG 0.05 (baseline offline 0.90) |
| Coverage `analyzer_evaluator` ≥ 90 % | đạt: 97.6 % |
| `ruff check src/ tests/`, `pytest tests/unit/` | đạt: 1344 passed |

### Benchmark chẩn đoán

| Bộ / config | Precision | Recall | FP ảnh sạch | MAE | Ghi chú |
|---|---|---|---|---|---|
| synthetic / baseline `--offline`, trước D4 (base `4571146`) | 0.33 | 0.65 | 0.90 (n=20) | 0.67 | khớp `baseline.md` |
| synthetic / baseline `--offline`, sau D4 | **0.84** | 0.64 | **0.05** | **0.14** | luật dùng severity hiệu chỉnh |
| synthetic / measure_only | 0.86 | 0.64 | 0.05 | 0.14 | 0 lần gọi Gemini |
| real / measure_only | 0.21 | 0.33 | 0.77 (n=13) | 0.80 | already_good 0.00, group_small_faces 0.00, intentional_style 1.00 |

Real `measure_only`: FP trên ảnh sạch chỉ còn ở `intentional_style` (low-key → underexposed,
hoàng hôn → warm, high-key → overexposed). Số đo không biết chủ ý; đó là việc của VLM/preserve
trong `fusion` (bảng P6: severity ≥ 1 + VLM preserve → preserved).

### Pipeline SHIP/RE_PROCESS (D4), offline, 32 ảnh thật

`run_pipeline(max_iterations=3)` không key (luật dự phòng), base `4571146` so với nhánh này:

| | Trước | Sau |
|---|---|---|
| Quyết định cuối | SHIP 25, STOP_BEST_EFFORT 7 | SHIP 26, STOP_BEST_EFFORT 6 |
| Ảnh sạch/chủ ý có plan rỗng (SHIP vòng 1) | 4 / 13 | 5 / 13 |
| Ảnh có lỗi được xử lý | 8 | 8 |
| Tổng số vòng / rollback | 60 / 2 | 56 / 1 |
| Thời gian (32 ảnh) | 50 s | 174 s |

Đổi theo ảnh: hết `blur` báo sai (backlit_02, multi_defect_01, intentional_02/04/09), hết denoise
thừa ở `already_good_02`, `intentional_07`; mất denoise ở `noisy_01` (σ 1.42) và `noisy_03` (σ 2.85,
sát ngưỡng 2.86). Không xấu đi về SHIP/RE_PROCESS. **Chưa chạy** `--set real --config baseline` với
VLM (lane không được gọi Gemini): lead chạy sau merge (32 lần gọi) cho checkpoint FP.

## Hạn chế / việc cho lead

1. **Ngưỡng nhiễu mức 1 bị nuốt bởi `clean_grass_01`:** Liu đo 3.7 trên cỏ sạch (texture), nên bộ
   khớp chọn bỏ dải mức 1 (σ RGB 2) thay vì báo nhiễu trên mọi biến thể cỏ. Ảnh thật nhóm `noisy`
   ở 1600 px có σ 1.4–4.1 → 2/3 dưới ngưỡng. Cần bộ ước lượng chống texture (hoặc key độ đồng đều
   σ giữa tile phẳng) ở lane `measure` rồi hiệu chỉnh lại.
2. **Nhiễu camera thật tương quan không gian:** bản full-res nhóm `noisy` đo σ 0.4–1.7, *thấp hơn*
   bản 1600 px (1.4–4.1). Demosaic/NR làm nhiễu tương quan; patch 7×7 đo thiếu cho tới khi thu nhỏ.
   Đề xuất (`measure`/`fusion`): đo nhiễu thêm trên bản thu nhỏ ~1600 px hoặc đa tỉ lệ.
3. **Nhãn synthetic (bảng `_sev` của bench):** `underexposed` không có ca mức 1; tint xanh 1.12
   quá nhẹ (chỉ mức 1) và CCT cũng dịch trục xanh nên `color_cast_green` không khớp được; bão hòa
   tuyệt đối phụ thuộc nội dung. Lỗi phơi sáng/tương phản/bão hòa có `reliability` thấp: số đo
   tuyệt đối không biết độ sáng chủ ý của cảnh. Muốn tốt hơn cần thêm ảnh nguồn đa dạng hoặc nhãn
   tương đối.
4. **Hiệu năng:** `analyze_image` 1080p ảnh nhiễu ngẫu nhiên 1.5 s không cache (trước 0.03 s); test hiệu năng nâng
   ngưỡng 1 s → 5 s (plan chấp nhận đo nặng hơn). Pipeline offline chậm ~3.5×.
5. **Hợp đồng:** `contract-changes.md` mục 5 (`long_side` tùy chọn cho C4, chờ quyết định) và
   mục 6 (key motion mới, `defect_severity` vào prompt VLM, ngưỡng sáng còn lại trong `perception`).
6. Ảnh benchmark thật: worktree thiếu 6 ảnh `intentional_style` và `benchmark_fullres`, đã copy từ
   `../bench` (gitignored).
7. Person 1 cần review `analyzer.py`, `measurement/{color,sharpness,calibration}.py`.

Không push, không mở PR, không merge.

LANE DONE
