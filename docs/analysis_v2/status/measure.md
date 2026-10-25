# Lane `measure` — trạng thái (P1: lõi đo cục bộ)

Nhánh `feat/analysis-v2-measure`, worktree `iip-wt/measure`. Brief: `docs/analysis_v2/agents/measure.md`.
Hợp đồng: `docs/analysis_v2/contracts.md` §C1.

## Đã làm

Package `src/analyzer_evaluator/measurement/` đúng hợp đồng C1. Export `Estimate`,
`MeasurementReport`, `MeasurementResult`, `RegionStats`, `measure_image`, `aggregate`, `MAP_KEYS`.
Không sửa `analyze_image` / `TechnicalMetrics`.

| Module | Nội dung |
|---|---|
| `schemas.py` | `Estimate` (từ chối NaN và confidence ngoài [0,1]), `MeasurementReport`, `RegionStats`, `MeasurementResult` |
| `tiles.py` | lưới tile (ceil), `tile_mean`, `tile_sum`, `tile_mean_at`, `aggregate` (mask mềm, trọng số theo tile) |
| `noise.py` | Liu 2013 (patch 7×7 bước 3, lặp chọn patch phẳng, σ = λ_min hiệu chỉnh Marchenko–Pastur), σ luma / Cb / Cr, σ(I) 8 bin, `sigma_dark_ratio`, Immerkær, bản đồ σ theo tile |
| `exposure.py` | L\* (float CIELAB), clip theo kênh (any/all, cao/thấp), blob cháy ≥ 0.5 % (connected components), key, bản đồ `mean_l`, `clip_high`, `clip_low` |
| `sharpness.py` | độ rộng cạnh 10–90 % dọc pháp tuyến, `edge_width_norm` (chuẩn 1000 px), defocus Zhuo–Sim, structure tensor (coherence, góc), Laplacian var, bản đồ `edge_width` và `defocus_sigma` |
| `color.py` | pixel trung tính (không cháy, L\* > 15, C\* < 12); Gray-World, White-Patch p99, Shades-of-Gray p = 6, Gray-Edge p = 6 σ = 2; góc lệch, chromaticity, độ đồng thuận; C\* float (p50, p95), `neutral_fraction`, bản đồ `chroma` |
| `compression.py` | blockiness theo biên khối 8×8 (tỉ số biên/trong, trung bình hình học hai chiều) |
| `haze.py` | dark channel (min RGB, erode 15×15), median ngoài vùng trời (nửa trên, sáng, ít màu, phẳng) |
| `_util.py` | `Section`, `make_estimate` / `put_estimate` (không ghi key khi giá trị không hữu hạn), `srgb_to_linear` |
| `__init__.py` | `measure_image` (chuẩn hóa RGB uint8, ảnh xám (H,W) / (H,W,1) lặp thành 3 kênh, cờ grayscale) |

Các commit (tăng dần, mỗi bước test xanh):

```
a799e39 docs(eval): record measure lane status for analysis v2
2ac1b5b fix(eval): chunk cv2.remap in edge-width sampling for large images
f37fa75 feat(eval): add JPEG blockiness and dark-channel haze measurements
feca772 feat(eval): add colour measurement with neutral-pixel illuminant and chroma
34ff741 feat(eval): add sharpness measurement with edge width, defocus and motion
e1efab7 feat(eval): add exposure measurement with clip, burnt blobs and key
dc21008 feat(eval): add Liu 2013 noise estimator with sigma(I), chroma and Immerkaer
951d145 feat(eval): add measurement schemas, tile grid and mask aggregate
```

Commit sau đó: `2ac1b5b` (chia khối `cv2.remap`, xem Lỗi đã sửa) và file này (`a799e39`, cập nhật thêm ở commit kế tiếp).

Test (offline, synthetic trong test): `tests/unit/test_measurement_{tiles,noise,exposure,sharpness,color,compression_haze}.py`,
tổng 70 test, tất cả xanh.

## Tiêu chí (brief + plan §1 / §0.4)

| Tiêu chí | Kết quả |
|---|---|
| σ nhiễu ∈ [2, 30], sai số ≤ 15 %, phẳng và texture | đạt; sai số đo: −0.2 … +1.1 % (phẳng và texture biên độ 25) |
| texture sạch → σ thấp | đạt: σ = 0.0 (texture biên độ 25, 40, 80) |
| mờ Gaussian tăng → độ rộng cạnh tăng đơn điệu | đạt: 2.34 / 4.00 / 6.50 / 10.17 px cho σ_mờ 0.8 / 1.5 / 2.5 / 4.0 (lý thuyết 2.56·σ) |
| nhiễu không làm "nét hơn" (độ rộng, defocus) | đạt: σ_mờ = 3, nhiễu 0 / 5 / 10 / 20 → độ rộng 7.71 / 7.95 / 7.82 / 8.00; defocus 3.05 / 2.78 / 3.04 / 2.89 |
| ánh xạ kênh biết trước trên patch xám, sai số góc ≤ 3° | đạt: gains (1.1, 1.0, 0.9), đo 4.708°, đúng 4.668°, sai số +0.04° |
| cảnh ấm có patch trung tính không ám | đạt: góc 0.0°, độ tin cậy 1.0 |
| đốm phản xạ nhỏ không báo cháy; mảng cháy lớn thì báo | đạt: đốm 25 px → `largest_clip_blob` = 0; khối 64×64 → 0.0625 |
| grayscale, (H,W,1), 32×32, toàn đen / toàn trắng không crash, không NaN | đạt (test tham số hóa) |
| `aggregate`: mask mềm, mask toàn 0 (confidence 0), mask khác shape (ValueError) | đạt |
| 12 MP < 30 s trên CPU (C1) | đạt: 8.5 s (ảnh tổng hợp 4000×3000) |
| coverage `analyzer_evaluator` ≥ 90 % | đạt: 98.9 % (toàn bộ `pytest --cov=src/analyzer_evaluator tests/unit/`) |
| `ruff check src/ tests/`, `pytest tests/unit/` | đạt: 1221 passed (toàn bộ unit), ruff sạch |

Sweep đã dùng để chọn hằng số Liu (noise.py): biên độ texture {0, 25, 40}, kích thước {128, 256, 512},
σ {2, 5, 10, 20, 30}. Sai số lớn nhất 6.7 % với `GRAD_THRESHOLD_MARGIN = 3.0`, `MIN_EFF_SAMPLES = 8·DIM`.
Với hệ số 1.5 sai số lên tới −22 %, nên không dùng.

Hiệu năng: 12 MP ≈ 8.5 s trên máy này (đo một lần, ảnh tổng hợp).

## Lệch khỏi plan (cần lead biết)

1. **Độ rộng cạnh (plan §1.3, Marziliano):** dùng khoảng cách 10–90 % biên độ dọc pháp tuyến thay vì
   cực trị. Ảnh 8-bit mờ thường đơn điệu, không có cực trị ổn định. Đơn vị và hướng như plan.
2. **Khử nhiễu trước khi đo độ nét (plan §1.3):** plan yêu cầu khử nhiễu nhẹ rồi đo. Bộ lọc song phương
   đã thử và bị loại: nó làm dốc chuyển tiếp mờ, đẩy defocus giảm khoảng 40 %. Thay bằng: khi
   σ ≥ 3 thì làm mịn Gaussian σ_s = clamp(σ/5, 1, 2), ngưỡng cạnh nằm trên mức gradient nhiễu sau làm
   mịn (K = 5), rồi trừ đúng σ_s khỏi độ rộng (√(w² − (2.56 σ_s)²)) và khỏi defocus (√(σ² − σ_s²)).
   Với nhiễu 0 thì không làm mịn, kết quả giống hệt đường không nhiễu.
3. **Xác nhận hướng mờ bằng FFT (plan §1.3)** chưa làm: chỉ có coherence và góc từ structure tensor.
4. **Tenengrad** (plan §1.3, chỉ số phụ) chưa làm; C1 chỉ yêu cầu `laplacian_var`.
5. **Zone histogram 11 vùng (plan §1.1)** chưa làm; không có key trong C1.
6. **Kiểm tra skin locus khi có khuôn mặt (plan §1.4)** chưa làm; phần này gắn với vùng khuôn mặt của lane `region`.
7. **Bản đồ tile σ:** plan gợi ý λ_min theo tile; thực tế dùng phương sai trung bình các patch phẳng trong tile
   (cần ≥ 8 patch). λ_min cần số mẫu lớn nên không dùng được ở mức tile.
8. **Vị trí blob cháy** (plan §1.1) chưa báo, chỉ có diện tích tỉ lệ `largest_clip_blob`.
9. **Ảnh nhỏ:** σ cần ít nhất 8·49 patch hiệu dụng. Ảnh 32×32 không ước lượng được σ (không có `sigma_luma`,
   report ghi chú "few flat patches for noise estimate"). Ảnh 64×64 vẫn đo nhưng kém chính xác hơn (σ = 5 đo ra 5.19).
10. **Bộ synthetic của bench:** chưa chạy. Bộ `data/synthetic/calibration` chỉ có trên nhánh `feat/analysis-v2-bench`
    (commit `c696470`, worktree `../bench`), chưa vào `feat/analysis-v2`. Điều kiện brief không đạt, nên lead chạy
    sau khi merge. Các ngưỡng synthetic hiện đã có test offline.
11. **numpy:** môi trường này có numpy 2.1.3, repo ghim `<2`. Chỉ dùng API có từ numpy 1.20 (`sliding_window_view`,
    `percentile`). Chưa chạy trên numpy 1.x.

## Lỗi đã sửa trong lane

- Canny lấy L1 nhưng ngưỡng tính theo L2: sửa bằng `L2gradient=True`.
- Ngưỡng Canny cao bằng đúng phân vị 90 nên không có cạnh nào vượt (so sánh `>`): hạ còn 0.9 lần.
- Với nhiễu, hysteresis thấp (0.8 lần) nối các pixel nhiễu thành chuỗi: đặt thấp = cao.
- `cv2.remap` giới hạn số hàng bản đồ (SHRT_MAX), lỗi khi ảnh 12 MP có hàng chục nghìn cạnh: chia khối 20 000 điểm.
- Gray-Edge: gradient quanh biên ấm/trung tính và nhiễu số ở vùng phẳng làm hướng ngẫu nhiên (40° spread);
  đã loại vùng quanh pixel không trung tính (bán kính ceil(4σ)+2) và pixel có gradient < 1e-3.
- Haze: nhánh "gần như toàn bộ là trời" không đến được vì trời chỉ nằm ở nửa trên: đã bỏ.

## Việc còn lại / cho lead

- Merge vào `feat/analysis-v2` khi review. Lane `region` và `agent` có thể code theo `measure_image` và `aggregate`
  (đã commit ở `951d145`).
- Chạy benchmark synthetic khi bench đã merge (mục 10) và ghi sai số vào file này.
- Xem lại các lệch ở trên (đặc biệt 2 và 3), vì đây là quyết định về phương pháp.
- Không có thay đổi hợp đồng: không cần `contract-changes.md`. Không đụng `analyze_image`, `TechnicalMetrics`,
  `src/agent/`, `src/region_engine/`.

Không push, không mở PR, không merge (theo brief).

LANE DONE
