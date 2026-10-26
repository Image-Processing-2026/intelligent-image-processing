# Lane `color` — trạng thái (đo ám màu trên ảnh thật, P2 bổ sung)

Nhánh `feat/analysis-v2-color` (worktree `iip-wt/color`), tạo từ `feat/analysis-v2` @ `69c23f0`.
Brief: `docs/analysis_v2/agents/color.md`. Hợp đồng: `contracts.md` §C1 §C4.

Luật đã giữ: không mở hold-out (`data/holdout_manifest.json`, `holdout.md`, `data/real/holdout/`),
0 lời gọi Gemini (chỉ `measure_only`, `benchmark_measurement.py`, `calibrate_severity.py`), không
đổi `numpy<2`. Dữ liệu: lead copy bộ synthetic (150 ảnh) và `intentional_style_05..10`; ảnh nguồn
`clean_grass_01`, `clean_landscape_01` copy từ `../integration/data/real/calibration_sources/`
(gitignored) để sinh thêm mức ám nhẹ. 150 ảnh synthetic cũ được sinh lại giống hệt từng byte (md5).

## Đã làm

| Việc (brief) | Kết quả |
|---|---|
| 1. Không có pixel trung tính | `measurement/color.py::_neutral_set`: cổng C\* < 12 thiếu `MIN_NEUTRAL_PIXELS` (100) → (a) nới 18 → 24 → 30; vẫn thiếu → (b) mọi pixel không cháy, chỉ Shades-of-Gray + Gray-Edge (bỏ Gray-World, White-Patch vì bị màu cảnh kéo mạnh nhất), không lặp chọn trung tính. Đường (a) vẫn lặp bù–lọc lại với cổng đã nới. Confidence: (a) × 12/cổng, (b) × 0.2. Cả hai cần ảnh có cạnh (≥ 100 pixel có gradient): một mặt màu phẳng không phân biệt được màu vật với màu nguồn sáng nên vẫn không ước lượng (giữ `test_uniform_coloured_image_is_not_a_cast`). Key mới `illuminant_method` (0/1/2), `illuminant_gate_chroma`: `contract-changes.md` mục 9. |
| 2. Ngưỡng ám nhẹ | Bảng phân bố bên dưới. **Không tách được → giữ ngưỡng** (cool 0.4856, warm 0.3466 / 0.8012). Thêm 5000 K, 7000 K vào `build_calibration_set.py` (mức cũ không đổi). `calibrate_severity.py`: hai mức này là ca **dò**, không khớp, báo riêng trong `report.md` (`PROBE_CCT`). Chạy lại: `calibration.yaml` `p2-20261026`. |
| 3. Test offline | `test_measurement_color_fallback.py` (8): cổng nới 18/24 đúng góc ≤ 3° với confidence ≤ 12/cổng, đường mọi pixel (C\* > 30 ở mọi pixel) đúng góc, confidence ≤ 0.2, ấm không lạnh; fallback ≤ 0.5 × confidence đường chuẩn; hoàng hôn tổng hợp không thành ám lạnh; mặt màu phẳng không ước lượng. `test_build_calibration_set.py` (nhãn 5000 K/7000 K), `test_calibrate_severity.py` (ca dò, mục báo cáo). |

### `calibration.yaml` p2-20261025 → p2-20261026
Chỉ đổi `version`, `date`, `measurement_fingerprint` và số khớp của `color_cast_*`. **Mọi ngưỡng giữ
nguyên**, kể cả các lỗi khác (đưa ca dò vào khớp từng làm đổi ngưỡng noise 2.86/2.86/9.00 →
2.86/4.22/9.00 và hạ cool về 0.377, nên đã tách ca dò ra).

| | warm | cool | green |
|---|---|---|---|
| ngưỡng | 0.3466 / 0.8012 / 0.8012 (giữ) | 0.4856 (giữ) | không đo được (giữ) |
| ca dương tính (khớp) | 10 | 5 | 5 |
| LOSO MAE cân bằng (rỗng) | 0.514 → **0.364** (1.0) | 0.421 → 0.455 (0.5) | 0.586 → 0.500 (0.5) |
| reliability | 0.486 → **0.636** | 0.159 → **0.090** | – |

Warm khớp tốt hơn vì `clean_grass_01__cct_3000` giờ có ước lượng (warm 3, đúng nhãn). Cool kém đi
một chút: biến thể cỏ giờ có ước lượng (trước đó bỏ qua) nhưng ở thang tuyệt đối lệch xanh lá
(trục green +0.47 trên ảnh cỏ sạch), nên kiểm chéo bỏ nguồn cỏ tệ hơn.

## Số đo

### Mục 1: ảnh không có pixel trung tính

| Ảnh | Trước | Sau |
|---|---|---|
| color_cast_03 (warm 3; 0 % C\* < 12, 1.1 % < 30) | không ước lượng → `(0, 0.0)` | đường (a) cổng 24: góc 38.7°, ln(r/b) +3.01 → **warm 3**, confidence severity 0.375 (ảnh chuẩn như `intentional_style_05`: 0.546) |
| intentional_style_08 (2 pixel C\* < 12 dùng được) | ước lượng từ 2 pixel → lặp, 19.4°, conf 0.99 | đường (a) cổng 18: 18.8°, conf 0.51 (cùng hướng, confidence hợp lý hơn) |
| synthetic, 160 ảnh | `clean_grass_01` ×5 không ước lượng | có ước lượng; đổi severity duy nhất: `clean_grass_01__cct_3000` → warm 3 (đúng nhãn) |

So sánh từng ảnh mã cũ/mới (160 synthetic + 32 dev, ngưỡng hiện hành): chỉ 6 ảnh đổi, đều là
"không ước lượng" → có ước lượng (color_cast_03 warm 3, grass cct_3000 warm 3, 4 biến thể cỏ 0).
**Không thêm FP ám màu nào.**

`benchmark_measurement.py` (vector nguồn sáng so với `__clean` cùng nguồn): 20 ca cũ MAE 2.29°,
median 1.57°, 65 % ≤ 3°, **0 thiếu** (trước: 18 ca có số, MAE 2.0°, 72 % ≤ 3°, 2 thiếu là
`clean_grass_01` 3000 K và tint; hai ca này giờ có số nhưng sai hơn trung bình). 10 ca dò: MAE 1.05°,
80 % ≤ 3°.

### `measure_only` trên bộ dev (30 ảnh chấm, 13 sạch; 0 lời gọi Gemini)

| | Trước | Sau |
|---|---|---|
| **Ảnh sạch có FP ám màu (/13)** | **4** | **4** (không tăng) |
| — đó là | is_01 warm 1, is_05 warm 3, is_08 cool 1, is_09 warm 1 | giống hệt |
| Hoàng hôn is_05 / is_09 | warm 3 / warm 1, **không cool** | giống hệt |
| FP ám màu trên ảnh không nhãn ám khác | low_light_02 warm 1, noisy_03 warm 1 | giống hệt |
| color_cast_warm tp / fp / fn | 0 / 5 / 1 | **1 / 5 / 0** |
| color_cast_cool tp / fp / fn | 0 / 1 / 2 | 0 / 1 / 2 |
| Recall toàn bộ (15) | 5 (0.33) | **6 (0.40)** |
| MAE severity (cặp khớp) | 0.80 (5 cặp) | **0.67** (6 cặp; color_cast_03 khớp đúng mức 3) |
| FP ảnh sạch, mọi loại (/13) | 10 | 10 (không đổi, là low-key/high-key/hoàng hôn, việc của preserve) |

FP cool trên `intentional_style_08` (low-key xanh, ln(b/r) +0.56) có từ trước, không do lane này.

### `measure_only` synthetic

160 ảnh (gồm 10 ca dò): precision 0.852, recall 0.613, FP ảnh sạch 1/20, MAE 0.13. Bỏ 10 ca dò
(150 ảnh như trước): precision 0.865, recall 0.643, giống số của `calib` (0.86 / 0.64 / 0.05 / 0.14).
Ca dò: 5000 K warm bắt 2/5 (ag_01, cỏ), 7000 K cool bắt **0/5**.

## Mục 2: phân bố trục màu và quyết định ngưỡng

ln(b/r) = −ln(r/b); green = ln g − (ln r + ln b)/2 (cùng hàm điểm của `calibration.py`). Số trong ô:
min / median / max. Đo sau khi có đường dự phòng.

| Nhóm | n | ln(b/r) | ln(r/b) | trục green | góc (°) |
|---|---|---|---|---|---|
| synthetic `__clean` | 5 | −0.21 / +0.02 / +0.20 | −0.20 / −0.02 / +0.21 | −0.05 / −0.03 / +0.47 | 0.8 / 4.8 / 13.8 |
| dev sạch (already_good + intentional_style) | 13 | −1.00 / +0.00 / +0.56 | −0.56 / −0.00 / +1.00 | −0.48 / −0.01 / +0.22 | 0.0 / 5.1 / 21.6 |
| dev sạch, bỏ 2 hoàng hôn | 11 | −0.43 / +0.00 / +0.56 | −0.56 / −0.00 / +0.43 | −0.48 / −0.01 / +0.07 | 0.0 / 1.9 / 18.8 |
| dev không nhãn ám, nhóm khác | 16 | −0.39 / −0.08 / +0.24 | −0.24 / +0.08 / +0.39 | −0.17 / −0.03 / +0.15 | 0.3 / 4.4 / 9.2 |
| synthetic warm 1: 5000 K (dò) | 5 | | +0.12 / +0.26 / +0.52 | −0.05 / −0.01 / +0.39 | 2.9 / 6.2 / 12.8 |
| synthetic warm 1: 4500 K | 5 | | +0.06 / +0.39 / +0.67 | −0.07 / +0.00 / +0.38 | 2.4 / 9.0 / 15.5 |
| synthetic cool 1: 7000 K (dò) | 5 | −0.16 / +0.17 / +0.38 | | −0.11 / −0.04 / +0.46 | 4.3 / 4.8 / 13.0 |
| synthetic cool 1: 8000 K | 5 | +0.03 / +0.42 / +0.53 | | −0.07 / −0.04 / +0.49 | 1.6 / 10.0 / 14.2 |
| synthetic warm 3: 3000 K | 5 | | +0.58 / +0.94 / +1.04 | −0.02 / +0.08 / +0.41 | 13.4 / 19.7 / 22.4 |
| synthetic green 1: tint 1.12 | 5 | | | +0.13 / +0.23 / +0.48 | 6.3 / 8.0 / 13.7 |

Ảnh ám thật trên bộ dev (**4 ảnh dương tính: 3 cool mức 1, 1 warm mức 3; không có ca green**):

| Ảnh | Nhãn | ln(b/r) | góc | confidence |
|---|---|---|---|---|
| color_cast_01 | cool 1 | **+0.11** | 2.6° | 0.71 |
| color_cast_02 (ambiguous) | cool 1 | **−0.29** (hơi ấm) | 7.6° | 0.48 |
| multi_defect_01 | cool 1 | **−0.08** (hơi ấm) | 2.2° | 0.66 |
| color_cast_03 | warm 3 | −3.01 (ln(r/b) +3.01) | 38.7° | 0.46 |

Tách được không (34 ảnh không ám: 5 `__clean` + 13 dev sạch + 16 dev khác):

| Dương tính | Ngưỡng bắt hết | Ảnh không ám lọt qua ngưỡng đó |
|---|---|---|
| 3 cool 1 thật | ln(b/r) ≥ −0.29 | 27 / 34 |
| 3 cool 1 thật, bỏ ca ambiguous | ≥ −0.08 | 18 / 34 |
| 5 cool 7000 K | ≥ −0.16 | 21 / 34 |
| 5 cool 8000 K | ≥ +0.03 (bắt 3/5 ở ≥ +0.42: 1 / 34) | 9 / 34 |
| 5 warm 5000 K | ln(r/b) ≥ +0.12 (bắt 2/5 ở ≥ +0.39: 3 / 34) | 13 / 34 |
| 5 warm 4500 K | ≥ +0.06 (bắt 4/5 ở ≥ +0.37: 5 / 34) | 16 / 34 |

Trục green và góc nguồn sáng cũng không tách: ảnh sạch có góc tới 9.7° (bỏ hoàng hôn/low-key),
cao hơn cả 3 ca cool thật (2.2–7.6°); green của ảnh sạch tới +0.22 (is_09) và +0.47 (cỏ sạch), chồng
lên tint 1.12 (+0.13…+0.48).

**Quyết định: mức 1 không đo được bằng nguồn sáng, để VLM + verify. Giữ ngưỡng.**
- Ba ca cool mức 1 thật nằm giữa phân bố ảnh sạch (hai ca còn đo hơi ấm). Không ngưỡng nào bắt được
  chúng mà không báo nhầm phần lớn ảnh sạch.
- Bộ ước lượng không phải nguyên nhân: so với chính ảnh sạch cùng nguồn, ám nhẹ đo đúng (ca dò sai
  1.05°, dịch ln(b/r) 7000 K +0.04…+0.19, 8000 K +0.23…+0.40 trên cả 5 nguồn). Cái không tách được là
  màu "trung tính" của từng cảnh (±0.4 trên ln(b/r) giữa các ảnh sạch). Ám nhẹ chỉ đo được *tương đối*:
  vòng ≥ 2 (`measurement/diff.py` của fusion) hoặc vùng chủ thể so với nền, không phải ngưỡng tuyệt đối.
- Mức 2–3 vẫn đo được: 3000 K (+0.58…+1.04) và color_cast_03 (+3.01) nằm ngoài ảnh sạch, trừ hoàng
  hôn chủ ý (+0.72, +1.00): phần đó do preserve/verify ở fusion chặn.

## Hạn chế / việc cho lead

1. **Ít nhãn ám màu thật:** chỉ 4 ảnh dương tính trên bộ dev (3 cool 1, trong đó 1 ambiguous), không
   có ca warm 1–2 hay green. Kết luận "mức 1 không tách được" dựa trên 3 ca thật + 10 ca synthetic.
   Muốn kiểm lại cần thêm ảnh ám nhẹ có nhãn qua trang review của reporter. Lane chưa hỏi vì kết luận
   không đổi khi thêm vài ảnh (ảnh sạch đã trải ±0.4).
2. **`multi_defect_01`, `color_cast_02`** đo hơi ấm dù nhãn cool: có thể nhãn chủ quan (thảm/vật xanh
   chiếm khung) hoặc nguồn sáng thật trung tính. Nên xem lại nhãn khi duyệt.
3. **Fusion:** severity ám màu với `illuminant_method ≥ 1` (confidence thấp) nên đi verify. Đề xuất
   trong `contract-changes.md` mục 9; ghi hai key vào §C1.
4. FP ám màu còn lại trên ảnh sạch (4/13) là cảnh chủ ý (hoàng hôn, low-key xanh) và một ảnh ấm nhẹ
   (`intentional_style_01`, ln(r/b) +0.43): lane không đổi ngưỡng warm vì brief chỉ cho hạ ngưỡng khi
   tách được, và nâng ngưỡng sẽ đổi khớp synthetic.
5. Person 1 cần review `measurement/color.py`.

Commit: xem `git log feat/analysis-v2-color` (code + test, rồi calibration, rồi docs). Không push,
không mở PR, không merge.

LANE DONE
