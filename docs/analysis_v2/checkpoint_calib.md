# Checkpoint sau `calib` (trước khi merge `agent`)

Lead, 2026-10-25. Code: `feat/analysis-v2` sau merge `calib` (trước `agent`), nên config `baseline`
vẫn là luồng perceive cũ, chỉ khác ở `analyze_image` (D4: mức đo suy từ calibration).
Model `gemini-3.5-flash-lite` (như `baseline.md`). Cùng 32 ảnh thật (30 chấm, 15 defect mong đợi,
13 ảnh sạch/chủ ý). Kết quả thô: `data/real/benchmark_results/cp-calib-*` (gitignored).

| Lần chạy | Gọi Gemini | Fallback | Lỗi |
|---|---|---|---|
| real / baseline | 32 | 0 | 0 |
| real / measure_only | 0 | – | 0 |
| synthetic / measure_only | 0 | – | 0 |

## Bộ thật — đếm, không chỉ tỉ lệ

| | baseline trước (`baseline.md`) | **baseline sau calib** | measure_only |
|---|---|---|---|
| Ảnh sạch có ≥ 1 FP (n = 13) | **7** (0.54) | **5** (0.39) | 10 (0.77) |
| — already_good (3) | 3 | 2 | 0 |
| — intentional_style (10) | 4 | 3 | 10 |
| Defect đúng / mong đợi (recall) | 7 / 15 (0.47) | 5 / 15 (0.33) | 5 / 15 (0.33) |
| Defect dự đoán / FP | 40 / 33 | 22 / 17 | 24 / 19 |
| MAE severity (trên cặp khớp) | 1.00 (7 cặp) | 0.80 (5 cặp) | 0.80 (5 cặp) |
| Preserve recall (21) | 0.76 | 0.71 | 0 (không có preserve) |

Theo loại, baseline trước → sau (tp / fp / fn):

| Loại | Trước | Sau |
|---|---|---|
| backlit_subject | 0 / 10 / 1 | 0 / **11** / 1 |
| blur | 1 / 9 / 0 | 0 / **1** / 1 |
| low_contrast | 0 / 6 / 1 | 0 / **0** / 1 |
| noise | 3 / 1 / 1 | 2 / 0 / 2 |
| underexposed | 3 / 5 / 2 | 3 / 2 / 2 |
| color_cast_cool | 0 / 1 / 2 | 0 / 1 / 2 |
| color_cast_warm | 0 / 1 / 1 | 0 / 2 / 1 |

## Đọc kết quả

1. **FP ảnh sạch giảm 7 → 5 / 13 (0.54 → 0.39). Không phải giảm rõ về phía ≤ 0.10.** Chênh 2 ảnh
   nằm trong dao động của một lần chạy VLM. Tổng FP giảm thật (33 → 17): calib bỏ gần hết
   `blur` (9 → 1) và `low_contrast` (6 → 0) báo sai.
2. **FP còn lại chủ yếu là `backlit_subject` origin `measured` (11)**, sinh ở
   `perception._measured_defects` bằng ngưỡng tay cũ trong `analyzer.py` (`_BRIGHTNESS_*`,
   `_*_CLIP_*`), **chưa** qua calibration (calib ghi ở `contract-changes.md` mục 6.3; `perception.py`
   thuộc lane `agent`/`fusion`). Ví dụ: `already_good_02` (dogs, person), `already_good_03` (person),
   `intentional_style_04` (hill). VLM tự nó chỉ gây FP ở 2 ảnh (`intentional_style_01`, `_08`).
3. **Recall giảm 7 → 5 / 15**: mất 1 `noise` (nhiễu camera thật dưới ngưỡng σ 2.86, calib hạn chế
   1–2) và 1 `blur` thật. Cả hai từng được bắt nhờ ngưỡng cũ quá nhạy.
4. **measure_only trên ảnh thật:** đúng 0/3 FP ở already_good, nhưng 10/10 intentional_style bị báo
   (low-key → underexposed, hoàng hôn → warm, high-key → overexposed). Số đo không biết chủ ý; đây
   là việc của preserve trong `fusion`, không phải lỗi calibration.
5. **Synthetic measure_only:** precision 0.86, recall 0.64, FP 1/20 ảnh sạch, MAE 0.14. Nhãn
   synthetic là phỏng đoán đầu của script sinh, nên MAE ≤ 0.4 chỉ cho thấy ngưỡng khớp phỏng đoán đó;
   không dùng làm bằng chứng chính.

## Tham chiếu: lane `agent` (perceive v2, chưa merge lúc đo)

Lane `agent` đo `vlm_only` v2 trên cùng bộ: FP ảnh sạch 0 / 13, recall 7 / 15, MAE 1.14 (7 cặp),
96 lần gọi. **Prompt rev 2 được chỉnh trên chính 30 ảnh này**, nên số này lạc quan, cần bộ giữ riêng.
Số đó không thuộc checkpoint calib.

## Kết luận checkpoint

Theo điều kiện người dùng đặt: FP ảnh sạch **không** giảm rõ từ 0.54 về phía ≤ 0.10 sau `calib`
(0.39, giảm 2 ảnh). → **Dừng trước wave 3, báo người dùng.**
