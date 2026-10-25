# Sai số ước lượng của lõi đo P1 trên bộ synthetic

Lần đo đầu tiên của luồng phân tích mới (lead, 2026-10-25, sau khi merge `bench` + `measure`).
Không gọi Gemini.

- Bộ dữ liệu: `scripts/build_calibration_set.py --fetch-sources` (seed 7) → 150 ảnh, 5 ảnh nguồn
  (`already_good_01..03`, `clean_grass_01`, `clean_landscape_01`), cạnh dài ≤ 1024 px.
- Script: `python scripts/benchmark_measurement.py` (kết quả trong
  `data/real/benchmark_results/measurement-synthetic-2/`, gitignored). Trung bình 0.47 s/ảnh.
- Giá trị "thật" tính từ tham số degradation **và** số đo trên biến thể đối chứng `__clean` cùng
  nguồn (ảnh sạch vẫn có chút nhiễu, mờ, ám màu). Cách tính ghi trong docstring của script.
- numpy: 70 test `test_measurement_*` xanh trên numpy 1.26.4 (opencv 4.10, scipy 1.17,
  scikit-image 0.24), như trên numpy 2.1.3.

## Kết quả

| Đại lượng | n | MAE | median | trong ngưỡng | sai lệch TB (có dấu) | thiếu |
|---|---|---|---|---|---|---|
| σ nhiễu luma (Gaussian σ 2–25) | 25 | 5.8 % | 5.0 % | 96 % (≤ 15 %) | +3.0 % | 0 |
| σ nhiễu luma, có gamma kèm (combo) | 10 | 15.7 % | 11.2 % | 50 % (≤ 15 %) | −5.5 % | 0 |
| σ defocus (Gaussian σ 0.8/1.5/3) | 15 | 13.1 % | 13.5 % | 93 % (≤ 25 %) | **−13.0 %** | 0 |
| σ defocus, có nhiễu kèm | 5 | 20.8 % | 23.5 % | 60 % (≤ 25 %) | −20.8 % | 0 |
| độ rộng cạnh / 2.563, so với đối chứng | 15 | 7.4 % | 6.0 % | 100 % (≤ 25 %) | −2.7 % | 0 |
| độ rộng cạnh, có nhiễu kèm | 5 | 6.8 % | 7.4 % | 100 % | +3.8 % | 0 |
| hướng mờ chuyển động (độ, chu kỳ 180°) | 10 | 10.9° | 6.6° | 90 % (≤ 15°) | – | 0 |
| vector nguồn sáng (độ) | 18 | 5.5° | 5.2° | 39 % (≤ 3°) | – | 2 |

## Đọc kết quả

1. **Nhiễu: đạt.** σ luma đúng tới ~5 %. Chú ý đơn vị: `gaussian_noise.sigma` trong params là σ
   **mỗi kênh RGB**; σ luma tương ứng = 0.669·σ. Lane `calib` phải quy đổi khi khớp ngưỡng.
   Khi có gamma đi kèm, nhiễu phụ thuộc độ sáng nên một σ toàn cục kém hơn: dùng `sigma_dark_ratio`.
2. **Defocus: dùng được, lệch thấp có hệ thống ~13 %** (nặng hơn khi có nhiễu, −21 %). Đây là
   hệ quả của làm mịn trước + trừ giải tích (lệch 2 bên dưới). Đơn điệu, nên hiệu chỉnh được.
3. **Độ rộng cạnh tuyệt đối không phải chỉ báo mờ trên ảnh thật.** Trên ảnh sạch, độ rộng cạnh
   đã là 7–13 px (cạnh tự nhiên không phải bậc thang), nên σ mờ 0.8 và 3 chỉ đổi 5.02 → 5.35
   (đơn vị /2.563). Chỉ khi so với chính ảnh đó (vòng 2, hoặc vùng chủ thể so với nền) thì nó
   chính xác (7 %). → `calib` dùng `defocus_sigma` cho severity mờ; `edge_width` chỉ dùng tương đối.
4. **Mờ chuyển động: hướng đúng khi ảnh vốn ít hướng trội, sai khi ảnh có texture có hướng.**
   `clean_grass_01` và `clean_landscape_01` **sạch** đã có `motion_coherence` 0.75–0.77; ca
   `clean_landscape_01__blur_m5` đo 70° (thật 22.5°) vì hướng của nội dung lấn át. Vì vậy
   coherence toàn cục **không được** dùng một mình làm bằng chứng mờ chuyển động (sẽ báo nhầm trên
   cỏ, tóc, vân gỗ). Xem quyết định cho lệch 3 bên dưới.
5. **Nguồn sáng: ám nhẹ/vừa đạt (4500 K, 8000 K: 0.5–5.7°), ám mạnh bị đo thiếu.** CCT 3000 K đo ra
   ám 5–13° trong khi kỳ vọng 19–24° (sai 8–14°); `clean_grass_01` ở 3000 K và tint xanh không
   ước lượng được (không còn pixel trung tính). Nguyên nhân khả dĩ: cổng pixel trung tính
   `C* < 12` loại chính các pixel xám đã bị ám mạnh, chỉ giữ pixel vốn hơi ngược màu. Kết quả vẫn
   đơn điệu theo CCT nhưng bị nén, nên severity 3 cho ám màu sẽ khó đạt. Ghi chú: "thật" giả định
   gain trên sRGB ≈ gain^2.2 tuyến tính; sai số mô hình này nhỏ so với 10°.

## Quyết định về các lệch khỏi plan của lane `measure` (lead)

| # | Lệch | Quyết định |
|---|---|---|
| 1 | độ rộng cạnh 10–90 % thay vì cực trị Marziliano | Chấp nhận. Kết quả 3 ở trên cho thấy dùng tương đối là đúng. |
| 2 | không khử nhiễu song phương; làm mịn Gaussian + trừ giải tích | **Chấp nhận.** Gaussian ∗ Gaussian cộng theo bình phương nên phép trừ đúng về lý thuyết; song phương phi tuyến làm méo cạnh (lane đo: defocus −40 %). Giá phải trả: defocus lệch thấp 13–21 % có hệ thống → `calib` hiệu chỉnh, không sửa trong `measure`. |
| 3 | không xác nhận hướng mờ bằng FFT | **Chưa chấp nhận bỏ.** Số đo 4 cho thấy coherence toàn cục báo nhầm trên ảnh sạch có texture. Giao lane `calib` (mở rộng phạm vi sang phần motion của `measurement/sharpness.py`): thêm một phép xác nhận hướng mờ, ví dụ dị hướng của defocus theo hướng pháp tuyến cạnh (mờ chuyển động làm cạnh vuông góc hướng chuyển động rộng hơn hẳn cạnh song song) hoặc phổ FFT như plan. Cho tới khi có, severity `motion_blur` từ đo = 0, chỉ để VLM quyết định. |
| 4 | không Tenengrad | Chấp nhận: thừa so với `laplacian_var` + defocus. |
| 5 | không zone histogram 11 vùng | Hoãn. Không có key trong C1; `fusion` thêm nếu cần cho `low_contrast`. |
| 6 | không kiểm tra skin locus trên mặt | Hoãn sang lane `agent` (có mask mặt từ `resolve_subject_mask`), trường thêm có default. |
| 7 | σ theo tile bằng phương sai patch phẳng thay λ_min | Chấp nhận. |
| 8 | không báo vị trí blob cháy | Hoãn; thêm trường có default khi `evidence.py` (lane `agent`) cần crop vùng cháy. |
| 9 | ảnh nhỏ không có σ | Chấp nhận (thiếu key, có note). |
| 10 | bộ synthetic chưa chạy | Đã chạy, file này. |
| 11 | numpy 1.x chưa kiểm | Đã kiểm: 70/70 xanh trên numpy 1.26.4. |

Việc mở cho `calib` từ file này: hiệu chỉnh defocus lệch thấp; quy đổi σ RGB → σ luma; xác nhận
mờ chuyển động (lệch 3); ám màu mạnh bị nén (xem xét chọn pixel trung tính lặp: ước lượng
Gray-World trước, bù rồi mới lọc `C*`), chỉnh trong `measurement/color.py`.
