# Module 3: Benchmark và hiệu chỉnh tham số

Kết quả trong thư mục này được sinh tự động bởi:

```bash
python scripts/benchmark_module3.py --out-dir docs/module3/benchmark
```

Mỗi phần có một file `.csv` (từng ảnh × từng cấu hình) và một file `.md` (tóm tắt trung bình).
Thời gian chạy khoảng 10–20 phút trên CPU; phần chậm nhất là NLM.

## 1. Thiết lập

| Hạng mục | Giá trị |
|---|---|
| Ảnh sạch | `astronaut`, `coffee`, `chelsea`, `rocket`, lấy từ `skimage.data` (đóng gói sẵn, **chạy offline**) |
| Kích thước | Cạnh dài 384 px (`cv2.INTER_AREA`) |
| Ngẫu nhiên | Seed `2026` + `zlib.crc32(tên ảnh, loại suy giảm)`, nên tái lập được giữa các lần chạy |
| Chỉ số | PSNR, SSIM (`data_range=255`, `channel_axis=2`, giống Module 1), ΔE CIEDE2000 trung bình |
| Suy giảm | Nhiễu Gauss σ ∈ {5, 10, 15, 20, 25, 40}; gamma tối 1.5/2.0/2.5; nén tương phản về [80, 160]; mờ defocus đĩa r = 2; ám màu nhân (warm/cool/green) |

## 2. Quy ước `strength` của khử nhiễu

**σ_target = 12.5 · strength**, trong đó σ là độ lệch chuẩn nhiễu Gauss theo thang 0–255.
Planner kẹp strength ∈ [0.1, 2.0], tương ứng σ ∈ [1.25, 25].
**σ = 40 nằm ngoài mức mục tiêu của pipeline** và chỉ dùng để quan sát.

Mỗi method suy ra tham số OpenCV từ σ_target qua một hệ số. Hệ số được chọn bằng phần
`calibrate`: biết trước σ, đặt strength = σ/12.5, rồi quét hệ số trên σ ∈ {5, …, 25}.
**Quy tắc chọn:** lấy PSNR trung bình cao nhất; nếu có ứng viên kém không quá 0.15 dB nhưng SSIM
cao hơn thì chọn ứng viên đó (chi tiết trong [`calibrate.md`](calibrate.md)).

| Method | Tham số OpenCV | Trước | Sau hiệu chỉnh |
|---|---|---|---|
| bilateral | `σ_color = σ_space`, `d` | `50·s`, `max(5, 5s)` | `4·σ`, `max(5, 0.4σ)`: **giống hệt từng byte** |
| nlm | `h` | `10·s` | `0.8·σ`: **giống hệt từng byte** (0.7σ được +0.13 dB nhưng SSIM thấp hơn) |
| gaussian | `σ_blur`, kernel | `1.5·s` = 0.12σ, kernel `5s` (bị cắt cụt) | `0.06·σ`, kernel tự tính (±3σ_blur) |
| median | kernel | `max(3, 3s)` → 7×7 tại s = 2 | `2⌊σ/17.5⌋ + 3` → 3×3 / 5×5 / 7×7 |

Hai method được playbook của Agent dùng (bilateral, nlm) **không đổi hành vi**, nên các con số
trong `src/agent/knowledge/cards/*.yaml` vẫn đúng.

### So sánh trước và sau tại đúng strength mục tiêu (PSNR dB / SSIM)

| Method | σ = 5 (s = 0.4) | σ = 10 (s = 0.8) | σ = 25 (s = 2.0) |
|---|---|---|---|
| gaussian | 35.08 → **34.27** / 0.924 → 0.848 | 30.23 → **32.20** / 0.850 → 0.821 | 25.69 → **27.71** / 0.705 → 0.742 |
| median | 32.40 → 32.40 | 30.90 → 30.90 | 26.68 → **27.25** / 0.714 → 0.689 |
| bilateral | 38.85 (không đổi) | 34.41 (không đổi) | 28.99 (không đổi) |
| nlm | 35.32 (không đổi) | 31.35 (không đổi) | 27.35 (không đổi) |

**Mức độ khớp quy ước** (strength tối ưu đo được so với σ/12.5, trích [`denoise.md`](denoise.md)):

| σ | Mục tiêu | gaussian | median | bilateral | nlm |
|---|---|---|---|---|---|
| 5 | 0.40 | 0.6 | 0.1 | 0.4 | 0.25 |
| 10 | 0.80 | 0.8 | 0.1 | 0.8 | 0.6 |
| 15 | 1.20 | 1.0 | 0.1 | 1.25 | 1.0 |
| 20 | 1.60 | 1.25 | 1.5 | 1.5 | 1.5 |
| 25 | 2.00 | 1.5 | 1.5 | 1.75 | 1.75 |

### Nhận xét

- **Bilateral khớp quy ước tốt nhất** và cho PSNR cao nhất ở mọi σ ≤ 25 (ví dụ 34.41 dB tại σ = 10).
  Nó cũng nhanh hơn NLM khoảng 100 lần (khoảng 0.006 s so với 0.62 s mỗi ảnh 384 px), nên hợp làm mặc định.
- **Gaussian:** một hệ số tuyến tính không khớp được cả hai đầu dải σ. Tại σ = 5, σ_blur = 0.3 px
  gần như không làm gì (kém bản cũ 0.8 dB); tại σ ≥ 10 thì tốt hơn khoảng 2 dB. Strength tối ưu
  vẫn thấp hơn mục tiêu ở σ cao, vì làm mờ đều phải đánh đổi chi tiết, nên nhiễu càng mạnh thì càng
  không nên làm mờ tương ứng.
- **Median** là bộ lọc rời rạc và không hợp với nhiễu Gauss. Cửa sổ 3×3 là tối ưu đến σ ≈ 15
  (cột "0.1" nghĩa là bản nhỏ nhất đã tốt nhất). Method này dành cho nhiễu muối tiêu, loại nhiễu
  mà quy ước theo σ không mô tả được.
- **NLM** tối ưu ở khoảng 0.75 lần mục tiêu, tức là đang hơi mạnh tay. Mình giữ h = 0.8σ theo
  quy tắc chọn ở trên.

## 3. Phơi sáng, tương phản, làm nét ([`exposure.md`](exposure.md), [`sharpen.md`](sharpen.md))

| Ca | Tốt nhất | PSNR đầu vào → sau | SSIM đầu vào → sau |
|---|---|---|---|
| Thiếu sáng, số mũ 1.5 | gamma = 1.5 | 18.90 → 49.99 | 0.835 → 0.995 |
| Thiếu sáng, số mũ 2.0 | gamma = 2.0 | 14.61 → 44.69 | 0.619 → 0.977 |
| Thiếu sáng, số mũ 2.5 | gamma = 2.5 | 12.56 → 37.30 | 0.467 → 0.942 |
| Tương phản thấp [80, 160] | CLAHE clip = 2.0 | 15.42 → 16.32 | 0.660 → 0.798 |
| Defocus r = 2 | unsharp, amount = 0.8 | 29.59 → 30.50 | 0.875 → 0.896 |
| Defocus r = 2 | laplacian, amount = 0.5 | 29.59 → 30.02 | 0.875 → 0.881 |

- Gamma tốt nhất luôn bằng đúng số mũ suy giảm. Điều này xác nhận quy ước `I_out = 255·(I/255)^(1/γ)`
  (γ > 1 làm sáng) khớp với Module 1 và bộ sinh dữ liệu. PSNR giảm khi ảnh càng tối vì mất mức xám
  do lượng tử ở vùng tối, không thể khôi phục.
- CLAHE cải thiện SSIM rõ nhưng PSNR chỉ nhích nhẹ: CLAHE là cân bằng cục bộ, không phải phép nghịch
  đảo của việc nén tương phản tuyến tính.
- Làm nét không đảo ngược được mờ (thông tin tần số cao đã mất); nó chỉ tăng cảm nhận độ nét. Unsharp
  tốt hơn Laplacian ở cùng mức thay đổi.

## 4. Màu sắc ([`color.md`](color.md))

| Ám màu | Đầu vào ΔE | Gray World ΔE (bất biến) | White Patch ΔE (bất biến) | `temperature_shift` tốt nhất (oracle) |
|---|---|---|---|---|
| warm (1.20, 1, 0.80) | 8.00 | 15.82 (2.95) | 12.80 (9.61) | −1.0 → 2.58 |
| cool (0.85, 1, 1.18) | 7.48 | 15.31 (1.31) | 6.48 (3.31) | +1.0 → 0.92 |
| green (0.90, 1.15, 0.90) | 12.86 | 15.25 (1.32) | **6.90** (2.42) | +0.2 → 12.49 |

**Cách đọc:** "ΔE bất biến" là ΔE giữa WB(ảnh bị ám) và WB(ảnh sạch). Chỉ số này đo việc thuật toán
có cho cùng một kết quả bất kể nguồn sáng hay không, tức là đúng mục tiêu của cân bằng trắng.

- **Gray World bất biến tốt** (ΔE 1.3–3.0) nhưng **lệch xa ground-truth**. Lý do là 4 ảnh gốc tự thân
  không trung tính: `coffee` có trung bình kênh (R, G, B) = (159, 86, 52), `chelsea` = (148, 111, 87).
  Gray World ép mọi cảnh về xám, nên sẽ khử luôn màu chủ đạo vốn có của cảnh. Đây là giới hạn đã biết
  của giả thuyết, đã ghi trong `HANDOFF_PERSON4.md` (mục playbook "avoid").
- **White Patch sửa được ám xanh lá** (12.86 → 6.90). Đây là ám màu mà `temperature_shift` gần như
  không chạm tới được (12.86 → 12.49), vì nó chỉ dịch trục đỏ–lam. Với ám ấm, White Patch kém đi do
  kênh R bị cháy ở vùng sáng (gain 1.2 đẩy R > 255): vật trắng không còn đo được đúng màu.
- Cột `temperature_shift` là **oracle**: giá trị được chọn bằng cách so với ground-truth, và với một
  ám thuần đỏ–lam như ở đây thì đúng là mô hình của nó. Cột này không so sánh công bằng được với WB,
  vốn hoạt động mù (không biết ground-truth).

## 5. Giới hạn của benchmark

1. Chỉ có 4 ảnh, độ phân giải nhỏ (384 px). Đủ để chọn hệ số, nhưng chưa đủ để kết luận tổng quát.
2. Suy giảm là tổng hợp (nhiễu Gauss độc lập theo pixel, ám màu nhân thuần túy). Nhiễu thật của cảm
   biến phụ thuộc tín hiệu và tương quan giữa các kênh.
3. Hệ số được chọn theo PSNR (kèm luật hòa theo SSIM). Chọn theo chỉ số cảm nhận khác (BRISQUE/NIQE
   của Module 1) có thể cho kết quả khác.
4. Ground-truth không trung tính màu, nên ΔE so với ground-truth chưa phải thước đo công bằng cho
   white balance. Vì vậy benchmark báo cáo thêm chỉ số bất biến.
