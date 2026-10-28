# Bộ ảnh thật để test các ca cơ bản — v1

Ngày chọn: **27/10/2026**. Bộ ảnh nằm tại **`data/real/basic_v1/`**. Mở
**`index.html`** trong thư mục đó để lọc nhóm, xem ảnh và lấy đúng input PNG.
Manifest có thể tái lập: **`data/basic_test_manifest.json`**.

## Phạm vi

**30 ảnh thật tải từ Wikimedia Commons**, gồm 14 ca basic, 4 ca tiếp theo,
9 ca giữ nguyên và 3 ca kiểm tra giới hạn. Không có ảnh suy giảm do script tạo.
Đã xem ảnh dự tuyển để chọn; đây là bộ khởi đầu để phát triển/test thủ công,
chưa đủ lớn để suy ra tỷ lệ thành công của hệ thống.

| Nhóm | Số ảnh | Cần kiểm tra |
|---|---:|---|
| `underexposed` — basic | 4 | Nâng sáng vừa phải; giữ texture lông, băng, vỏ trứng |
| `overexposed` — basic | 2 | Hạ chói chủ thể còn chi tiết; không hứa cứu trời cháy |
| `low_contrast` — basic | 2 | Tăng độ rõ nhẹ; giữ màu da, không halo |
| `oversaturated` — basic | 3 | Giảm rực theo intent yêu cầu màu tự nhiên |
| `noise` — basic | 3 | NIND ISO3200, crop 1:1; bớt hạt mà không bệt |
| `warm_cast` — secondary | 2 | Giảm ám vàng trên mẫu cây/giấy; không chấm tint xanh–tím |
| `local_dark` — secondary | 2 | Nâng riêng mặt/hoa; giữ tóc tối, nền và viền mask |
| `already_good` — control | 6 | Không chỉnh thừa: chân dung, tóc đen, vải, đồ vật và xe đạp |
| `intentional_style` — control | 3 | Giữ bình minh ấm, đèn đêm, tương phản thấp có chủ ý |
| `highlight_clipping`, `severe_color`, `dark_noisy` — boundary | 3 | Nhận giới hạn và dừng; không tính vào thành công basic |

**Khoảng trống:** chưa có nhóm ảnh thật nhạt màu được xác nhận cần tăng saturation;
không gán nhãn ép cho cảnh pastel hoặc cảnh có màu tự nhiên nhẹ. Nhóm thiếu tương phản
cũng còn ít. Chưa có tập chân dung nhiễu đủ đa dạng tông da.

## Nguồn và cách chuẩn bị input

- Tìm bằng browser và Commons Search/API từ các nhóm
  [Underexposure](https://commons.wikimedia.org/wiki/Category:Underexposure),
  [Underexposed images](https://commons.wikimedia.org/wiki/Category:Underexposed_images),
  [Overexposed images](https://commons.wikimedia.org/wiki/Category:Overexposed_images),
  [Oversaturated images](https://commons.wikimedia.org/wiki/Category:Oversaturated_images),
  [Wrong white balance](https://commons.wikimedia.org/wiki/Category:Wrong_white_balance),
  [Quality images](https://commons.wikimedia.org/wiki/Category:Quality_images), và
  [Natural Image Noise Dataset](https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset).
- Khoảng 140 ảnh dự tuyển đã được xem qua các contact sheet; loại tranh, sơ đồ, ảnh
  HDR có bảng cầu vật liệu, mẫu ám màu quá nặng và các ảnh không hợp ca cơ bản.
- `sources/` giữ nguyên byte đã tải. Trường `source_variant` phân biệt **bản gốc**
  với **thumbnail chuẩn do Commons cung cấp**; không gọi mọi file nguồn là ảnh gốc.
  URL và kích thước ảnh gốc vẫn được giữ trong manifest. Dùng thumbnail chuẩn cho phần
  lớn ảnh sau khi Commons giới hạn tải ảnh gốc.
- Input ở `<category>/<case_id>.png`: xử lý hướng EXIF, chuyển RGB, cạnh dài tối đa
  1600 px; một số ảnh nhỏ chỉ có 500 px. PNG tránh thêm một lần nén JPEG khi chuẩn bị input.
- Riêng `noise/`: tải ảnh gốc NIND rồi crop **1600×1200 ở tỷ lệ 1:1**,
  không nội suy; tọa độ crop được ghi trong `download_receipt.json`. Giữ ảnh gốc trong
  `sources/` để xem cả cảnh. Preview trong gallery đã thu nhỏ, **không dùng preview để chấm nhiễu**.
  Ảnh droid lấy crop phía dưới để giữ đồ chơi, hai ảnh còn lại lấy crop giữa ảnh.
- `author`, `license`, `license_url`, `source_page` có cho từng ảnh. Khi dùng ảnh/ảnh đã sửa
  trong báo cáo hoặc chia sẻ lại, lấy attribution từ manifest và tuân thủ giấy phép từng file.
  Không coi giấy phép trang category là giấy phép cho mọi ảnh.

## Dùng để test

1. Bắt đầu bằng **14 ca basic**; lấy đúng PNG từ gallery và dùng `intent` đã ghi.
2. Chạy **9 ca control** để kiểm tra hệ thống giữ ảnh đã tốt hoặc phong cách có chủ ý.
3. Sau đó thử 4 ca secondary và 3 ca boundary; báo cáo kết quả riêng theo nhóm.
4. Điền `results_template.csv`: `run_id`, lỗi mục tiêu có cải thiện không, bảo toàn có đạt
   không, artifact quan sát được, chấp nhận hay không và người duyệt. So trước/sau ở
   cùng kích thước, xem vùng quan trọng tại 100%; không chỉ nhìn điểm IQA.
5. Với ca boundary, đạt khi hệ thống giới hạn can thiệp và giải thích phần không cứu được;
   không đòi khôi phục chi tiết đã mất.

Nhãn hiện có là **`agent_reviewed`**, `human_reviewed=false`; cần người dùng duyệt trước
khi dùng làm ground truth. Severity là gợi ý; các ca sáng/tương phản/màu nhẹ mang tính
chủ quan và phải đọc cùng intent. Chưa chạy pipeline/VLM trên bộ này và chưa có điểm chất lượng đầu ra.

Bộ này là **development/manual test**, không phải hold-out. Script kiểm tra không trùng
`commons_title` với hai manifest benchmark/hold-out hiện có và không có file nguồn giống
hệt byte trong bộ mới. Điều này không chứng minh ảnh không từng xuất hiện trên mạng
hoặc cùng cảnh không có trong dữ liệu khác. Hai ảnh bút NIND ISO200/3200 cố ý cùng cảnh.

**Không tính PSNR/SSIM với ảnh input hoặc ảnh ISO200 chỉ vì cùng cảnh**: các ảnh ở đây
đặt `reference_eval_eligible=false`; cặp NIND chưa được kiểm tra căn chỉnh pixel và exposure.
Manifest riêng có các trường phục vụ duyệt thủ công; chưa cắm trực tiếp vào
`scripts/benchmark_analysis.py` và không sửa schema benchmark hiện tại.

## Tải lại và kiểm chứng

```powershell
.\.venv\Scripts\python.exe -X utf8 data/download_basic_test_set.py
.\.venv\Scripts\python.exe -X utf8 data/download_basic_test_set.py --verify-only
```

Script tải tuần tự, bỏ qua nguồn đã có, dừng khi Commons trả 429. Khi gặp giới hạn, đợi
nguồn cho phép rồi chạy lại; không đổi host hoặc cố vượt giới hạn. `--verify-only` chạy
offline, kiểm tra decode, kích thước, phép chuẩn bị input và SHA-256. Biên nhận chỉ ghi
hoàn tất khi mọi ca tải đủ. Gallery và contact sheet được tạo sau khi kiểm chứng đủ 30 ca.

Lượt tải này lấy được 27 ảnh bằng script; ba bản gốc NIND tải bằng thao tác download
trên browser rồi chép vào `sources/` theo `source_file` trong manifest, do request ảnh gốc
từ Python bị 429. Nếu tái lập gặp cùng giới hạn, tải thủ công từ `source_page` theo cách này,
sau đó chạy lại script để tạo input và kiểm chứng checksum. `source_sha256` đã lưu cho đủ 30 nguồn.

Ảnh và kết quả test trong `data/real/` được gitignore như quy ước repo; manifest,
script tải và tài liệu này có thể commit để người khác tái lập.
