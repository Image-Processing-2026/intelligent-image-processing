# A/B tiếp theo: sửa lỗi cơ bản trên ảnh thật

## Thiết kế đã chốt trước khi chạy

17 ảnh Wikimedia có sẵn trong `data/basic_test_manifest.json`, xem contact sheet trước khi chọn. Không tạo suy giảm giả, không dùng holdout. 12 ca sửa lỗi, 3 đối chứng, 2 giới hạn. Một lượt mỗi phiên bản trên mỗi ảnh (34 lượt), nhằm sàng lọc trên nhiều cảnh; chưa đo được độ ổn định theo lần gọi.

| Nhóm | ID sau tiền tố `basic_` | Số |
|---|---|---:|
| Thiếu sáng toàn ảnh | underexposed_01, underexposed_03, underexposed_04 | 3 |
| Chủ thể tối cục bộ | local_dark_01, local_dark_02 | 2 |
| Chói | overexposed_01 | 1 |
| Bạc màu / tương phản thấp | low_contrast_01 | 1 |
| Nhiễu | noise_01, noise_02, noise_03 | 3 |
| Màu quá rực theo yêu cầu màu tự nhiên | oversaturated_01, oversaturated_03 | 2 |
| Đối chứng | already_good_01, already_good_06, intentional_style_01 | 3 |
| Giới hạn | highlight_clipping_01, dark_noisy_01 | 2 |

Thay hai ứng viên xe cứu hỏa và đường đất vì biểu hiện lỗi không rõ bằng hai ca giảm rực. Không gọi tất cả ảnh là lỗi nặng: ảnh trứng có mức thiếu sáng vừa/nhẹ; sắc độ băng và màu rực còn yếu tố sở thích. Ảnh đền có trời đã trắng, chỉ chấm mục tiêu giảm chói phần còn chi tiết, không yêu cầu phục hồi trời.

Mỗi ảnh có một yêu cầu tiếng Việt cố định trong protocol, giống nhau cho V1/V2. Nhãn `expected` chỉ phục vụ báo cáo, không truyền cho LLM. Intent được truyền qua trường `notes` hiện hữu. Vì vậy đây là phép thử pipeline tự chẩn đoán với ghi chú mục tiêu, **không phải phép thử riêng khả năng sửa lỗi khi được cung cấp chẩn đoán đúng**. Trong mã hiện tại, Perceive/Verify không nhận notes và nhánh không có actionable defect có thể dừng trước Plan. Không tự sửa hành vi đó giữa thử nghiệm.

`src` lấy nguyên byte từ snapshot pilot trước. Prompt V2 giữ nguyên; runner chỉ thêm khả năng truyền notes. Cả hai dùng model trong protocol, tối đa ba vòng, memory tắt, cùng ảnh và metadata. Hai cặp chạy đồng thời; trong mỗi cặp chạy tuần tự, thứ tự phiên bản xen kẽ theo ảnh. Vì chạy đồng thời, thời gian chỉ là số đo vận hành tham khảo, không phải benchmark latency độc lập.

Các snapshot và SHA-256 được lưu ở `data/prompt_ab/2026-10-29-basic/protocol.json`. Toàn bộ prompt, schema, ảnh gửi, response, execution/rollback và ảnh đầu ra được lưu theo ảnh/phiên bản. Bộ đếm calls là các lời gọi qua wrapper phase đã instrument, không bảo đảm bao trùm lời gọi nội bộ khác hoặc retry HTTP.

## Tham chiếu thuật toán cố định

Đã tạo trước khi xem đầu ra V1/V2: gamma 0.72 cho thiếu sáng, gamma 1.25 cho chói, CLAHE trên L cho tương phản, NLM cho nhiễu, saturation ×0.8 cho màu rực, identity cho đối chứng. Tham số đầy đủ trong protocol. Gamma ở ca cục bộ áp toàn ảnh, không có mask oracle; chỉ giúp xem khả năng nâng sáng, không là lời giải đúng cho yêu cầu giữ nền.

Đây là ảnh tham khảo, **không phải ground truth**, cũng chưa được chứng minh tốt. Không tính PSNR/SSIM với ảnh này. Gallery đặt tham khảo trong mục thu gọn và hướng dẫn xem sau khi chấm X/Y để giảm neo đánh giá.

## Chấm kết quả

1. Đọc yêu cầu của ảnh; so từng đầu ra với ảnh gốc ở cùng kích thước.
2. Với X và Y riêng: đạt mục tiêu = không đạt / một phần / rõ / chưa xác định.
3. Với X và Y riêng: tác dụng phụ mới = không thấy / nhẹ / nặng / chưa xác định.
4. Sau đó chọn X / Y / ngang nhau / cả hai không đạt / chưa rõ, kèm ghi chú.

Đối chứng đạt khi giữ được ảnh tốt/phong cách; ít thay đổi không tự là kém. Ca giới hạn cần kiểm tra cả thông báo giới hạn trong trace, không thể chấm đầy đủ từ ảnh đơn thuần. Ba nhóm báo cáo riêng, không gộp thành một win rate. Cặp đầu ra giống hệt không mang bằng chứng phiên bản nào tốt hơn. Chỉ coi 17 ảnh là 17 đơn vị độc lập, không coi 34 lượt là 34 ảnh.

Gallery giữ nhãn X/Y, có crop đồng tọa độ và mở PNG đầy đủ. JSON đánh giá schema 2 ghi run_id để tránh lẫn pilot cũ. Không chuyển các lựa chọn thẩm mỹ cũ thành điểm đạt mục tiêu mới.

## Tái chạy

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/prompt_ab_basic.py --prepare-only
.\.venv\Scripts\python.exe -X utf8 scripts/prompt_ab_basic.py
.\.venv\Scripts\python.exe -X utf8 scripts/prompt_ab_report.py data/prompt_ab/2026-10-29-basic
.\.venv\Scripts\python.exe -X utf8 scripts/prompt_ab_basic_gallery.py data/prompt_ab/2026-10-29-basic
```

Runner bỏ qua lượt đã có result.json; giữ snapshot/protocol của lần chuẩn bị. Muốn thí nghiệm khác dùng `--output` mới. Chưa thử Clarify/Refine theo hội thoại, chưa ablation và chưa có chấm người dùng cho bộ mới. Không tuyên bố V2 tốt hơn dựa vào ACCEPT hoặc số pixel đổi.
