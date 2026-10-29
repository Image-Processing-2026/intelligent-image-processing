# Đánh giá noisy_lena — 2026-10-23

## Điều kiện thử

- Ảnh: `C:/Users/Tien/Workspace/University/imageProcessing/intelligent-image-processing/data/real/noisy_lena.png`.
- Run: `3f75e65e-f006-489b-b1b8-c2d52bc9e3a2`; output gốc: [result.json](runs/3f75e65e-f006-489b-b1b8-c2d52bc9e3a2/result.json).
- Cùng PNG 534 × 528, không resize; SHA256 `e59abf428bcc1895317ce2d36fff80cc924368128aaa0334e22870bac021b568`.
- Cùng system instruction mặc định vision-free-v1; chỉ ảnh và instruction, không filename/context/tools/schema. Một lời đáp gồm phân tích và đề xuất.
- Temperature 0.4, tối đa 1000 output tokens; cấu hình thinking off/minimal theo provider. Cùng ảnh không có nghĩa các provider tokenize ảnh giống nhau.
- Gemma gặp HTTP 500 rồi 503, thành công ở lần gọi thứ ba. Hai model còn lại chỉ gọi một lần. Thời gian dưới đây là request thành công, không gồm các lần lỗi.

## Quan sát độc lập trước khi đọc output

Chân dung thang xám của một người phụ nữ đội mũ rộng vành, quay nhìn qua vai. Phần vai hiện rõ. Nhiễu hạt sáng/tối dày khắp vùng ảnh: mặt, vai, mũ và nền. Mắt, môi, biên mặt/mũ và các sợi trang trí còn nhận ra nhưng chi tiết nhỏ bị nhiễu che lấp. Viền xám ngoài ảnh tương đối đều. Không có căn cứ chỉ từ ảnh để xác nhận ISO, thiết bị, quy trình tạo nhiễu, phân phối Gaussian/AWGN hay ý định nghệ thuật. Không dùng tên ảnh làm dữ kiện cho model.

## Kết luận cho lượt thử này

**Output tốt nhất: Gemini 3.5 Flash-Lite; tiếp theo Gemma 4 31B; cuối cùng Qwen 3.8 27B.** Đây là đánh giá định tính của output quan sát được, không phải bảng xếp hạng năng lực tổng quát.

| Model | Điểm tốt | Điểm yếu | Hoàn thành / thời gian |
|---|---|---|---|
| Gemini 3.5 Flash-Lite | Đúng ảnh thang xám, tư thế nhìn qua vai, còn thấy vai; nhận ra nhiễu nặng và các cấu trúc còn giữ được. Đề xuất khử nhiễu, nêu vùng xử lý, chi tiết cần bảo vệ và nguy cơ mất texture/da bị sáp. | Gọi nhiễu là muối tiêu hoặc Gaussian chưa được xác minh. Kết luận chắc chắn không phải phong cách nghệ thuật vượt quá bằng chứng. Nên gắn lựa chọn bộ lọc với việc kiểm tra loại nhiễu. | STOP, 950 tokens, 5.55 s |
| Gemma 4 31B | Mô tả đúng chủ thể, thang xám, nhiễu và chi tiết. Có đề xuất đầy đủ, lưu ý mắt/môi/lông vũ và nguy cơ làm mờ. | Ban đầu nói nhiễu nặng rồi nói trung bình vì vẫn nhận ra cấu trúc — tiêu chí này chưa đủ. Từ giả thuyết Gaussian chuyển thành khẳng định “Vì đây là nhiễu Gaussian”. Gọi Bilateral “tối ưu nhất” khi chưa thử và so sánh. Cũng quá chắc chắn về ý định tạo nhiễu. | STOP, 993 tokens, 45.09 s |
| Qwen 3.8 27B | Nhận ra nhiễu rất nặng, dấu hiệu hạt và ảnh hưởng đến chi tiết. Thận trọng khi đánh giá độ nét dưới nhiễu. | Nói “Phần vai và cơ thể bị cắt bỏ” dù vai hiện rõ. Suy đoán màu trắng/kem từ ảnh xám, nền bokeh, ISO 3200/6400, nguồn 300×300 và lỗi scan mà không có dữ kiện xác nhận. Trộn khả năng pixelation do phóng to với nhiễu hạt. Không có phần đề xuất trong output nhận được. | length, 1000 tokens, 3.55 s; bị cắt |

## Diễn giải công bằng

- Nếu chỉ xét hiểu nội dung và lỗi ảnh, Gemini cũng ít lỗi cụ thể hơn trong lượt này; cả ba đều nhận ra vấn đề chính là nhiễu. Nhận ra tên ảnh mẫu quen thuộc không được tính là bằng chứng về khả năng quan sát tốt hơn.
- Qwen thiếu đề xuất vì chạm giới hạn output, không thể kết luận model không biết đề xuất. Xếp cuối ở đây còn dựa trên lỗi quan sát phần vai và các suy đoán thiếu căn cứ, không chỉ độ đầy đủ.
- Không trừ điểm chất lượng thị giác của Gemma vì lỗi dịch vụ. Tuy vậy hai lần lỗi là vấn đề thực tế khi demo.
- Chưa có ảnh sạch đối chiếu, xác minh nguồn nhiễu hay thực nghiệm lọc ảnh; không coi lời khuyên của bất kỳ model nào là phương án đã được chứng minh tối ưu.
- Một ảnh quen thuộc và một output thành công/model chưa đủ chọn model chung cho project. Bước tiếp theo phù hợp là dùng thêm ảnh lỗi đa dạng, ảnh tốt đối chứng, và yêu cầu chung ngắn gọn để cả ba trả lời đủ trong ngân sách 1000 tokens.
