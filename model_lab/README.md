# Image Doctor · Model Lab

So sánh khả năng hiểu ảnh của **Gemma 4 31B**, **Qwen 3.8 27B** và
**Gemini 3.5 Flash-Lite**. Chế độ hiện tại: **ảnh + một system prompt dùng chung**,
một request và một câu trả lời cho mỗi model. Không gọi hệ thống xử lý ảnh thật.

## Chạy

Từ thư mục `intelligent-image-processing`, PowerShell:

```powershell
& .\.venv\Scripts\python.exe model_lab\app.py
```

Mở <http://127.0.0.1:7862>. Dùng runtime Python hiện có của project,
không cần cài framework frontend. Đọc `GEMINI_API_KEY` và `GROQ_API_KEY` từ
`intelligent-image-processing/.env`; Gemma và Gemini dùng chung key Google.
Ghi đè file bằng `MODEL_LAB_ENV` hoặc đặt key trong environment. Key chỉ ở server.

## Cách thử

1. Chọn ảnh và chỉnh **System prompt dùng chung** nếu cần.
2. Chọn cạnh dài và giới hạn output; bấm **Phân tích ảnh**.
3. Đọc ba câu trả lời cạnh nhau; copy riêng từng model hoặc tải JSON.
   Ảnh PNG và kết quả tự lưu trong `model_lab/runs/<id>/`.

Mỗi model nhận **cùng byte ảnh PNG và nguyên văn system prompt**. User message chỉ
chứa ảnh; không có text ngầm, tên file, mục tiêu phụ, lịch sử, kết quả model khác,
tool schema, danh sách thuật toán hay giới hạn tham số được chèn thêm.

Instruction mặc định yêu cầu quan sát nội dung và chi tiết ảnh, phân biệt bằng
chứng với suy đoán, đánh giá chất lượng và đề xuất cải thiện nếu cần. Model tự chọn
phương pháp, công cụ, cách trình bày và mức chi tiết; không có whitelist toolbox,
thứ tự xử lý, số vấn đề, số bước hoặc số từ bắt buộc. Không có request lập kế hoạch
riêng, không validator hoặc xử lý pixel. Toàn bộ instruction hiển thị và sửa được.

Ảnh được xoay theo EXIF, chuyển RGB, ghép nền trắng nếu trong suốt, resize giữ tỉ
lệ và không upscale. Tối đa 20 MB / 32 MP. Đối chiếu bằng đúng preview ảnh gửi đi.
Temperature 0.4, thinking tắt/tối thiểu, output mặc định 1.000 tokens/model
(có thể giảm xuống 512/768). Đây vẫn là giới hạn kỹ thuật; cảnh báo nếu API báo
bị cắt. Các giới hạn và chính sách của nhà cung cấp vẫn áp dụng.

Mỗi lượt mới = **3 request ảnh**, thay cho 6 request của luồng cũ.
Groq chờ 65 giây và Gemini 13 giây giữa các lượt để giảm lỗi quota; thời gian chờ
hiển thị riêng. Không tự đổi model, retry hoặc xoay key. Thử lại chỉ gọi model lỗi,
giữ nguyên ảnh và instruction. Dừng chỉ bỏ request chưa gửi; request đang gửi vẫn
hoàn tất hoặc timeout (150 giây).

## Lịch sử

Lượt mới có `protocol_version: vision-free-v1`, `system_prompt` và một response
mỗi model. Lượt cũ vẫn đọc/copy/export được, có nhãn **lượt cũ theo toolbox**;
prompt cũ không tự được khôi phục vào ô instruction mới. Không retry luồng cũ
để tránh gọi lại các ràng buộc; chọn lại ảnh và chạy lượt mới.

Giao diện local chỉ nhận một lượt đồng thời, bind 127.0.0.1, không có auth và
không dùng làm dịch vụ public. Không gửi ảnh trước khi bấm chạy.
