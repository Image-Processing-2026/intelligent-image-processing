# AI Image Doctor — demo báo cáo tiến độ

## Chạy giao diện

Từ repo root, dùng môi trường Python hiện có:

```powershell
python frontend/app.py
```

Mở http://localhost:7860. Đã kiểm tra API với Gradio 6.28.0; theme/CSS truyền vào `launch()`. CSS đọc theo đường dẫn của `app.py`, không tải font/icon bên ngoài. `.env` chỉ được nạp khi chạy trực tiếp, không khi import để test. Khi chạy file trực tiếp, UI thêm repo root vào đường dẫn import để tìm `src`; đã xác nhận lệnh khởi động trả HTTP 200 (bộ nhớ được tắt riêng cho lượt kiểm tra khởi động).

## Kịch bản 3–5 phút

1. **0:00–0:30:** giới thiệu AI kết hợp xử lý ảnh cổ điển, đây là demo tiến độ.
2. **0:30–1:00:** chọn mẫu **Thiếu sáng**. Mẫu chỉ nạp ảnh local, không tự chạy pipeline. Mặc định 3 vòng/3 phiên bản; mở Tuỳ chọn xử lý để chọn 1 phiên bản nếu cần giảm thời gian. Ground truth chỉ dành cho ảnh chuẩn tương ứng trong đánh giá thử nghiệm.
3. **1:00–2:30:** bấm **Phân tích & xử lý**, chờ trạng thái thực tế. So sánh ảnh gốc và ảnh sau xử lý. Nếu có nhiều phiên bản, chọn rồi **Áp dụng phiên bản đã chọn** để render độ phân giải gốc. Tải ảnh bằng nút tải có sẵn trong khung kết quả.
4. **2:30–4:00:** xem Timeline và chẩn đoán/phác đồ ngay dưới kết quả. Giải thích từng vòng đo/chẩn đoán → lập kế hoạch → xử lý → đánh giá. JSON nằm trong mục đóng mặc định, giữ nguyên mã quyết định.
5. **4:00–5:00:** giải thích giới hạn: SHIP chỉ đạt ngưỡng đánh giá của hệ thống; STOP_BEST_EFFORT nghĩa là đã dừng, cần xem kết quả/lịch sử. Điểm là đánh giá ước lượng, không phải độ chính xác hay phần trăm chất lượng. Chat là phần bổ sung nếu còn thời gian: chọn ảnh → phân tích/hỏi ý → trả lời → góp ý.

## Preflight và giới hạn

- Thử trước đúng ảnh sẽ trình bày và đo thời gian thực tế trên máy/mạng demo; thời lượng trên là phân bổ thuyết trình, không phải cam kết latency.
- Nếu mạng/API không sẵn sàng, nói rõ đang dùng fallback. Xem nguồn trong diagnosis/phác đồ thực tế; có API key không chứng minh Gemini đã chạy thành công.
- Không cam kết xử lý tốt ảnh cháy sáng hoặc segmentation. UI không hiển thị mask vì output hiện tại chưa có dữ liệu này.
- Đổi/xoá ảnh hoặc ground truth xoá kết quả, phác đồ và case id liên quan. Làm mới xoá trạng thái tab xử lý, không xoá SQLite. Đổi ảnh chat xoá hội thoại, câu hỏi, session và phác đồ. Session trả lời được tiêu thụ; muốn trả lời lại phải bắt đầu phiên mới.
- Trong job, các input/nút liên quan bị khoá, không có tiến độ phần trăm giả. Khi lỗi, kết quả cũ được xoá, nút phục hồi, UI chỉ hiện câu ngắn để thử lại; exception được log phía server.

## Kiểm tra bàn giao (2026-10-13)

```powershell
python -m pytest tests/unit/test_frontend_app.py -q --basetemp=scratch/pytest-ui-demo
python -m ruff check frontend/app.py tests/unit/test_frontend_app.py
python -m ruff format --check frontend/app.py tests/unit/test_frontend_app.py
$env:GEMINI_API_KEY = $null
python -m pytest tests/unit -q --basetemp=scratch/pytest-ui-demo-all
```

Tạo thư mục `scratch` trước nếu chưa có. Đã chạy các lệnh trên; lượt xác nhận cuối dùng basetemp có hậu tố riêng trong `scratch`. Frontend: **16 passed**; unit suite offline: **748 passed**; Ruff check/format và `git diff --check`: pass. Fixture repo xoá API key cho mỗi test; không sửa `.env`, không gọi Gemini. Có warning Typer/pyiqa; một số lượt có warning quyền ghi pytest cache, không có test failure ở lượt cuối.

Browser smoke bằng Playwright: viewport CSS thực tế 1366×768 và 390×844; không tràn ngang, hai khung ảnh xếp dọc ở mobile. Đã chạy ảnh thiếu sáng qua pipeline fallback thực tế với 1 và 3 phiên bản; kiểm tra trạng thái chờ/khoá reset, timeline, JSON, render full-res, đổi mẫu/xoá kết quả và làm mới. Đã chạy chat fallback phân tích → trả lời → góp ý. Lỗi pipeline được **mock riêng trên server kiểm tra**, xác nhận thông báo không lộ payload và nút chạy phục hồi; đây không phải bằng chứng Gemini thật.

Ảnh chụp/log smoke nằm trong `scratch/` và `.playwright-mcp/` (artifact kiểm tra):

- `scratch/ui-demo-mobile-empty.png`: màn hình rỗng.
- `scratch/ui-demo-desktop-result.png`: kết quả pipeline fallback thực tế, không phải kết quả mock.

Chưa xác minh Gemini thật, latency trước buổi trình bày, chất lượng toàn bộ ảnh mẫu, mọi tổ hợp ground truth và điều hướng bàn phím toàn giao diện. Không coi unit/smoke này là nghiệm thu chất lượng dự án. Server fallback có log không có mask dùng được cho vùng face và telemetry MediaPipe không kết nối được; không sửa backend theo phạm vi plan.

Chỉ sửa `frontend/app.py`, `frontend/styles.css`, `frontend/README.md`, `tests/unit/test_frontend_app.py`; giữ nguyên file untracked có sẵn. Không sửa backend/dependencies, không commit/push.


## Sửa lỗi giao diện sáng/tối (2026-10-13)

CSS nền sáng cố định xung đột với chữ sáng của theme tối, làm tiêu đề/trạng thái gần như biến mất (tương phản tiêu đề đo được 1,03:1 trước sửa). Nền, chữ, viền và focus hiện dùng biến màu của theme Gradio. Blocks dùng `fill_width=True`, khung ảnh cao 300px và các nút mẫu/hành động có `min_width=0` để tránh cột quá hẹp, nút xuống dòng không cần thiết.

Sau khi đổi CSS/Python, dừng tiến trình UI cũ bằng Ctrl+C rồi chạy lại `python frontend/app.py`; refresh trình duyệt đơn thuần không nạp lại CSS đã đọc vào bộ nhớ server.

Kiểm tra riêng lượt sửa này: 16 test frontend pass; Ruff check/format và `git diff --check` pass. Browser kiểm tra cả hai tab ở chiều rộng 1366, 995, 768, 390px, chế độ sáng/tối: không tràn ngang, mobile xếp dọc; chữ kiểm tra đạt ít nhất 13,33:1, nút chính nằm trong viewport desktop 768px. Luồng ảnh mẫu thiếu sáng → 1 vòng/1 phiên bản → kết quả fallback → làm mới pass, không có lỗi console. Không gọi Gemini trong lượt kiểm tra này.

Kiểm tra hồi quy trình duyệt ở `tests/browser/frontend_layout.js`: file chứa hàm `async (page)` để chạy trên trang UI đã mở bằng Playwright (có thể truyền file vào `browser_run_code_unsafe`). Hàm kiểm tra màu render thực tế, tràn ngang, độ rộng container, vị trí nút chính và hai cột ở mobile; không nằm trong bộ pytest unit. Hàm đã báo lỗi với bản cũ và pass với bản sửa.
