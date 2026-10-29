[English](documentation-verification.md) | [Tiếng Việt](documentation-verification.vi.md)

# Kiểm chứng tài liệu

Ngày: 2026-10-30 (Asia/Bangkok). Mốc source: `02dd6f9`. Sửa doc không đổi hành vi ứng dụng.

| Kiểm tra | Kết quả |
|---|---|
| Đối chiếu source | Đã xem graph, route/auth/config workspace, schema và công thức xử lý |
| Tương đương EN/VI | Rà đủ mười cặp trang cho người đọc và bản ghi này về phạm vi/lệnh |
| Ruff | `ruff check src/ tests/` đạt; lint script hình đạt |
| Frontend | Typecheck đạt; 13 file / 80 test đạt; build production đạt, còn cảnh báo chunk lớn |
| Hình | Script chạy thành công, đã kiểm tra trực quan |
| Unit suite | 1625 đạt, 6 cảnh báo; 119.02 s |
| Liên kết nội bộ | Đã kiểm tra trang mới; đủ đích liên kết |

Lượt unit trong sandbox ngừng tiến triển nên bị ngắt; đã chạy lại suite bắt buộc với quyền cache bình thường. Không gọi Gemini trả phí hoặc chạy benchmark chất lượng ảnh mới. Số liệu lịch sử giữ nguồn gốc/giới hạn cũ. Đối chiếu source không xác nhận kiểm tra trình duyệt/API thủ công hoặc renderer Markdown/Mermaid. Không trình bày screenshot như bằng chứng UI thật trong đợt này.

Đợt này xây nền doc song ngữ. Walkthrough trình duyệt đầy đủ, screenshot UI và kiểm tra renderer theo kế hoạch vẫn chưa xong trừ khi có bản ghi bổ sung xác nhận bên dưới. Trạng thái remote nằm trên PR/bàn giao cuối; file này tự nó không xác nhận merge main.

Smoke trình duyệt: build production ở cổng 8012 với kho scratch riêng và key Gemini trống. Đăng ký, upload, chẩn đoán và thực thi phác đồ rỗng hoàn tất. Chụp ba screenshot; đã xem trực quan screenshot phiên. Chưa kiểm tra đầy đủ tinh chỉnh/export ảnh thật. Bản ghi này thay thế nhận định trước đó về chưa chụp screenshot.
