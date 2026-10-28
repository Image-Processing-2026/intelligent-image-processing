# Rà tương thích backend / frontend — 2026-10-27

Mốc so sánh: `e40e7cf` → `cc6da5f`, cùng thay đổi working tree.
Các commit backend: `346b672` (quyết định theo lỗi từng vùng), `c252ea1` (bỏ câu phong cách,
DiagnosisEdit), `2ee8477` (truyền edit qua phiên), `cc6da5f` (chat phân tích và overlay).
Nguồn: `docs/interfaces.md`, `docs/plan_intent_chat.md`, schema và logic hiện hành trong backend.

## Standards

Hai lỗi đã tái hiện và sửa:

- Overlay bị ghi đè cùng artifact ID nhưng frontend giữ nguyên `src`: phiên bản URL theo
  `updated_at` khiến trình duyệt tải lại ảnh sau chỉnh sửa. Đã xác nhận request ảnh HTTP 200 thật.
- Chuyển phase khi đang gửi analysis note làm local state của chat reset, vô tình mở lại nút:
  khóa thao tác bằng pending state của mutation ở cấp trang.

Kiểm tra Git phát hiện cả `web/src/lib/` bị luật Python `lib/` bỏ qua (actions, edits, labels,
theme, utils). Thêm ngoại lệ cho thư mục frontend; các file nay xuất hiện trong Git status,
không tự stage/commit chúng.

## Spec

Đã đồng bộ:

- `defect_status`, `side_effects`: hiện mức lỗi trước/sau, trạng thái theo vùng, tác dụng phụ
  trong flow log, phase Xử lý và tổng kết. Hỗ trợ phiên cũ không có các field mới.
- Dùng `measurement`, `region_stats`, `fusion_notes`, `diff` thật cho bằng chứng và thay đổi.
- Khi bấm Xem vùng sau khi đã bật ảnh gốc, bật lại overlay.
- Ảnh trung gian khi rollback là **ảnh thử đã bị loại**, không phải ảnh được giữ.
  Đầu vào vòng sau lấy ảnh được giữ gần nhất, bỏ qua các ảnh thử đã rollback.
- `final_evaluation` của backend vẫn mô tả ảnh thử cuối khi rollback. Frontend chỉ dùng đánh giá
  của vòng được giữ trước đó; không có bản ghi thì báo thiếu. Không gán điểm ảnh thử cho ảnh giữ.
- Đánh giá theo vùng của một vòng rollback luôn được ghi rõ là thuộc ảnh thử bị loại.
- Nhãn ảnh trước trong so sánh/phóng to phân biệt ảnh đầu vòng với ảnh gốc; bổ sung tên lỗi
  mới và nhãn lỗi ngoài khả năng toolbox.

Luồng một kết quả, không có câu phong cách, danh sách câu hỏi rỗng, thêm/xóa edit và API chat
đã được frontend hiện tại xử lý; giữ tương thích dữ liệu phiên cũ.

## Kiểm chứng

Regression mới đi từ lỗi sang pass; toàn bộ frontend: 69 test pass, TypeScript và build pass.
Browser với API thật: tạo phiên, thêm lỗi mặt, xác nhận cùng artifact có URL revision mới,
request ảnh tải lại 200, tắt overlay rồi Xem vùng bật lại đúng. Kiểm tra dữ liệu từng vùng
trên phiên xử lý thật và responsive sáng/tối theo script `tests/browser/studio_layout.js`.
Test bằng rule-based; không gọi Gemini và không sửa thuật toán backend.
Không chạy lại Python suite trong lượt thay đổi frontend này. Chưa commit/push.
