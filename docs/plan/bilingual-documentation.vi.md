# Kế hoạch triển khai tài liệu song ngữ

[English](bilingual-documentation.md) | [Tiếng Việt](bilingual-documentation.vi.md)

Trạng thái: đã lập kế hoạch. Ngày: 2026-10-30. Tài liệu này xác định công việc cần làm; không xác nhận bộ doc hoặc việc phát hành đã hoàn tất.

## Kết quả và phạm vi

Xây dựng bộ tài liệu thống nhất cho AI Image Doctor, mặc định tiếng Anh và có bản tiếng Việt tương đương. Người đọc phải có thể chạy ứng dụng, hiểu pipeline và thuật toán cổ điển đã triển khai, tái lập bằng chứng đánh giá hiện có và tích hợp API.

Khi triển khai xong, phải commit, push và merge tài liệu vào `main` qua quy trình PR của repo. Đây là phần bắt buộc trong yêu cầu của người dùng, không phải bước tùy chọn. Riêng công việc lập kế hoạch này chưa bắt đầu viết toàn bộ doc hoặc phát hành.

## Quy ước ngôn ngữ và tên file

- EN là bản mặc định tại đường dẫn chuẩn hiện có: `README.md`, `docs/architecture.md`… VI dùng cùng tên với hậu tố `.vi.md`: `README.vi.md`, `docs/architecture.vi.md`.
- Đặt liên kết English / Tiếng Việt hai chiều ở đầu mỗi cặp trang. Liên kết nội bộ ưu tiên giữ ngôn ngữ đang đọc nếu đã có bản dịch.
- Hai bản phải tương đương về thông tin kỹ thuật, phạm vi, ví dụ, giới hạn, lệnh và bằng chứng. Viết tiếng Việt tự nhiên, không dịch máy móc từng chữ.
- Giữ nguyên identifier, endpoint, biến môi trường, lệnh và đơn vị. Dịch diễn giải, bảng, chú thích và nhãn sơ đồ.
- Khi hành vi thay đổi, cập nhật cả hai bản trong cùng PR. Ghi commit nguồn đã đối chiếu và ngày thu thập bằng chứng cho thông tin phụ thuộc implementation hoặc phép đo.
- Giữ đường dẫn và liên kết cũ. Không đổi tên báo cáo lịch sử chỉ để áp dụng quy ước mới.
- Không dịch toàn bộ kế hoạch cũ, audit, nhật ký thực nghiệm hoặc hướng dẫn nội bộ cho agent. Phân loại chúng là lịch sử/tham khảo trong mục lục. Prompt runtime và knowledge-base card cũng ngoài phạm vi; việc dịch doc không được làm đổi hành vi chương trình.

## Sản phẩm bàn giao

Mỗi trang dành cho người đọc được tạo/cập nhật trong bảng đều có file EN chuẩn và file VI `.vi.md` đi kèm.

| File (đường dẫn EN) | Nội dung bắt buộc | Nguồn chính cần đối chiếu |
|---|---|---|
| `README.md`, `docs/README.md` | Mục đích, ràng buộc xử lý pixel bằng thuật toán cổ điển, quickstart, đường đọc theo đối tượng, phân biệt doc hiện hành và lịch sử | README hiện có, dependency, entry point ứng dụng |
| `docs/architecture.md` | Thành phần hệ thống, ranh giới 4 module, luồng API/UI/dữ liệu, state và điều kiện dừng thực tế, fallback, lưu trữ, vòng đời hỏi ý/tinh chỉnh | `src/agent/graph.py`, state/session, `src/api/`, `web/` |
| `docs/algorithms.md` | Ranh giới đo lường và chẩn đoán; denoise, gamma, CLAHE, sharpen, chỉnh màu; mask/blending; công thức, tham số, thứ tự, giới hạn, ảnh trước/sau | Processing, measurement, region engine, giới hạn planner và test |
| `docs/evaluation.md` | Phân biệt test phần mềm và chất lượng ảnh; dữ liệu và nguồn gốc; metric có/không tham chiếu thực sự được triển khai; baseline, lệnh tái lập, kết quả, ca thất bại và giới hạn | `data/`, script evaluation/benchmark, evaluator, báo cáo hiện có |
| `docs/user-guide.md` | Upload, phân tích, hỏi ý nếu có, xử lý, so sánh, tinh chỉnh, xuất ảnh; screenshot hiện tại, cách đọc điểm, kịch bản demo ngắn | React app và hành vi workspace API hiện tại |
| `docs/api.md` | Đăng nhập, vòng đời phiên, ví dụ request/response chạy được, lỗi, giới hạn upload, export, lưu trữ, route hiện tại và cũ | Đăng ký route FastAPI, schema, auth/config và API test |
| `docs/interfaces.md` | Hợp đồng ảnh/mask, schema dùng chung và bất biến đã kiểm chứng; dẫn nguồn định nghĩa đầy đủ thay vì sao chép toàn bộ code | Schema, executor, ranh giới module và contract test |
| `web/README.md`, `frontend/README.md` | Cài đặt/cấu hình web và Gradio cũ tương ứng; phân biệt rõ entry point | Build script, mẫu môi trường, phục vụ static và entry point frontend |

Giữ `CONTRIBUTING.md`, `docs/decisions.md`, README module và báo cáo benchmark chi tiết làm tài liệu tham khảo có liên kết. Ghi rõ tài liệu nào chưa dịch; không yêu cầu dịch mọi tài liệu lịch sử hoặc chuyên sâu để hoàn tất đợt này.

## Thứ tự thực hiện

### 1. Xác lập nguồn tài liệu

- Đọc hướng dẫn repo, kiểm tra working tree, nhánh upstream và PR hiện có trước khi sửa. Giữ nguyên thay đổi không liên quan của người dùng.
- Phân loại doc thành hiện hành, có phần lỗi thời, lịch sử hoặc tham khảo. Tạo bảng phạm vi bao phủ trong mục lục.
- Đối chiếu hành vi với source và test. Giải quyết khác biệt giữa sơ đồ cũ và implementation trước khi mô tả pipeline.
- Tạo bảng thuật ngữ EN/VI chung trong mục lục, gồm diagnosis, treatment plan, mask, refinement, fallback và evaluation.

Nghiệm thu: mỗi đầu mục có nguồn, đối tượng đọc và cách kiểm chứng; không mô tả tính năng chưa có là đã triển khai.

### 2. Viết kiến trúc và thuật toán

- Soạn EN cho kiến trúc và thuật toán, rồi hoàn thiện bản VI tương ứng trước khi chuyển bước.
- Dùng Mermaid cho sơ đồ thành phần, sequence hoặc state khi hữu ích. Node và transition phải khớp implementation.
- Theo một ví dụ chân dung ngược sáng xuyên suốt đo lường, chẩn đoán, chọn mask, thao tác đã validate, blending và đánh giá. Tham số số học phải lấy từ lần chạy thực tế.
- Làm rõ vai trò điều phối của AI và xử lý pixel cổ điển. Ghi đúng phần mở rộng hoãn triển khai và lỗi ảnh chưa sửa được.
- Đối chiếu `docs/interfaces.md` với contract hiện tại và tạo bản VI.

Nghiệm thu: sơ đồ, ý nghĩa/miền tham số, mask, điều kiện dừng và fallback khớp source; EN/VI mô tả cùng hành vi.

### 3. Viết bằng chứng có thể tái lập

- Kiểm kê dataset và output benchmark hiện có trước khi quyết định cần chạy thêm gì.
- Với mỗi thực nghiệm, ghi dataset/split, số mẫu, tiền xử lý, lệnh, dependency/model, phần cứng khi liên quan, revision nguồn và cách hiểu metric.
- Chỉ so sánh điều kiện tương thích. Tách đánh giá ảnh tổng hợp có tham chiếu, ảnh thật không tham chiếu và chấm điểm bằng người.
- Chỉ dùng lại kết quả đủ nguồn gốc. Ghi rõ phần chưa đo; không tự tạo số liệu hoặc coi unit test đạt là chứng minh chất lượng ảnh.
- Có ví dụ thành công và thất bại, ảnh gốc/kết quả, overlay mask khi cần, chú thích cả hai ngôn ngữ.
- Lưu hình được phép chia sẻ vào `docs/assets/`; không đưa ảnh riêng tư, credential, model lớn hoặc mẫu không được phép phân phối vào Git.

Nghiệm thu: mỗi nhận định định lượng có artifact hoặc lần chạy tái lập; giới hạn rõ ràng và tương đương ở hai bản.

### 4. Viết hướng dẫn sử dụng và tích hợp

- Kiểm chứng quickstart từ cài dependency đến khởi động, gồm chế độ không có API key. Tách hướng dẫn model semantic tùy chọn.
- Chạy hành trình người dùng hiện tại, chụp screenshot từ phiên bản đó. Không mô tả UI dự kiến như tính năng đã có.
- Đối chiếu ví dụ API với route/schema; ưu tiên dẫn OpenAPI sinh tự động cho danh mục đầy đủ sau khi xác minh đường dẫn.
- Mô tả authentication, workspace endpoint hiện tại, giới hạn route cũ, lưu trữ và lỗi cài đặt thường gặp.
- Cập nhật README gốc/frontend; hoàn thiện mục lục song ngữ cuối cùng khi mọi trang đích đã tồn tại.

Nghiệm thu: người mới có thể làm theo quickstart, hoàn thành demo và hiểu điều kiện chạy ví dụ API bằng cả hai ngôn ngữ.

### 5. Kiểm tra toàn bộ

- Kiểm tra liên kết file, anchor, hình, liên kết đổi ngôn ngữ và danh sách cặp file.
- Render/preview Markdown, Mermaid, công thức, bảng và screenshot. Kiểm tra khả năng đọc và thông tin tương đương EN/VI.
- Đối chiếu lệnh/ví dụ với repo. Ghi rõ nội dung chưa chạy và lý do; không đánh dấu đã kiểm chứng khi chưa làm.
- Rà từng cặp trang về thuật ngữ, nội dung kỹ thuật, cấu hình mặc định và giới hạn.
- Chạy kiểm tra repo bắt buộc: `ruff check src/ tests/` và `pytest tests/unit/ -v`. Nếu buộc phải sửa code/config frontend, chạy thêm typecheck, test và build frontend. Không thêm sửa code không liên quan để mở rộng công việc doc.
- Chạy demo/API/benchmark phù hợp với tuyên bố trong hướng dẫn. Tránh API trả phí nếu chưa được cho phép; có thể dùng đường chạy offline và bằng chứng hiện có để kiểm chứng đúng phạm vi của chúng.

Nghiệm thu: không còn liên kết nội bộ hỏng, thiếu cặp bản dịch, tuyên bố thiếu cơ sở, lộ secret hay file không liên quan; kiểm tra bắt buộc đạt và ghi rõ giới hạn kiểm chứng.

### 6. Commit, push và merge vào main

1. Fetch upstream, tạo nhánh tài liệu theo `CONTRIBUTING.md`, ví dụ `docs/bilingual-documentation`, từ base tích hợp đã kiểm tra. Không commit trực tiếp vào `main`.
2. Review diff, stage rõ từng tài liệu và asset cần thiết. Commit các cặp EN/VI liên quan theo Conventional Commits, ví dụ `docs: add bilingual architecture and algorithm guides`.
3. Push nhánh, mở PR vào `develop`, mô tả phạm vi, danh sách cặp trang và kết quả kiểm chứng thực tế. Gắn PR đã tạo vào chat.
4. Chờ CI bắt buộc và reviewer approval theo repo. Sửa góp ý đồng thời cả hai bản, rồi merge bằng chiến lược repo hỗ trợ.
5. Xem toàn bộ diff `develop` so với `main` trước khi mở release PR. Nếu có thay đổi ngoài phạm vi chưa được phép phát hành, không vô tình đưa chúng vào release; tạo nhánh chỉ chứa doc từ `main` bằng các commit tài liệu đã review và mở PR giới hạn phạm vi.
6. Mở và gắn PR release/doc vào `main`, đáp ứng check và review rồi merge. Không force-push hoặc vượt qua bảo vệ nhánh.
7. Fetch lại, xác nhận tài liệu đã có trên remote `main`. Báo commit ID, URL PR, kiểm tra và giới hạn còn lại.

Người dùng đã yêu cầu commit/push/merge bộ doc này; không hỏi lại quyền một cách dư thừa. Nếu credential, bảo vệ nhánh, review bắt buộc hoặc từ chối phê duyệt chặn công việc, giữ nhánh/PR đã push nếu có thể và báo đúng điểm chặn. Chưa xác nhận remote thì không tuyên bố đã giao vào `main`.

## Checklist hoàn tất

- [ ] Đủ sản phẩm và cặp EN/VI; EN là điểm vào mặc định.
- [ ] Tài liệu tham khảo hiện hành vẫn truy cập được; nội dung lịch sử được phân loại.
- [ ] Kiến trúc, thuật toán và contract khớp implementation đã đối chiếu.
- [ ] Kết quả đánh giá có bằng chứng tái lập và giới hạn trung thực.
- [ ] Quickstart, hành trình sử dụng và ví dụ API có kết quả kiểm chứng được ghi nhận.
- [ ] Liên kết, preview, tương đương ngôn ngữ và kiểm tra repo bắt buộc đạt.
- [ ] Bộ doc đã commit, push, review và merge vào remote `main`.
- [ ] Bàn giao có link PR, commit ID và tóm tắt kiểm chứng.
