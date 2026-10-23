# Plan chỉnh giao diện phục vụ demo báo cáo tiến độ

Ngày: 2026-10-13. Trạng thái: **đề xuất để thực thi**, chưa triển khai.

## 1. Mục tiêu và phạm vi

Làm giao diện Gradio hiện có gọn, thống nhất và đủ ổn định cho bài demo 3–5 phút: chọn ảnh → chạy xử lý → so sánh trước/sau → giải thích chẩn đoán và các vòng xử lý. Đây là chỉnh UI phục vụ báo cáo tiến độ, không phải nghiệm thu chất lượng toàn dự án.

Giữ Python/Gradio và pipeline hiện có. Không chuyển sang React/Next.js, không thiết kế lại backend, không thêm model, không sửa thuật toán hay cố cải thiện ảnh bằng frontend. Không gọi API Gemini trong kiểm thử tự động. Không thêm dependency nếu component Gradio hiện có đáp ứng được.

**File được sửa:**
- `frontend/app.py`: layout, copy, event handlers và trạng thái UI.
- `frontend/styles.css` (mới, nếu cần): CSS có phạm vi theo `elem_id`/`elem_classes`.
- `frontend/README.md`: cách chạy, kịch bản demo và hạn chế đã biết.
- `tests/unit/test_frontend_app.py`: chỉ bổ sung test hành vi mới có rủi ro.

Không sửa `src/`, requirements, model/data, các plan cũ hoặc file untracked sẵn có. Nếu phát hiện lỗi backend, ghi vào báo cáo bàn giao; không mở rộng phạm vi tự sửa.

## 2. Hiện trạng đã kiểm tra

- `create_app()` đang có ba tab: xử lý, lịch sử, trò chuyện. Upload ground truth và nhiều thông số nằm ngay màn hình chính; thông tin và emoji khá dày.
- Đã có Before/After, timeline, chẩn đoán, JSON, gallery biến thể và render độ phân giải gốc. Tái sử dụng các chức năng này.
- `process_with_variants()` trả 10 đầu ra, `process_interface()` trả 5, `render_chosen_variant()` trả 2. Giữ các hợp đồng này; nếu cần cập nhật thêm component, dùng wrapper UI riêng và cập nhật wiring rõ ràng.
- Các callback xử lý chính gọi pipeline đồng bộ; UI cần trạng thái chờ rõ ràng nhưng không có dữ liệu tiến độ từng node. Không hiển thị tiến độ phần trăm hoặc bước đang chạy giả.
- Nhãn `STOP_BEST_EFFORT` hiện ghi “mức tối ưu tốt nhất”, mạnh hơn điều hệ thống chứng minh được.
- Đổi ảnh chưa có handler xoá kết quả/phác đồ cũ; cần tránh render phác đồ của ảnh A lên ảnh B.
- Ba ảnh local có sẵn: `data/real/coffee_underexposed.jpg`, `astro_noisy.jpg`, `cat_warm_cast.jpg`. Chưa đánh giá lại chất lượng đầu ra của chúng trong lượt lập plan này.
- `requirements` chỉ đặt lower bound Gradio. Trước khi code, kiểm tra phiên bản thực tế và API của phiên bản đó; không tự nâng/hạ phiên bản để chạy mẫu code.

## 3. Thiết kế chốt để triển khai

Phong cách sáng, nền xám rất nhạt, card trắng, một màu nhấn teal, chữ tối dễ đọc. Dùng font hệ thống, không tải font/icon bên ngoài. Giảm emoji còn logo đầu trang nếu muốn. Nội dung tiếng Việt nhất quán.

Khung nội dung rộng tối đa khoảng 1200–1280px, padding 24px, gap 16–24px, bo góc 12px. Không dùng CSS theo class tự sinh của Gradio. Ảnh dùng `contain`, không crop/stretch; hai khung trước/sau cùng chiều cao khoảng 360–420px ở desktop. Màn hình nhỏ xếp dọc, không tràn ngang.

```text
AI Image Doctor                         Bản demo báo cáo tiến độ
Chẩn đoán và cải thiện ảnh bằng AI kết hợp xử lý ảnh cổ điển

[ Xử lý ảnh ]  [ Trò chuyện & tinh chỉnh ]

1. Chọn ảnh                                  2. Xem kết quả
┌────────────────────────┐                  ┌────────────────────────┐
│ Ảnh gốc / upload       │                  │ Ảnh sau xử lý          │
│                        │                  │                        │
└────────────────────────┘                  └────────────────────────┘
Ảnh mẫu: Thiếu sáng / Nhiễu / Ám màu          Trạng thái + số vòng
▸ Tuỳ chọn xử lý                            Điểm đánh giá ước lượng
[ Phân tích & xử lý ] [ Làm mới ]

▸ Các phiên bản ảnh (mở khi có dữ liệu)
3. Quá trình xử lý (cùng tab, dưới kết quả)
Timeline → chẩn đoán/phác đồ từng vòng
▸ Dữ liệu kỹ thuật (JSON)
```

- Chuyển lịch sử xuống cùng tab xử lý để thuyết trình không phải chuyển tab; giữ đầy đủ dữ liệu đang có.
- `Tuỳ chọn xử lý` đóng mặc định: số vòng, số phiên bản, ground truth. Giữ mặc định hiện tại 3 vòng/3 phiên bản và giới hạn hợp lệ hiện tại; có thể chọn 1 phiên bản lúc demo để giảm thời gian.
- Ground truth giải thích ngắn: “Chỉ dùng khi có ảnh chuẩn tương ứng để đánh giá thử nghiệm”.
- Ảnh mẫu chỉ nạp ảnh, **không tự chạy pipeline**. Chỉ hiển thị file thực sự tồn tại; thiếu file thì bỏ mẫu đó, không tải mạng. Tên mô tả tình trạng đầu vào, không hứa chất lượng đầu ra.
- Nút chính: “Phân tích & xử lý”. Nút render: “Áp dụng phiên bản đã chọn”; dùng khả năng tải ảnh có sẵn của component nếu phiên bản Gradio hỗ trợ. Không gắn nhãn “Tải xuống” cho hành động chỉ render ảnh.
- Trạng thái kết quả: `SHIP` → “Đạt ngưỡng đánh giá của hệ thống”; `STOP_BEST_EFFORT` → “Đã dừng — xem kết quả và lịch sử”; score → “Điểm đánh giá ước lượng”, không gọi là độ chính xác/phần trăm chất lượng.
- Nếu thể hiện nguồn Gemini/fallback, lấy từ diagnosis/plan thực tế. Có API key không chứng minh lần chạy đã dùng Gemini thành công. Không gọi model để lấy thông tin trạng thái.
- Tab trò chuyện giữ chức năng hiện có, áp dụng cùng màu/chữ/khoảng cách và hướng dẫn 3 bước; không xây thêm chức năng mới.
- Chưa thêm hiển thị mask: dữ liệu mask chưa có trong output UI, làm việc này sẽ mở rộng sang backend.

## 4. Các bước thực thi tuần tự

### UI-01 — Bố cục và nội dung

1. Đọc `create_app`, các callback và test frontend hiện có; ghi nhận tuple outputs trước khi sửa.
2. Áp dụng theme/CSS tương thích phiên bản cài đặt. Đọc CSS bằng đường dẫn tương đối với `__file__`.
3. Xây layout theo mục 3: hai tab, ảnh cân đối, cấu hình thu gọn, timeline cùng trang, JSON đóng mặc định.
4. Thay nhãn trạng thái gây hiểu nhầm; giữ mã quyết định và dữ liệu JSON nguyên nghĩa. Điều chỉnh test nếu chỉ thay copy, không xoá kiểm tra hành vi.

**Đạt khi:** `create_app()` dựng được; không mất component/output/event hiện có; desktop nhìn rõ ảnh và nút chính; JSON không chiếm trang.

### UI-02 — Luồng demo và quản lý trạng thái

1. Thêm ảnh mẫu và nút Làm mới. Reset toàn bộ kết quả tab xử lý: ảnh sau, status, timeline, reasoning, JSON, variants, lựa chọn, phác đồ và case id.
2. Khi input image hoặc ground truth đổi/xoá, xoá kết quả và state phụ thuộc; reset không xoá dữ liệu SQLite. Dùng event phù hợp, tránh vòng lặp reset khi UI cập nhật output.
3. Nút áp dụng phiên bản chỉ hoạt động khi có lựa chọn hợp lệ. Tab chat: đổi ảnh phải reset session/questions/chat history/actions/case id và output; nút trả lời/góp ý chỉ bật khi có state tương ứng.
4. Hiển thị loading thật bằng cơ chế Gradio và thông báo “Đang xử lý, thời gian phụ thuộc ảnh và số vòng”. Không dựng thanh tiến độ giả. Ngăn submit lặp.
5. Trong khi job đang chạy, khoá các input/reset có thể làm sai quan hệ ảnh–kết quả. Bảo đảm mở khoá cả khi thành công và khi lỗi; dùng wrapper UI có xử lý exception/finalization phù hợp phiên bản Gradio. Giữ mỗi lượt gọi pipeline đúng một lần.
6. Lỗi hiển thị câu ngắn, có thể thử lại; log exception ở server bằng logger, không đưa traceback/key/payload lên UI. Xoá kết quả cũ để tránh hiểu là kết quả lần chạy lỗi. Không thay logic fallback của pipeline.

**Đạt khi:** ảnh mới không mang kết quả/phác đồ/session ảnh cũ; lỗi không kẹt nút; bấm lại không tạo job trùng; tải lại ảnh mẫu không phát sinh gọi Gemini.

### UI-03 — Kiểm tra đủ cho demo

Chỉ thêm test cần thiết cho reset và wrapper lỗi/thành công; mock pipeline/session/refine, không gọi API thật. Kiểm tra số output, xoá state cũ, và mỗi click xử lý chỉ gọi pipeline một lần. Không viết test snapshot CSS hoặc kiểm tra mọi nhãn.

Chạy từ repo root (PowerShell):

```powershell
python -m pytest tests/unit/test_frontend_app.py -q --basetemp=scratch/pytest-ui-demo
python -m ruff check frontend/app.py tests/unit/test_frontend_app.py
python -m ruff format --check frontend/app.py tests/unit/test_frontend_app.py
```

Sau đó chạy unit suite offline theo repo: bỏ API key khỏi môi trường process test và giữ nguyên file `.env`; `frontend/app.py` không tự load `.env` khi import. Nếu suite fail ngoài scope, báo riêng baseline failure và bằng chứng, không sửa lan.

```powershell
python -m pytest tests/unit -q --basetemp=scratch/pytest-ui-demo-all
```

Browser smoke ở khoảng 1366×768 và 390×844:
- Trang rỗng, upload/chọn mẫu, chạy thành công, timeline/JSON, chọn/render phiên bản.
- Chưa upload, đổi/xoá ảnh sau khi xử lý, reset, lỗi pipeline được mock, kiểm tra nút phục hồi.
- 1 phiên bản không để radio rỗng/nút render hoạt động sai; nhiều phiên bản vẫn chọn đúng.
- Tab chat: bắt đầu → trả lời → góp ý; đổi ảnh phải bắt đầu phiên mới.
- Ảnh không méo; không tràn ngang; nhãn dễ đọc và nút dùng được bằng bàn phím.

Chụp ít nhất màn hình rỗng và có kết quả. Có thể dùng kết quả mock để kiểm tra UI, nhưng phải ghi rõ trong bàn giao; không coi đó là bằng chứng pipeline thực tế. Nếu không có công cụ browser thì ghi phần kiểm tra trực quan còn thiếu, không tuyên bố đã pass.

### UI-04 — Bàn giao demo

Cập nhật `frontend/README.md` với:
- Lệnh khởi động `python frontend/app.py`, URL `http://localhost:7860`.
- Kịch bản 3–5 phút: giới thiệu → chọn ảnh thiếu sáng → chạy → so sánh → mở lịch sử → giải thích vòng lặp và giới hạn. Chat là phần bổ sung nếu còn thời gian.
- Preflight: thử trước ảnh sẽ trình bày và đo thời gian thực tế; nếu mạng/API không sẵn sàng, nói rõ đang dùng fallback. Không cam kết ảnh cháy sáng/segmentation đã được xử lý tốt.
- Danh sách kiểm tra đã chạy, phần chưa kiểm chứng và file screenshot nếu có.

**Điểm dừng:** giao diện đủ đẹp và luồng chính demo ổn; không thêm dashboard, animation phức tạp, mask overlay, export report, backend endpoint hay chỉnh model. Không commit/push trừ khi được yêu cầu.

## 5. Prompt giao cho model thực thi

> Thực thi `docs/plan_ui_progress_demo.md`, tuần tự UI-01 đến UI-04. Trước khi sửa kiểm tra branch/status và hướng dẫn repo, giữ nguyên mọi thay đổi có sẵn. Chỉ sửa file trong allowlist của plan. Giữ Gradio và hợp đồng callback hiện có; không sửa backend, thuật toán, dependencies hoặc gọi Gemini trong test. Ưu tiên layout sạch và trạng thái UI đúng, không mở rộng tính năng. Chạy kiểm tra theo plan, kiểm tra browser nếu có công cụ, cập nhật README. Kết thúc báo ngắn: file thay đổi, kiểm tra pass/fail, hạn chế chưa xác minh và cách chạy demo. Không commit/push.
