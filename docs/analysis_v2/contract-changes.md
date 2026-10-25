# Analysis v2 — Đề xuất đổi hợp đồng ngoài phạm vi lane

Ghi lại việc lane cần đổi file ngoài phạm vi sở hữu. Lead quyết định khi merge.

## [meta] Metadata: nối `image_metadata` vào luồng phiên tương tác và giao diện Gradio

Trạng thái: **chưa làm**, lý do bên dưới.

1. `src/agent/session.py::start_session` gọi `initial_doctor_state` không truyền `image_metadata`.
   Vì vậy `POST /api/v1/sessions` (`src/api/workspace.py::create_session`) chưa đưa metadata vào
   state. Cần thêm `image_metadata: Optional[ImageMetadata] = None` cho `start_session` rồi truyền
   xuống; route `create_session` đọc bytes upload bằng `read_metadata`. `session.py` không thuộc
   phạm vi lane `meta`.
2. `frontend/app.py`: `gr.Image(type="numpy")` không giữ bytes file, nên không đọc được EXIF.
   Muốn đọc EXIF phải đổi input sang `type="filepath"`, kéo theo `process_job`, `render_chosen_variant`,
   `image.change`, và mẫu ảnh (đang gán đường dẫn). Đổi này lớn và có rủi ro với UI; cần lead
   quyết định (đổi filepath, hay chấp nhận UI chưa có metadata).
3. `src/agent/perception.py` và `src/agent/planner.py` chưa đọc `image_metadata`. Đó là việc của lane
   `agent` (theo brief `meta.md`).
