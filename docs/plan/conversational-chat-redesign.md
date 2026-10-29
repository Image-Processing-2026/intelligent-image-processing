# Kế hoạch Triển khai: Thiết kế lại Conversational AI Image Doctor (Chat-First + Inspection Side Pane)

> **Mục tiêu:** Chuyển đổi toàn diện hệ thống full-stack từ dạng form/studio chia tab (Phase Studio) sang trải nghiệm **hội thoại tự nhiên tương tự ChatGPT/Claude**, tích hợp thanh chat đa phương thức (Ảnh + Prompt ban đầu) và bảng soi kỹ thuật mở rộng (**Inspection Side Pane**) ở bên phải, chuẩn hoá 100% với kiến trúc backend và vòng đời phiên hiện có.

---

## 1. Bối cảnh & Nguyên tắc cốt lõi

### 1.1 Vấn đề của luồng Studio chia tab hiện tại
- **Chia pha rời rạc:** `SessionPage.tsx` tách biệt thành 3 tab cứng (*Phân tích* $\rightarrow$ *Xử lý* $\rightarrow$ *Kết quả*), làm gãy mạch ngữ cảnh hội thoại.
- **Tạo phiên thiếu ngữ cảnh ban đầu:** `NewSessionPage` chỉ cho upload ảnh, người dùng không thể gửi kèm mong muốn/ngữ cảnh (ví dụ: *"Người trong ảnh bị tối, hãy làm sáng mặt nhưng giữ nguyên nền trời"*).
- **Thao tác bị phân tán:** Khung chat bổ sung phân tích, form câu hỏi trắc nghiệm và card kết quả nằm ở các vị trí khác nhau.
- **Tràn ngập số liệu:** Dữ liệu kỹ thuật chuyên sâu (Module 1, mặt nạ Module 2) chiếm diện tích lớn, gây ngợp cho người dùng phổ thông.

### 1.2 Mô hình tương tác Chat-First
1. **Dòng chảy hội thoại tuyến tính (Linear Chat Thread):** Tất cả các bước (ảnh gốc + prompt $\rightarrow$ chẩn đoán $\rightarrow$ tương tác sửa phân tích $\rightarrow$ kết quả $\rightarrow$ refine góp ý) là các bong bóng tin nhắn (Message Bubbles) hiển thị theo trình tự thời gian.
2. **Thanh nhập liệu đa phương thức (ChatInputBar):** Hỗ trợ kéo thả, chọn file và **Paste trực tiếp từ clipboard (Ctrl+V)** (tối đa 1 ảnh, kiểm tra MIME/kích thước client) kèm ô nhập text ($\le 1000$ ký tự) và mục *Tùy chọn nâng cao* (chọn ảnh mẫu, ảnh chuẩn `ground_truth` theo ADR-002, số vòng lặp `max_iterations`).
3. **Phản hồi tự nhiên + Đề xuất hành động:** AI chẩn đoán ngắn gọn, kèm câu hỏi trắc nghiệm (nếu có), các pill gợi ý nhanh, và nút **"Bắt đầu xử lý ngay 🚀"**.
4. **Bảng soi kỹ thuật tách biệt (Inspection Side Pane):** Hiển thị ở panel bên phải (tỷ lệ 65/35 trên desktop, drawer trượt trên mobile) khi bấm *"Xem chi tiết"*, chia làm các tab chuyên biệt.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Top Bar: AI Image Doctor  |  Status  |  [Toggle Bảng Chi Tiết]          │
├─────────────────────────────────────────┬──────────────────────────────┤
│ KHUNG CHAT CHÍNH (65%)                  │ SIDE PANE KỸ THUẬT (35%)     │
│                                         │                              │
│ [User] [Ảnh đính kèm]                   │ [Tabs: Mặt nạ | Số đo | Lý do│
│ "Người trong ảnh bị tối..."             │  | Thực thi]                 │
│                                         │ ┌──────────────────────────┐ │
│ [AI] "Đã nhận diện: Chủ thể bị ngược    │ │ Preview Mask (Oval face) │ │
│ sáng (L* = 18.4). Đề xuất:..."          │ │ Backend: MediaPipe       │ │
│ • [Câu hỏi làm rõ nếu có]               │ ├──────────────────────────┤ │
│ [Xem chi tiết 📊] [Xử lý ngay 🚀]       │ │ Module 1:                │ │
│                                         │ │ L* mặt: 18.4  (Sev 3)    │ │
│ [User] "Chỉ làm sáng mặt thôi nhé"      │ │ L* nền: 64.2  (Optimal)  │ │
│                                         │ ├──────────────────────────┤ │
│ [AI] [Kết quả: Before/After Slider]     │ │ Lý do & Căn cứ:          │ │
│ "Đã xử lý xong, outcome: ACCEPTED"      │ │ Khóa nền theo ý user     │ │
│ [Chấp nhận ✅] [Góp ý / Tinh chỉnh 🔄]  │ └──────────────────────────┘ │
├─────────────────────────────────────────┴──────────────────────────────┤
│ Thanh Input: [📷 Paste/Upload] [Nhập yêu cầu...] [⚙ Nâng cao] [Gửi ➔] │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Kiến trúc & Ánh xạ Dữ liệu (Chuẩn hoá Backend)

### 2.1 Thứ tự thực thi chuẩn tại Backend (`src/api/workspace.py`)

Khi gọi `POST /api/v1/sessions` kèm `user_prompt`:
1. **Lưu phiên & chạy chẩn đoán:** Gọi `start_session(..., prompt=...)` $\rightarrow$ `record_analyze()` để tạo chẩn đoán ban đầu (`diagnosis`).
2. **Lưu tin nhắn người dùng:** Lưu message đầu tiên vào bảng `messages` với role `user`, kind `user_prompt`, data chứa thông tin ảnh và text prompt ($\le 1000$ ký tự).
3. **Parse prompt thành `draft_edits`:**
   - Sau khi đã có `diagnosis`, gọi `parse_edits(user_prompt, diagnosis)` (luật trước, Gemini sau; giới hạn bởi `MAX_DRAFT_EDITS`).
   - Ghi vào `session.draft_edits`.
   - Thêm message `analysis_update` **ngay sau message `diagnosis`** để `_unparsed_notes` hoạt động chính xác (chỉ gom các note đứng sau lần chẩn đoán cuối cùng vào Plan prompt).
4. **Bảo toàn prompt khi `clarify_again`:** Khi người dùng bấm "Hỏi lại" hoặc khi chẩn đoán lại, `clarify_again` sau khi reset và chạy lại chẩn đoán sẽ tự động đọc lại message `user_prompt` đã lưu để re-parse vào `draft_edits`, tránh mất ngữ cảnh ban đầu. Nếu `start_session` lỗi (`failed`), prompt vẫn được lưu an toàn trong session.

### 2.2 Ánh xạ dữ liệu Side Pane vào `SessionDetailOut` có sẵn (Không trùng lặp)

Không thêm các trường thừa (`inspection_data`, `technical_metrics`), mà ánh xạ trực tiếp từ các trường backend đã trả về:

| Tab Side Pane | Nguồn dữ liệu từ `SessionDetailOut` | Nội dung hiển thị |
|---|---|---|
| **Mặt nạ (Masks)** | `region_overlay_url`, `regions[]`, `draft_diagnosis.preserve` | Ảnh overlay trực quan, danh sách vùng kèm nhãn backend (MediaPipe/SAM), độ tin cậy và vùng cần bảo vệ. |
| **Số đo (Metrics)** | `diagnosis.measurement`, `diagnosis.region_stats`, `diagnosis.region_metrics` | Độ sáng L* từng vùng, mức chênh lệch L* gap, độ nhiễu σ (PCA). *Chỉ hiển thị các số liệu có thật trong payload, không sinh thêm.* |
| **Lý do (Reasoning)** | `process.history[].plan.reasoning`, `diagnosis.fusion_notes`, `process.treatment[].notes` | Căn cứ suy luận của hệ thống, ghi chú dung hợp (fusion notes) và các quy tắc/card tri thức đã áp dụng. |
| **Thực thi & Diff** | `process.history[].executed`, `process.artifacts.executed_overlay`, `process.history[].defect_status`, `side_effects` | Danh sách thao tác thực tế (`actual_mode`, `strength_scale`, `feather`), overlay mặt nạ thực thi, mức cải thiện từng lỗi và tác dụng phụ. |

### 2.3 Ánh xạ nút thao tác & Vòng đời phiên
- **Nút "Bắt đầu xử lý ngay 🚀":** Gọi `POST /api/v1/sessions/{sid}/answers` gửi kèm `answers` (lấy theo `question.default` nếu user không đổi) và danh sách `draft_edits`.
- **Nút "Chấp nhận kết quả ✅":** Gọi `PUT /api/v1/sessions/{sid}/ratings` với `{rating: "good"}` và kích hoạt tải ảnh kết quả về máy.
- **Nút "Xử lý lại / Tinh chỉnh 🔄":** Do kiến trúc pipeline chỉ chạy một lần mỗi session (`_has_finished_process`), nút này cung cấp 2 lựa chọn rõ ràng:
  1. *Góp ý tinh chỉnh (Refine):* Gọi `POST /api/v1/sessions/{sid}/refine` với ý kiến điều chỉnh của người dùng (tạo version `Treatment` mới).
  2. *Tạo phiên mới:* Sao chép ảnh gốc và mở một phiên chat mới với các thông số tùy chỉnh lại.

---

## 3. Danh sách Polymorphic Messages trong Khung Chat

Khung chat hiển thị danh sách `session.messages[]` dựa theo `message.kind`:

| `message.kind` | Role | Component hiển thị | Chức năng tương tác |
|---|---|---|---|
| `user_prompt` *(mới)* | `user` | `UserPromptBubble` | Ảnh thumbnail gốc + nội dung text prompt ban đầu. |
| `diagnosis` | `assistant` | `DiagnosisBubble` | Tóm tắt chẩn đoán tự nhiên + tích hợp câu hỏi làm rõ (`questions`) + danh sách chip `draft_edits` (có nút xóa gọi `deleteAnalysisEdit`) + các pill gợi ý nhanh + nút "Bắt đầu xử lý". |
| `analysis_note` | `user` | `UserBubble` | Nội dung tin nhắn người dùng gõ bổ sung phân tích. |
| `analysis_update` | `assistant` | `AnalysisUpdateNote` | Phản hồi của hệ thống về các edit vừa được nhận diện hoặc thông báo chưa hiểu. |
| `questions` | `assistant` | *(Được gộp hiển thị trong `DiagnosisBubble`)* | Tùy chọn radio trắc nghiệm (mặc định chọn `default`). |
| `answers` | `user` | `AnswersBubble` | Bản tóm tắt các câu trả lời và ghi chú mà user đã chốt khi bấm xử lý. |
| `result` | `assistant` | `ResultBubble` | `CompareSlider` (Trước/Sau) + Outcome badge (`ACCEPTED`/`UNVERIFIED`) + Nút Chấp nhận & Góp ý. |
| `feedback` | `user` | `UserBubble` | Tin nhắn góp ý tinh chỉnh sau khi có kết quả. |
| `refine_result` | `assistant` | `RefineResultNote` | Kết quả sau khi refine kèm CompareSlider cập nhật. |
| `notice` | `assistant` | `NoticeCard` | Cảnh báo hoặc thông báo lỗi từ hệ thống. |
| *(Kind lạ / fallback)* | Any | `FallbackBubble` | Hiển thị nội dung dạng text an toàn để không bao giờ bị crash giao diện. |

---

## 4. Kế hoạch Thực hiện chi tiết (Phases & Tickets)

### Chuẩn bị trước khi thực hiện
> **Bước 0: Lưu checkpoint code hiện tại:** Commit toàn bộ các thay đổi chưa commit trên branch `feat/api-analysis-chat` (gồm các file `components/studio/*` và test liên quan) với commit message: `chore: checkpoint current studio state before conversational redesign`. Điều này đảm bảo có điểm rollback an toàn 100%.

---

### Phase 1: Nâng cấp Backend (`src/api/`) (1 ngày)

- [x] **Ticket BE-1: Tiếp nhận `user_prompt` và lưu message khởi tạo**
  - **File:** `src/api/workspace.py`, `src/api/workspace_schemas.py`.
  - **Nội dung:**
    - Cập nhật schema `SessionCreateRequest` / form multipart hỗ trợ `user_prompt: Optional[str] = Form(None)`.
    - Sau `record_analyze()` (khi đã có `diagnosis`), gọi `parse_edits(user_prompt, diagnosis)`.
    - Lưu message `user_prompt`, ghi `draft_edits`, và lưu message `analysis_update` ngay sau `diagnosis`.
    - Cập nhật `clarify_again` để re-parse lại `user_prompt` đã lưu sau khi làm mới chẩn đoán.
  - **Kiểm thử:** Viết unit test trong `tests/unit/test_api_workspace.py` gửi ảnh + text, xác nhận thứ tự message đúng và `draft_edits` được nạp sẵn.

- [x] **Ticket BE-2: Cập nhật tài liệu hợp đồng `docs/interfaces.md`**
  - Ghi nhận `user_prompt` trong `SessionCreateRequest`.
  - Bổ sung định nghĩa `user_prompt` vào danh sách `MessageKind`.

---

### Phase 2: Xây dựng Thanh nhập liệu đa phương thức (`ChatInputBar`) (1 ngày)

- [x] **Ticket FE-1: Xây dựng `ChatInputBar` với Paste & Kéo thả**
  - **File:** `web/src/components/chat/ChatInputBar.tsx`.
  - **Nội dung:**
    - Hỗ trợ sự kiện `onPaste`: nhận tối đa 1 ảnh từ clipboard, kiểm tra MIME type (`image/png`, `image/jpeg`, `image/webp`) và giới hạn dung lượng ($\le 20MB$).
    - Hiển thị Thumbnail Chip của ảnh đính kèm (có nút ✕ để gỡ ảnh).
    - Textarea hỗ trợ tự co giãn, giới hạn $\le 1000$ ký tự, gửi bằng phím Enter (Shift+Enter để xuống dòng).
    - Popover/Modal **"Tùy chọn nâng cao"**: hỗ trợ chọn ảnh mẫu (`sample`), ảnh chuẩn (`ground_truth`), và thanh chỉnh `max_iterations` (1-5).
  - **Nghiệp vụ gửi:**
    - Ở màn hình khởi đầu (`/`): gửi gọi `createSession({ file, user_prompt, ... })` $\rightarrow$ chuyển hướng sang `/s/:sid`.
    - Trong phiên đang ở `awaiting_answers`: gửi gọi `addAnalysisNote(sid, text)`.
    - Trong phiên đã `done`: gửi gọi `refineTreatment(sid, text)`.

---

### Phase 3: Dòng chảy tin nhắn Chat & Tương tác (1.5 ngày)

- [x] **Ticket FE-2: Polymorphic Message Renderer & Container**
  - **File:** `web/src/components/chat/ChatContainer.tsx`, `web/src/components/chat/bubbles/*`.
  - **Nội dung:** Xây dựng danh sách tin nhắn scroll tự động xuống đáy khi có message mới, render chuẩn xác 10 kind message theo bảng ở Mục 3.

- [x] **Ticket FE-3: `DiagnosisBubble` (Tích hợp trắc nghiệm, Edit chips & Pill actions)**
  - Hiển thị tóm tắt chẩn đoán tự nhiên.
  - Render các câu hỏi clarification (nếu có câu hỏi) dạng radio, mặc định chọn `default`.
  - Render danh sách chip `draft_edits` hiện có kèm nút ✕ (gọi `deleteAnalysisEdit`).
  - Các pill gợi ý nhanh (bấm pill sẽ gọi `addAnalysisNote`).
  - Nút bấm **"Xem chi tiết chẩn đoán 📊"** (mở Side Pane đúng tab).
  - Nút chính **"Bắt đầu xử lý ngay 🚀"** (gọi `answerQuestions` với answers mặc định/đã chọn).

- [x] **Ticket FE-4: `ResultBubble` (Before/After Slider & Outcome Action)**
  - Tích hợp `CompareSlider` trực tiếp trong bong bóng kết quả.
  - Hiển thị badge Outcome (`ACCEPTED`, `UNVERIFIED`,...).
  - Nút **"Chấp nhận kết quả ✅"**: gọi `saveRating("good")` và kích hoạt tải ảnh.
  - Nút **"Góp ý tinh chỉnh 🔄"**: focus vào ô chat để user nhập feedback.
  - Nút **"Xem thông số thực thi 🔍"**: mở Side Pane tab Thực thi & Diff.

---

### Phase 4: Bảng soi kỹ thuật bên phải (`InspectionSidePane`) (1.5 ngày)

- [x] **Ticket FE-5: Khung Side Pane Collapsible & Responsive**
  - **File:** `web/src/components/sidepane/InspectionSidePane.tsx`.
  - **Nội dung:**
    - Desktop: chiếm 35% chiều rộng cạnh khung chat, có nút đóng/mở trên Top Bar.
    - Mobile ($< 1024px$): chuyển thành Sheet/Drawer trượt overlay đè lên màn hình.
    - 4 Tab: **Mặt nạ (Masks)**, **Số đo (Metrics)**, **Lý do (Reasoning)**, **Thực thi (Executed Diff)**.

- [x] **Ticket FE-6: Nội dung 4 Tab chuyên sâu**
  - Ánh xạ chính xác các trường dữ liệu thực tế từ `SessionDetailOut` theo bảng ở Mục 2.2.
  - Không sinh thêm dữ liệu giả lập không có trong payload.

---

### Phase 5: Tích hợp Router, Dọn dẹp & Kiểm thử (1 ngày)

- [x] **Ticket INT-1: Tích hợp Router & Trang Chat chính**
  - Route `/`: Hiển thị giao diện New Chat (khung chat trống với ô input sẵn sàng).
  - Route `/s/:sid`: Hiển thị `ChatSessionPage` với toàn bộ dòng chảy tin nhắn và Side Pane.
  - Đảm bảo tương thích ngược: các session cũ không có `user_prompt` sẽ lấy ảnh gốc hiển thị bình thường.
  - Dọn dẹp các import studio cũ không còn dùng trong trang chính.

- [x] **Ticket TEST-1: Kiểm thử hồi quy toàn diện**
  - Frontend:
    - Chạy `npx vitest run` (không dùng `npm test` để tránh treo watch mode).
    - Chạy `npm run typecheck` và `npm run build`.
  - Backend:
    - Chạy `ruff check src/ tests/` và `pytest tests/unit/ -q`.
  - Kiểm tra thủ công: Paste ảnh từ clipboard, nhập prompt, bấm xử lý, xem Side Pane, chấp nhận kết quả.

---

## 5. Rủi ro Kỹ thuật & Biện pháp Xử lý

1. **Độ trễ khi tạo phiên (Latency):**
   - *Nguyên nhân:* Khi `user_prompt` không khớp luật regex tiếng Việt, hệ thống sẽ gọi thêm 1 lượt Gemini VLM để parse sang `DiagnosisEdit`.
   - *Giải pháp:* Hiển thị skeleton / loading state "AI đang phân tích ảnh và yêu cầu của bạn..." rõ ràng ngay khi user bấm gửi.
2. **Xử lý sự kiện Paste từ Clipboard:**
   - *Nguyên nhân:* Người dùng có thể paste text chứa đường dẫn ảnh hoặc paste nhiều ảnh cùng lúc.
   - *Giải pháp:* Chỉ nhận item đầu tiên có `type.startsWith('image/')`. Nếu paste text thì giữ nguyên vào textarea. Giới hạn dung lượng tối đa 20MB.
3. **Quy ước mã nguồn:**
   - Docstring và chú thích bằng tiếng Việt có dấu.
   - Identifier, log, exception và commit message bằng tiếng Anh.
   - Tuân thủ nghiêm ngặt ADR-001 (không dùng generative pixel).
