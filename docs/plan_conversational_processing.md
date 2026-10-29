# Kế hoạch triển khai: hội thoại trước và sau xử lý ảnh

Ngày: 2026-10-30. Trạng thái: kế hoạch thực thi, chưa triển khai mã nguồn.

## 1. Mục tiêu và phạm vi

Chuyển từ chọn intent rồi chạy / gửi feedback rồi chạy ngay sang hai vòng hội thoại:

```text
Upload → Phân tích → Bảng phân tích + giải thích + phương án hiện hành
                         ↕ chat để hỏi, đính chính, điều chỉnh
                    [Bấm Xử lý ảnh]
                             ↓
                   Thực thi → Đánh giá → Trả ảnh + giải thích
                                               ↕ chat góp ý
                                      Phương án xử lý tiếp theo
                                        [Bấm Xử lý lại]
                                               ↓
                                      Thực thi → Trả kết quả → ...
```

Yêu cầu đã thống nhất:

- Chat không tự thực thi xử lý ảnh, kể cả user nhắn “đồng ý”, “làm đi”.
- User có thể xử lý ngay với phương án ban đầu, không bắt buộc chat thêm.
- Bỏ lựa chọn “Theo đề xuất của AI / Mô tả mong muốn khác”.
- Bỏ mục “Đánh giá của bạn”; phản hồi qua chat. Giữ evaluator/reviewer nội bộ.
- AI giải thích sau bảng phân tích và sau khi trả kết quả bằng tiếng Việt tự nhiên.
- Phương án tiếp theo luôn hiển thị rõ chỉnh gì, giữ gì, dùng ảnh nào và vì sao.
- Giữ ảnh gốc; hỗ trợ bắt đầu lại từ ảnh gốc hoặc chỉnh tiếp một kết quả trước đó.
- Không mở rộng toolbox, không dùng generative AI sửa pixel. Ví dụ cắt ảnh trong thảo luận chỉ minh họa business, không phải yêu cầu thêm crop vào toolbox hiện tại.

Câu hướng dẫn cố định, dùng chung qua một hằng số/template UI:

> Bạn có thể tiếp tục trao đổi với AI để làm rõ phân tích và điều chỉnh đề xuất. Khi hài lòng với phương án, hãy bấm “Xử lý ảnh” để áp dụng.

Sau kết quả, dùng cùng template với nút “Xử lý lại”. Hiển thị gần nút; đưa vào lời dẫn đầu phase, không lặp nguyên câu ở mọi lượt chat. Không phụ thuộc model tự sinh đúng câu này.

## 2. Căn cứ từ code hiện tại

| Thành phần | Hiện trạng | Hướng thay đổi |
|---|---|---|
| `src/api/workspace.py::add_analysis_note` | Parse tin nhắn thành edits, lưu draft diagnosis | Mở rộng thành hội thoại: trả lời câu hỏi, cập nhật diagnosis và phương án khi cần |
| `src/agent/intent.py`, `session.py`, `graph.py` | Clarify/answers rồi mới lập plan; MemorySaver trong RAM | Lập phương án trước click; thực thi snapshot đã duyệt, phục hồi được sau restart |
| `src/api/workspace.py::refine_treatment` | Gửi feedback là chạy refine và tạo ảnh mới từ ảnh gốc | Tách đề xuất chỉnh tiếp khỏi thực thi |
| `src/agent/refine.py` | Parse/adjust/render gắn trong luồng refine | Tái sử dụng parser/adjustment qua lớp planning không ghi ảnh |
| `src/api/models.py` | Có Message, Run, Artifact, Treatment.parent_id và version | Bổ sung proposal revision và quan hệ ảnh đầu vào thực tế |
| `web/src/pages/ChatSessionPage.tsx` | Điều phối answers, analysis-notes, refine | Điều phối chat và execute riêng |
| `ChatInputBar.tsx` | Mode sau kết quả gọi refine | Gửi chat ở cả hai phase, không chạy ảnh |
| `DiagnosisBubble.tsx`, `QuestionsCard.tsx` | Có nhánh approach recommend/custom | Thay bằng phương án hiện hành và nút xử lý |
| `RatingBox.tsx`, result/session components | Có UI rating | Bỏ UI và mutation rating khỏi luồng mới |

Kế hoạch này thay thế các phần business liên quan trong `docs/plan_intent_chat.md`; không thực hiện lại toàn bộ kế hoạch cũ. Working tree đang có thay đổi ở nhiều file agent và prompt tests: khi triển khai phải đọc diff và giữ nguyên công việc đang có, không reset/ghi đè.

## 3. Quyết định thiết kế

### 3.1. Tách hội thoại, lập phương án và thực thi

- Conversation service đọc ngữ cảnh, phân loại yêu cầu, trả lời và đề nghị thay đổi.
- Planning service tạo phương án có cấu trúc, validate bằng toolbox/guard hiện có.
- Execution service nhận ID + revision phương án đã lưu; không nhận danh sách action tùy ý từ client.
- Chỉ endpoint execute mới được gọi executor và tạo kết quả ảnh trong luồng mới.
- Câu hỏi giải thích không thay đổi proposal revision; có thể tăng conversation revision khi lưu tin nhắn.
- Lưu assistant reply, diagnosis draft, constraints và proposal mới trong một transaction để UI không nhìn thấy các phiên bản lệch nhau.

### 3.2. Trạng thái bền vững

Giữ session status hiện có để giảm ảnh hưởng tương thích; thêm trạng thái hội thoại/phương án:

- `phase`: `analysis | result`.
- `conversation_status`: `idle | updating | failed`.
- `proposal_status`: `needs_clarification | ready | executing | applied | superseded`.
- `conversation_revision`, `active_proposal_id`, `active_treatment_id`.

Chat ở `awaiting_answers` hoặc `done`; lúc updating/executing thì chặn thao tác xung đột ở cả UI và server. Có thể đọc lịch sử khi đang chạy. Không giữ transaction DB mở xuyên suốt lời gọi model: claim ngắn, xử lý, rồi compare-and-swap khi commit; cần cơ chế thu hồi claim quá hạn sau crash.

DB + artifacts là nguồn dữ liệu phục hồi. Không để khả năng tiếp tục phiên phụ thuộc duy nhất vào MemorySaver hoặc giới hạn 32 session trong RAM.

### 3.3. Đề xuất dữ liệu

Các tên sau là contract dự kiến; hoàn thiện ở T1 trước khi đổi API/UI.

- `ConversationState`: diagnosis hiện hành; yêu cầu/ưu tiên; điều cần giữ; phương án đã bác bỏ; câu hỏi chưa rõ; tóm tắt hội thoại có mốc message; revision.
- `ProcessingProposal` (bản lưu bất biến theo revision): phase; diagnosis revision; source message; ảnh user đang phản hồi; ảnh đầu vào thực thi; mode `rebuild | incremental`; actions có thứ tự; constraints; giải thích chọn ảnh; tóm tắt thay đổi; câu hỏi còn mở; status; phiên bản schema.
- `Run`: liên kết proposal snapshot, ảnh đầu vào, idempotency key, thao tác thực tế, thao tác bị bỏ qua/thất bại, outcome và ảnh đầu ra.
- `Treatment`: giữ version/parent hiện có; bổ sung nguồn ảnh thực tế và proposal/run liên quan. Không suy luận ảnh đầu vào từ `parent_id`: phiên bản đang được phản hồi có thể khác ảnh dùng để chạy.

Giữ `root_original_artifact_id`, `input_artifact_id`, `reviewed_artifact_id` riêng. Ảnh user đang xem trong lịch sử phải được xác định rõ khi gửi phản hồi, không ngầm luôn coi là bản mới nhất.

### 3.4. Ngữ cảnh AI theo phase

| Dữ liệu | Phân tích | Sau kết quả |
|---|---|---|
| Tin nhắn mới + tin gần đây + quyết định cũ liên quan | Có | Có |
| Diagnosis ban đầu và hiện tại | Có | Có, phân biệt diagnosis của từng ảnh |
| Constraints/preserve và câu hỏi chưa rõ | Có | Có |
| Proposal hiện hành | Có | Có, phân biệt chưa chạy và đã chạy |
| Ảnh | Ảnh gốc; crops khi cần kiểm chứng | Ảnh được phản hồi; ảnh gốc/input/phiên bản liên quan khi cần đối chiếu |
| Lịch sử execution và lineage | Chưa có | Actions thực tế, lỗi/skips, evaluation/review, phiên bản liên quan |
| Toolbox, parameter bounds, knowledge và guards | Có | Có |

Không gửi toàn bộ lịch sử và mọi ảnh ở mọi lượt. Context builder ưu tiên state có cấu trúc + recent messages, truy xuất quyết định/ảnh cũ theo tham chiếu. Tóm tắt không được làm mất constraints; mỗi quyết định cần truy về message nguồn. Cache analysis theo artifact/hash và phiên bản analyzer; không dùng nhầm cache mask/measurement của ảnh khác.

Phân biệt nhận định kỹ thuật và sở thích: “tôi thích ảnh tối” đổi mục tiêu, không tự xóa bằng chứng thiếu sáng. “Đó là sương, không phải nhiễu” yêu cầu kiểm chứng hình ảnh khi cần. User có quyền chọn cách sửa; phát biểu của user không tự trở thành số đo đã xác minh.

### 3.5. Chọn ảnh đầu vào và semantics của actions

- `rebuild`: dùng ảnh gốc hoặc ancestor phù hợp; actions chứa đầy đủ những chỉnh sửa còn cần từ ảnh đó, gồm các yêu cầu đã giữ.
- `incremental`: dùng ảnh kết quả được chọn; actions chỉ chứa chỉnh sửa bổ sung, không chạy lại toàn bộ lịch sử trên ảnh đã chỉnh.
- Muốn giảm/loại một hiệu ứng đã gây mất chi tiết: ưu tiên quay về nguồn trước hiệu ứng đó, không hứa đảo ngược ảnh đã mất thông tin.
- Muốn giữ kết quả và bổ sung thao tác: có thể chọn incremental sau khi kiểm tra tác động chồng và preserve.
- Nếu yêu cầu hoặc lựa chọn nguồn còn mơ hồ: hỏi làm rõ; proposal chưa ready. Không âm thầm đổi nguồn lúc click.
- Nếu không có thay đổi sau kết quả: không tạo lượt xử lý rỗng. Hướng dẫn user góp ý thêm; proposal applied không được chạy lại ngoài retry idempotent.
- Kiểm tra artifact thuộc cùng session, còn tồn tại, đúng kích thước/metadata; dựng mask và số đo trên đúng input.

### 3.6. Quan hệ với vòng lặp tự động hiện tại

Đề xuất cho luồng tương tác mới: một click thực thi đúng actions đã duyệt, sau đó đánh giá/review và có thể rollback khi cần. Nếu cần phương án khác, tạo đề xuất mới để user xem và bấm tiếp; không tự re-plan rồi thực thi một phương án chưa hiển thị.

Đây là lựa chọn triển khai để bảo đảm “phương án hiển thị = phương án chạy”. Giữ pipeline tự động cũ cho batch/legacy nếu còn consumer, qua chế độ rõ ràng. Không đơn giản đổi `max_iterations` rồi làm mất kiểm chứng sau xử lý. Evaluator so sánh input→output để đánh giá lượt này, và original→output khi cần kiểm tra ảnh hưởng tích lũy; reference metrics chỉ dùng đúng ground truth theo contract hiện có.

## 4. Các ticket triển khai

### T1 — Contract, schema và migration

Files: `src/agent/state.py`, `src/api/models.py`, `workspace_schemas.py`, `db.py`; module schema mới nếu cần.

- Chốt enums và Pydantic models của conversation/proposal/run.
- Thêm lưu proposal revisions, state hội thoại, input lineage, execution claim/idempotency.
- Migration additive, giữ messages/treatments/ratings cũ; không xóa dữ liệu rating.
- Phiên cũ dựng conversation state từ diagnosis/treatment/actions thực tế. Không đủ dữ liệu thì đánh dấu cần lập lại phương án, không bịa proposal sẵn sàng.
- Tách provenance của diagnosis gốc và diagnosis điều chỉnh.

Nghiệm thu: DB mới/cũ đều mở được; reload/restart giữ state; revision bất biến; không thể liên kết ảnh khác session.

### T2 — Context builder và hội thoại có cấu trúc

Files: thêm `src/agent/conversation.py`, `conversation_context.py` (tên dự kiến); dùng lại `user_analysis.py`, `intent.py`, planner, VLM adapter.

- Phân loại `explain | modify | correct_analysis | clarify | undo` bằng đầu ra schema.
- Model trả reply tự nhiên, proposed changes, câu hỏi mở và yêu cầu kiểm chứng; server validate trước khi cập nhật state.
- Xử lý tham chiếu “cái đó”, phủ định, yêu cầu mới thay thế yêu cầu cũ, câu có cả hỏi lẫn đề nghị sửa.
- Tái dùng rule parser khi phù hợp; không coi mọi tin không ra edit là không hiểu.
- Analysis ban đầu sinh explanation + proposal đã validate trước khi cho phép xử lý.
- Lỗi model: báo chưa cập nhật được, giữ phương án trước, không dựng lời giải thích giả hoặc làm mất tin user; hỗ trợ retry không nhân đôi message.
- Từ chối action ngoài toolbox bằng lời giải thích khả năng hiện có. Q&A về một thao tác không được tự trở thành action.

Nghiệm thu: hỏi không đổi proposal; góp ý được tích lũy; constraints không mất qua nhiều lượt; correction chỉ đo/xem lại khi cần; tests offline với model mock.

### T3 — Planning xử lý lại và lựa chọn nguồn ảnh

Files: `src/agent/refine.py`, `planner.py`, `conversation_context.py`, proposal service mới.

- Tách hàm parse/adjust khỏi render trong refine.
- Chọn source cùng lúc tạo actions; lưu reason và provenance.
- Áp dụng validation, preserve/contraindication guards phù hợp; không kế thừa việc bỏ guard vô điều kiện của refine cũ.
- Khi user chủ động thay preserve, cập nhật constraint rõ ràng trước khi validate plan mới.
- Giữ thứ tự actions có ý nghĩa; kiểm tra planner không tự sắp xếp lại recipe gây khác kết quả.

Nghiệm thu: “da bệt nhưng sáng ổn” tạo rebuild giữ tăng sáng và giảm làm mịn; “ấm hơn chút” bổ sung vào proposal đang chờ; incremental không lặp gamma/denoise cũ; source quyết định có căn cứ, không hard-code mọi review về ảnh gốc.

### T4 — Execution từ proposal snapshot

Files: `src/agent/graph.py`, `session.py`, `executor.py` nếu cần; `workspace_service.py`.

- Thêm entry point chạy proposal đã validate, tách khỏi đường tự lập plan khi resume answers.
- Tách root original khỏi execution input trong state để measurement/mask/rollback không dùng sai ảnh.
- Lưu snapshot trước chạy, actions requested/executed/skipped, input/output artifacts và evaluation đúng ảnh được trả.
- Không tự đổi sang proposal mới trong quá trình chạy; kết quả cần cải thiện thì trả chat kèm đề xuất, chờ click.
- Nếu lỗi hoặc rollback: giữ kết quả trước còn dùng được; giải thích trung thực, không nói đã áp dụng thao tác bị skip.
- Result explanation dựa trên execution + evaluation/review của ảnh cuối. Review thiếu/không chắc thì nêu giới hạn, không khẳng định đã sửa thành công.

Nghiệm thu: click chạy đúng source/actions/revision; restart vẫn thực thi được proposal; failure không làm mất ảnh cũ; đánh giá gắn đúng output sau rollback.

### T5 — API chat và execute độc lập

Files: `src/api/workspace.py`, `workspace_service.py`, `workspace_schemas.py`; xem các route legacy trong API trước khi chuyển đổi.

Contract đề xuất:

- `POST /sessions/{id}/chat`: `message`, `client_message_id`, `expected_conversation_revision`, `reviewed_treatment_id` khi phản hồi kết quả. Trả reply + conversation state + proposal hiện hành; tuyệt đối không render.
- `POST /sessions/{id}/execute`: `proposal_id`, `proposal_revision`, `expected_conversation_revision`, `idempotency_key`. Server tự đọc input/actions từ proposal.
- `GET /sessions/{id}`: thêm conversation/proposal/source summary, readiness và trạng thái job; tái dùng cơ chế refresh hiện có trước khi cân nhắc streaming.

Yêu cầu:

- Compare-and-swap/claim server để chặn chat và execute chạy đua; proposal cũ trả 409 và state mới.
- Retry cùng idempotency key trả cùng run; cùng key khác payload trả conflict. DB bảo đảm một execution claim đang hoạt động cho session.
- Xử lý chuyển active treatment: invalidate/rebase proposal và hiển thị lại, không mang nhầm proposal qua phiên bản khác.
- Migrate web sang API mới. Các route `/answers`, `/analysis-notes`, `/refine` phải được kiểm kê consumer; với session mới chuyển adapter về chat/execute có revision hoặc trả lỗi deprecated rõ ràng, không để đường chat cũ lách sang render ngay.
- Giữ lịch sử phiên cũ đọc được; kiểm tra Gradio và API stateless để legacy behavior được phân tách, không thay đổi ngầm.

Nghiệm thu: chat không tạo ảnh/run xử lý; duplicate click chỉ một execution; multi-tab conflict; thiếu source/khác owner bị chặn; chat lỗi không làm applied phương án cũ.

### T6 — UI hai phase hội thoại

Files: `web/src/pages/ChatSessionPage.tsx`, `components/chat/ChatInputBar.tsx`, `ChatContainer.tsx`, `bubbles/DiagnosisBubble.tsx`, `bubbles/ResultBubble.tsx`, `components/session/QuestionsCard.tsx`, `ResultCard.tsx`, `components/result/RatingBox.tsx`, `components/sidepane/`, `api/types.ts`, `api/workspace.ts`.

- ChatInputBar gửi chat ở analysis và result; không gọi refine trực tiếp kể cả nhánh fallback nội bộ của component.
- Thêm ProposalCard chung: chỉnh gì, giữ gì, input/source reason, câu hỏi mở, nút xử lý và hướng dẫn cố định.
- Bỏ approach recommend/custom. Chuyển câu hỏi làm rõ thật sự sang chat; không bỏ khả năng hỏi user.
- Bảng diagnosis hiển thị state hiện hành; messages lịch sử giữ snapshot lịch sử, không viết lại như AI đã biết trước.
- Hiển thị assistant explanation ngay sau diagnosis/result tương ứng.
- Bỏ RatingBox và callback/mutation liên quan; giữ chỉ số đánh giá kỹ thuật, reviewer nội bộ và dữ liệu rating lịch sử.
- Disable nút khi chưa ready/updating/executing; nếu còn text chưa gửi, yêu cầu gửi hoặc bỏ nội dung trước khi chạy để tránh bỏ sót mong muốn.
- Bảo toàn text khi gửi lỗi, báo 409 dễ hiểu, hỗ trợ reload. Xem ảnh lịch sử phải cho biết đang góp ý phiên bản nào.
- Lịch sử kết quả ghi nguồn xử lý và những gì thực sự áp dụng; không gán kết quả đánh giá của v1 cho mọi bản refine.
- Kiểm tra frontend Gradio: nếu còn duy trì tương tác, nối cùng service và hành vi click; không để web và Gradio tuyên bố cùng business nhưng thực thi khác nhau.

Nghiệm thu: không còn nút recommend/custom hoặc mục rating trong flow mới; Enter gửi chat không render; nút xử lý rõ ràng ở cả hai phase; desktop/mobile đọc được proposal và source.

### T7 — Kiểm thử, tài liệu và chuyển đổi

Backend tests hiện có cần cập nhật: `test_api_analysis_chat.py`, `test_api_workspace.py`, `test_api_sessions.py`, `test_api_refine.py`, `test_agent_intent.py`, `test_agent_refine.py`; bổ sung tests conversation/proposal/context/execution/recovery.

Web tests: `chat.test.tsx`, `flow.test.tsx`, `refine.test.tsx`, `rating.test.tsx`, `result.test.tsx`, `ChatInputBar.test.tsx`, studio compatibility và fixtures. Thay các assertion cũ “gửi feedback tạo ảnh ngay” bằng “chat chỉ tạo proposal; click mới tạo ảnh”.

Ma trận acceptance bắt buộc:

1. Analyze → có bảng + explanation + proposal → click ngay được nếu ready.
2. Hỏi lý do → chỉ có câu trả lời; proposal không thay đổi.
3. Góp ý nhiều lượt → cập nhật tích lũy, không mất điều cần giữ.
4. Đính chính nội dung ảnh → kiểm chứng phù hợp, không tự đồng nhất sở thích với diagnosis.
5. User nhắn “đồng ý/xử lý đi” → vẫn chờ nút.
6. Sau kết quả chat nhiều lượt → không có ảnh mới trước click.
7. Rebuild giữ chỉnh sửa đã chấp nhận; incremental không áp lại thao tác cũ.
8. Review một bản cũ / đổi active version → đúng source và proposal không bị dùng nhầm.
9. Stale revision, hai tab, double-click, retry request → không chạy trùng/sai plan.
10. Reload/restart/crash lúc updating hoặc executing → state nhất quán, giữ ảnh và tin nhắn.
11. Model lỗi, JSON lỗi, action không hỗ trợ, feedback mơ hồ → lời giải thích/clarification, không render ngầm.
12. Executor skip/fail/rollback → giải thích đúng ảnh cuối, không báo thành công giả.
13. Phiên cũ và DB migration → đọc lịch sử được, không mất ratings/treatments.

Kiểm tra bằng ảnh tổng hợp và model mock trước; kiểm tra tay vài ca ảnh thật đã có trong repo khi môi trường model sẵn sàng. Không yêu cầu key/network cho unit tests.

Lệnh nghiệm thu khi triển khai:

```text
ruff check src/ tests/
pytest tests/unit/ -v
cd web
npm run test -- --run
npm run typecheck
npm run build
```

Chạy integration tests liên quan pipeline/API và browser flow mới; format các file Python đã sửa. Cập nhật `docs/interfaces.md`, `src/agent/README.md`, `web/README.md` và hướng dẫn architecture liên quan; đánh dấu phần luồng cũ trong plan trước đã được thay thế. Nếu chỉnh tài liệu hướng dẫn agent, đọc skill viết tài liệu agent trước khi sửa.

## 5. Thứ tự giao việc và mốc hoàn thành

Thứ tự: **T1 → T2 → T3 → T4 → T5 → T6 → T7**. T1 chốt contract trước; T2/T3 có thể dùng tests độc lập nhưng chưa nối UI vào API cũ gây auto-render. Mỗi ticket thành một thay đổi review được theo workflow repo; không yêu cầu tạo agent song song.

- Mốc A, sau T1–T2: hội thoại tạo/cập nhật proposal có trạng thái và tests, chưa chuyển production flow.
- Mốc B, sau T3–T5: backend hỗ trợ cả hai phase, chọn nguồn và click-to-execute an toàn.
- Mốc C, sau T6–T7: UI hoàn chỉnh, migration và end-to-end đạt acceptance; mới bật luồng mới mặc định.

Điểm rủi ro trọng tâm: semantics của `original_image` đang dùng nhiều nơi, coupling giữa plan và execute trong graph, memory-only checkpoint, thao tác thực tế có thể khác request do fallback/skip, và route refine cũ còn khả năng render ngay. Phải xử lý các điểm này trong backend trước khi coi việc bỏ nút/đổi lời nhắn trên UI là hoàn thành.
