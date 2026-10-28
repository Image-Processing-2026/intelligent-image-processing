# Sửa các vấn đề audit + Chat Studio — tickets (deadline 3 ngày)

Ngày lập: 27/10/2026, cập nhật ưu tiên cùng ngày. Nguồn: [rà soát ca chân dung ngược sáng](../reports/2026-10-27-backlit-portrait-pipeline-review.md),
[audit đánh giá đầu ra](../reports/2026-10-27-output-evaluation-audit-notes.md).

**Ưu tiên:** (1) sửa các vấn đề đã audit (R-xx, 3 ngày); (2) luồng Chat Studio (C-xx) chỉ làm khi còn thời gian hoặc sau deadline.

> Mục tiêu đợt R: ca ngược sáng không còn bị nâng sáng cả người khi định vị được mặt; mask và tham số thực thi
> truy vết được; số đo không báo "đã hết lỗi" khi vùng con mâu thuẫn; kết quả chỉ được tính là đạt khi review ACCEPT.
> Các tham số mới trong đợt này là **tạm thời, chưa hiệu chỉnh**. Không chứng minh được "tóc luôn giữ đen" trước BL-10.

## 1. Lịch và đường cắt

| Ngày | Tickets | Finding |
|---|---|---|
| 1 | R-01, R-02, R-03, R-04 (bắt đầu) | F6, F9, F11, F12, F13, session A |
| 2 | R-04, R-05, R-06, R-07 | F1, F2, F3, F8, F10 (một phần) |
| 3 | R-08, R-09, R-10 | F4, F5, F7 |

Đường găng: R-03 → R-04 → R-05 → R-09; R-01 → R-08.
**Nếu trễ, cắt theo thứ tự:** nhánh REJECT của R-08 (chỉ giữ ACCEPT/UNCERTAIN/REVISE→NOT_MET) → phần chroma của R-07 → QA mask nâng cao của R-04.
Không cắt R-01, R-04 phần fallback, R-05, R-06.

## 2. Tickets ưu tiên (R)

### R-01 — Outcome trung thực + evaluation đúng ảnh `agent` `api` · ngày 1 · ~4h · F6, F11, session A
- Model `Outcome {status, accepted_by, candidate_sha256, reason}` (`state.py`), xuất qua `record_process` (`run.result.outcome`).
  Status: `ACCEPTED` | `UNVERIFIED` | `NOT_MET` | `KEPT_ORIGINAL` | `NO_CHANGE`. Chỉ `ACCEPTED` tương ứng `final_decision=SHIP`
  (synthetic đạt reference: `accepted_by=reference`). Hết vòng chỉ có nghĩa là dừng.
- `KEPT_ORIGINAL` xác định bằng so pixel với ảnh gốc khi intent có yêu cầu sửa.
- Sau rollback: tính lại `evaluation_result` cho ảnh được trả (`graph.py:541-574`).
- `_side_effects` (`graph.py:342`) trả kèm trạng thái `ok|unavailable`; lưu vào `HistoryItem`.
- Không đổi `EvaluationResult` của Module 1; agent/API ngừng đọc `visual_plausibility_passed`/`vlm_feedback`.
  Web/Gradio hiển thị `outcome` (`types.ts`, `StatusBadge`, `labels.ts`, `ResultCard`, `frontend/app.py`).
- Rà 29 assert `"SHIP"` (13 file test): ảnh thật không có review sẽ ra `UNVERIFIED`.
- **Nghiệm thu:** replay session A → `KEPT_ORIGINAL`, `final_evaluation` thuộc ảnh gốc; exception trong side-effect
  check → `unavailable`; không đường nào cho `SHIP` trên ảnh thật mà thiếu review ACCEPT.

### R-02 — Truy vết mask và môi trường `agent` `region` `api` · ngày 1 · ~3h · F9, F12, F13
- `execute_plan` ghi cho mỗi action: requested/actual `face_mode`, backend (`face_mediapipe|dino_sam|gemini_grabcut|…`),
  `fallback_reason`, feather thực, expand, `mask_hash`, diện tích; lưu vào `HistoryItem.executed` và export.
- Capabilities (`subject_capabilities()`, interpreter `sys.executable`, có `mobile_sam` hay không) ghi lúc startup,
  trả trong `/system` và trong export của phiên.
- Việc tay: xác định interpreter mà server thật đang chạy; cài `mobile_sam` vào `.venv` và chạy thử một lần inference.
  Ghi kết quả vào `docs/` (session A dùng `dino_sam`, session B rơi về GrabCut trên cùng ảnh).
- **Nghiệm thu:** export của một phiên đủ để biết mask nào, backend nào, feather bao nhiêu đã được dùng cho từng action.

### R-03 — Feather khai báo = feather thực thi `agent` `region` · ngày 1 · ~2h · F9
- `_resolve_subject_region` (`executor.py:239`) truyền `feather_radius` của action vào `resolve_subject_mask`;
  feather nằm trong cache key của subject mask.
- **Nghiệm thu:** test: action feather 30 → mask thực thi dùng 30 (khác 15); hai feather khác nhau không dùng chung cache.

### R-04 — Tách vùng dò tìm khỏi mask chỉnh, mask mặt bám đường viền `region` `agent` · ngày 1–2 · ~6h · F12, F13
- Phân biệt `detection_box` (bbox MediaPipe/Gemini, chỉ để tìm) với `edit_mask` (mask thực sự nhận xử lý).
- Thao tác exposure (`gamma_correct`, `clahe`) trên `face` dùng `face_mode=oval` (Face Landmarker, đã có) làm mặc định.
  Có MobileSAM thì thử SAM trong ROI mặt trước.
- QA mask tối thiểu: không rỗng, phần lớn nằm trong `detection_box`, không tràn ra ngoài quá mức, số thành phần hợp lý.
  QA trượt được tính là **fail**, giống exception.
- Fallback sang bbox: **không giữ cường độ**. Hoặc giảm cường độ về phía trung tính, hoặc bỏ action. Hệ số giảm là
  hằng số tạm thời có tên rõ, ghi `fallback_reason`. Sửa nhánh fallback hiện tại ở `executor.py:166-176`.
- Overlay (`region_overlay.py`) vẽ đúng mask đã thực thi (dùng `mask_hash`/mask đã lưu từ R-02), không dựng lại bằng đường khác.
- **Nghiệm thu:** test oval khả dụng → mask không phải hình chữ nhật; oval lỗi → bbox với cường độ giảm hoặc skip, có log;
  QA trượt → không áp nguyên cường độ; overlay khớp hash với mask đã thực thi.

### R-05 — Kế hoạch ngược sáng theo vùng con + liên kết action với defect `agent` · ngày 2 · ~6h · F1, F2, F10 (một phần)
- Thêm `RegionOperation.defect_ref: Optional[str]` (dạng `type@region`). `_targeted_defects` / `_needs_verification`
  (`graph.py:324-366`) dùng `defect_ref`, nhờ vậy action trên `face` vẫn theo dõi được lỗi `backlit_subject@person`.
- Card `dark-region` (`knowledge/cards/exposure.yaml:72`): khi defect ở chủ thể người và định vị được mặt (mask R-04
  đạt QA) thì recipe nhắm `face` làm chính. Việc có thêm một thao tác nhẹ trên `person` hay không được quyết định
  bằng so sánh ở R-09, không mặc định. Không định vị được mặt thì giữ recipe cũ, nhưng giảm một bậc và ghi lý do.
- Sửa rule 7 của `SYSTEM_PROMPT` (`vlm_diagnostician.py`): cho phép chỉnh vùng con của vùng defect, bắt buộc điền `defect_ref`.
- `apply_preserve_guard` (`planner.py:274-350`): coi `face` là con của `person` khi xét phạm vi. Bảo vệ `hair` đầy đủ để ở BL-02.
- Đồng bộ: schema plan, executor, prompt, card, `docs/interfaces.md`, `test_agent_knowledge.py`.
- **Nghiệm thu:** test ca backlit có mặt → action nhắm face với `defect_ref=backlit_subject@person` và tracking vẫn ra trạng thái;
  không có mặt → recipe cũ đã giảm cường độ; rule-based offline cũng đi đúng nhánh.

### R-06 — Đo tiến triển trên đúng vùng, chặn "resolved" mâu thuẫn `agent` · ngày 2 · ~3h · F3, F7
- `_track_defects` (`graph.py:369`): với `backlit_subject@person` khi mask mặt đạt QA, đo thêm vùng mặt.
  Nếu `person` ra `resolved` nhưng mặt không cải thiện tương ứng → trạng thái `unverified`, không SHIP theo số đo.
- Ghi nguyên nhân hết lỗi: do độ tối giảm hay do gap < `BACKLIT_MIN_GAP_L` (`fusion.py:462`).
- `FACE_MID_L` chỉ dùng cho **hướng thay đổi và sự mâu thuẫn**, không làm đích tuyệt đối.
- **Nghiệm thu:** replay session B → không còn `resolved`; test dựa vào sự mâu thuẫn giữa hai số đo, không assert "mặt ≥ 60".

### R-07 — Đo black-lift / đổi màu vật thể (ghi nhận, chưa chặn) `eval` `agent` · ngày 2 · ~3h · F8
- Kiểu side effect mới `black_lift` trong `measurement/diff.py`: xét pixel tối nhất của ảnh gốc **nằm trong mask đã chỉnh
  nhưng ngoài mask mặt**. Đo dịch phân vị L\* và thay đổi chroma, so **original→candidate** và previous→candidate.
- Không áp lên mọi pixel tối, để không chặn nhầm việc cứu sáng da tối. Severity là tạm thời: **không rollback**,
  chỉ (a) lưu vào history/export, (b) chọn crop nghi vấn cho reviewer R-08, (c) hiển thị cảnh báo.
- **Nghiệm thu:** replay session B phát hiện được black-lift ở vùng tóc; ảnh không có vùng tối bị chỉnh không báo; không gây rollback.

### R-08 — Cổng review tối giản trên ảnh sẽ trả `agent` · ngày 2–3 · ~6h · F4, F5, F7
- `review_candidate(original, previous, candidate, context) -> ReviewResult` qua `call_gemini_json`;
  cấu hình `REVIEW_ENABLED`, `REVIEW_MODEL`, `REVIEW_TIMEOUT_MS`, `REVIEW_MAX_CALLS`; `tests/conftest.py` mặc định tắt.
- **Input:** ảnh gốc, ảnh trước vòng cuối, **ảnh sẽ trả** (sau rollback nếu có); crop 100% ghép cặp vùng đã chỉnh
  và crop nghi vấn từ R-07; intent; defect ban đầu; thao tác đã thực thi lấy từ R-02 (không lấy từ plan).
- **Output:** `verdict` ACCEPT|REVISE|REJECT|UNCERTAIN; `goal[]`; `preserve[]`; `findings[]` (kind lấy từ enum,
  region, `crop_ids`, observation); confidence tự khai (không dùng làm ngưỡng); `reason` ngắn; `candidate_sha256`;
  model, `prompt_version`, latency, error.
- **Rẽ nhánh:** ACCEPT → `ACCEPTED`; REVISE → `NOT_MET` kèm findings (chưa có vòng REVISE tự động);
  REJECT → trả `previous_image`, ghi `UNVERIFIED`, hoặc `KEPT_ORIGINAL` nếu đó là ảnh gốc;
  UNCERTAIN/lỗi/timeout/không có key → `UNVERIFIED`. Chỉ thử lại một lần với lỗi tạm thời.
- Graph: node `review` chạy khi `decide` cho kết quả dừng, trước END. `decide_node` vẫn là bộ điều phối tất định.
  (Multi-variants đã bị gỡ khỏi `develop` ở `2d26c99`, nên không cần `reject_all` cho critic.)
- **Nghiệm thu:** test mock cho từng verdict; review nhận đúng hash ảnh trả (kể cả sau rollback); lỗi không bao giờ
  cho `ACCEPTED`; số lần gọi ≤ ngân sách; review có trong export.

### R-09 — Regression và so sánh cấu hình `test` · ngày 3 · ~4h
- Replay offline session A và B (ảnh đọc từ ngoài repo, skip nếu thiếu; chưa đưa ảnh cá nhân vào repo).
- So sánh 2–3 cấu hình recipe của R-05 trên ca này **và** các ảnh ngược sáng/chân dung trong `data/real/`. Báo cáo L\*
  vùng mặt, black-lift tóc/áo, ảnh ghép so sánh. Không chọn tham số chỉ để một ảnh pass; ghi rõ là lựa chọn tạm thời.
- Dùng lại script tái lập số đo (ROI, công thức) và đưa vào `docs/reports/`.
- CI: `ruff check src/ tests/`, `pytest tests/unit/`, Vitest, build web.
- Demo với Gemini thật (cần chủ dự án đồng ý): ghi lại outcome, review, latency, kể cả các ca `UNVERIFIED`/`NOT_MET`.
- **Nghiệm thu:** CI xanh; có bảng trước/sau cho ca chân dung; replay B không còn ra `SHIP`.

### R-10 — Tài liệu `docs` · ngày 3 · ~1h
- `docs/interfaces.md` (`defect_ref`, `executed`, `outcome`, `review`, `black_lift`); mục Pipeline trong `CLAUDE.md`; release note
  (những gì đã sửa, tham số nào là tạm thời, những gì còn ở backlog).

## 3. Chat Studio (C) — chỉ làm khi còn thời gian hoặc sau deadline

Luồng: ảnh + mô tả → AI phân tích có context (xác nhận vấn đề người dùng nêu, gợi ý thêm dưới dạng chip,
chỉ áp khi người dùng chọn) → chat/chip → nút **Xử lý** → mỗi vòng là một bước có preview và nút **Chi tiết**
mở trên pane chính → kết quả theo `outcome` → góp ý thì tinh chỉnh. Context của các node lấy từ lịch sử chat;
mỗi tin nhắn chỉ được parse một lần thành dữ liệu có cấu trúc, có ghi tin nhắn nguồn.

| ID | Việc | Ước lượng | Phụ thuộc |
|---|---|---|---|
| C-01 | `POST /sessions` nhận `prompt`, lưu message `user_prompt`, parse một lần thành `draft_edits`, đưa vào state | ~3h | — |
| C-02 | Perceive nhận prompt; schema thêm `user_concerns`, `suggestions`; fusion không đổi | ~5h | C-01 |
| C-03 | Intent từ chat/chip thay bảng câu hỏi; `accepted_suggestions`; tin "Mình sẽ: …" dựng tất định | ~4h | C-02 |
| C-04 | API `timeline` các bước (messages + runs, preview từng vòng, outcome/review) | ~3h | R-01 |
| C-05 | UI Chat Studio: pane chính + khung chat, renderer theo loại bước, Chi tiết | ~8h | C-03, C-04 |
| C-06 | Góp ý trong chat → refine + review; chip không hứa thao tác mà executor không làm được | ~4h | R-08, C-05 |

Nếu ngày 3 còn thời gian, ưu tiên C-01 và C-02: chỉ là backend và trực tiếp giúp phân tích có thêm context.

## 4. Backlog (sau deadline)

| ID | Việc | Finding |
|---|---|---|
| BL-02 | `edit_regions / protected_regions` đầy đủ, mask hair, phép trừ mask, guard theo quan hệ chứa nhau | F1, F10 |
| BL-03 | Vòng REVISE tự động: finding → ràng buộc, render lại từ ảnh gốc, kiểm tra khả năng sửa, chống lặp | F4 |
| BL-04 | Lưu mọi candidate + chọn candidate tốt nhất theo lịch sử; review lại candidate thay thế | F11 |
| BL-05 | Dùng black-lift/màu/texture để chặn sau khi đã hiệu chỉnh; ngân sách tác động cộng dồn | F8 |
| BL-06 | Benchmark exposure linear so với gamma; mode opt-in, đồng bộ 5 chỗ | F2 |
| BL-09 | Review cho refine | F11 |
| BL-10 | Hold-out tách calibration, người chấm cặp mù, bỏ lọt/chặn nhầm của reviewer, latency/cost; hiệu chỉnh `FACE_MID_L`, quy tắc gap, các hằng số tạm thời của R-04/R-07 | — |
| BL-11 | Phân tích cảnh có cấu trúc trong Plan (`expected_effect`, `failure_signs`) nối với reviewer | P1 của report |

## 5. Rủi ro

- Face oval và Face Landmarker có thể không khả dụng trên máy demo. R-02 phải cho thấy điều đó, R-04 phải xử lý trung thực.
- Recipe nhắm mặt có thể làm mặt sáng lệch với cổ/thân. R-09 phải xem bằng ảnh, không chỉ bằng số.
- Ảnh thật chạy offline không bao giờ ra `ACCEPTED`; test và nhãn UI phải đổi theo.
- Review thêm ít nhất một lời gọi Gemini mỗi lần trả ảnh; tỉ lệ bỏ lọt/chặn nhầm chưa đo được trước BL-10.
- Các hằng số mới (hệ số giảm khi fallback, severity black-lift) là tạm thời; không được trình bày như đã hiệu chỉnh.
