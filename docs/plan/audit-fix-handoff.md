# Handoff: chia R-01..R-10 cho các session song song

Tickets: [chat-studio-tickets.md](chat-studio-tickets.md) mục 2. Bằng chứng: [report](../reports/2026-10-27-backlit-portrait-pipeline-review.md),
[audit notes](../reports/2026-10-27-output-evaluation-audit-notes.md). Deadline ~30/10/2026.

## 1. Sơ đồ session

| Session | Tickets | Worktree / branch | Sở hữu (chỉ session này được sửa) |
|---|---|---|---|
| **S0 Integrator** | Contract (bước 0), R-09, R-10, merge | repo chính hoặc `../iip-wt/audit` · `fix/audit-backlit` | `src/agent/state.py` (phần contract), `docs/interfaces.md`, `CLAUDE.md`, `docs/reports/`, script replay |
| **S1 Mask & thực thi** | R-03 → R-02 → R-04 | `../iip-wt/audit-mask` · `fix/audit-mask` | `src/agent/executor.py`, `src/region_engine/*`, `src/agent/region_overlay.py`, endpoint `/system`, body của `process_node` |
| **S2 Kế hoạch & đo** | R-05 → R-06 → R-07 | `../iip-wt/audit-plan` · `fix/audit-plan` | `src/agent/planner.py`, `src/agent/knowledge/**`, `SYSTEM_PROMPT` trong `vlm_diagnostician.py`, `src/agent/fusion.py`, `src/analyzer_evaluator/measurement/diff.py`, các hàm `_needs_verification` / `_side_effects*` / `_targeted_defects` / `_track_defects` / `_decide_tracked` trong `graph.py` |
| **S3 Outcome & review** | R-01 → R-08 | `../iip-wt/audit-outcome` · `fix/audit-outcome` | `decide_node`, `evaluate_node`, `should_continue`, `build_doctor_graph` trong `graph.py`; `src/agent/review.py` (mới); `src/api/workspace_service.py`; hiển thị outcome ở web và Gradio; sửa các assert `"SHIP"` |

Ba lane S1/S2/S3 chạy song song **sau khi S0 xong bước 0**. File ngoài phần sở hữu: chỉ đọc. Cần sửa thì ghi vào báo cáo cho S0.

## 2. Lịch

| Thời điểm | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| Ngày 1 sáng | Bước 0 (~1–1.5h) | chờ / đọc code | chờ / đọc code | chờ / đọc code |
| Ngày 1 | Script replay baseline (A/B) | R-03, R-02 | R-05 | R-01 |
| Cuối ngày 1 | Merge S1 → S2 → S3 vào `fix/audit-backlit`, chạy CI | rebase | rebase | rebase |
| Ngày 2 | Replay sau merge | R-04 | R-06, R-07 | R-08 (dùng dữ liệu giả tới khi R-02/R-07 vào) |
| Cuối ngày 2 | Merge, CI | rebase | rebase | rebase |
| Ngày 3 | R-09 so sánh cấu hình, demo, R-10 | sửa lỗi từ R-09 | sửa lỗi từ R-09 | sửa lỗi, UI outcome |

Thứ tự merge mỗi lần: **S1 → S2 → S3** (S3 dùng dữ liệu của hai lane kia). Merge cục bộ bằng merge commit, không squash.
PR `fix/audit-backlit → develop` do chủ dự án tự mở và merge.

## 3. Bước 0: contract (S0 làm trước, một commit)

Tạo `fix/audit-backlit` từ `develop` (`3915240`; multi-variants đã bị gỡ). Commit `refactor(agent): add audit-fix contracts`
chỉ gồm khung và giá trị mặc định, hành vi không đổi, toàn bộ test vẫn pass:

1. `state.py`:
   - `RegionOperation.defect_ref: Optional[str] = None` (dạng `type@region`).
   - `ExecutedAction(BaseModel)`: `region_id, operation, parameters, requested_mode, actual_mode, backend, fallback_reason, feather_requested, feather_actual, strength_scale=1.0, mask_hash, mask_area, skipped: bool, skip_reason`.
   - `SideEffectCheck(BaseModel)`: `status: Literal["ok", "unavailable", "not_run"] = "not_run"`, `previous: List[Dict]`, `original: List[Dict]`, `error: Optional[str]`.
   - `ReviewFinding`, `ReviewResult` (`status: completed|unavailable|not_run = "not_run"`, `verdict: Optional[ACCEPT|REVISE|REJECT|UNCERTAIN]`, `goal`, `preserve`, `findings`, `confidence`, `reason`, `candidate_sha256`, `model`, `prompt_version`, `latency_ms`, `error`).
   - `Outcome(BaseModel)`: `status: ACCEPTED|UNVERIFIED|NOT_MET|KEPT_ORIGINAL|NO_CHANGE`, `accepted_by: Optional[review|reference]`, `candidate_sha256`, `reason`.
   - `HistoryItem`: thêm `executed: List[ExecutedAction] = []`, `side_effect_check: SideEffectCheck`, `review: Optional[ReviewResult] = None`.
   - `DoctorState` và `initial_doctor_state`: `executed_actions: List[ExecutedAction]`, `outcome: Optional[Outcome]`, `review: Optional[ReviewResult]`.
2. Plumbing tối thiểu trong `graph.py`:
   - `execute_plan(..., report: Optional[List[ExecutedAction]] = None)`: khung rỗng, S1 điền.
   - `process_node` truyền list và trả `executed_actions`.
   - `decide_node` chép vào `HistoryItem.executed`.
   - Đổi `_side_effects(previous, current)` thành `_side_effects_check(state) -> SideEffectCheck`; tạm thời bọc logic cũ (previous→current, status `ok`/`unavailable`). S2 sở hữu phần ruột, S3 chỉ gọi.
3. `docs/interfaces.md`: mục "Audit-fix contracts" mô tả các kiểu trên.
4. Tạo 3 worktree từ commit này:
   `git worktree add ../iip-wt/audit-mask -b fix/audit-mask`, tương tự cho `audit-plan` và `audit-outcome`.

## 4. Quy tắc chung (dán kèm mọi prompt)

- Đọc `CLAUDE.md` và mục ticket của mình trước. Làm việc trên code **hiện tại** của branch; số dòng trong report có thể lệch.
- Chỉ sửa file thuộc phần sở hữu. Cần đổi contract trong `state.py` thì ghi lại để báo S0, không tự sửa.
- Docstring và comment bằng tiếng Việt có dấu; log, exception, identifier và commit message bằng tiếng Anh. Conventional Commits
  với scope `agent|engine|region|eval|api|ui`. **Không** thêm Claude làm author/co-author.
- Mỗi ticket: test offline (không key, không mạng; Gemini được mock như `tests/unit/test_vlm_mock.py`), sau đó chạy
  `ruff check src/ tests/ --fix && ruff format src/ tests/ && ruff check src/ tests/` và `pytest tests/unit/ -q`.
- Hằng số mới phải có tên rõ, comment "tạm thời, chưa hiệu chỉnh (BL-10)", và không được chỉnh chỉ để một ảnh pass.
- Không dùng AI sinh ảnh (ADR-001). Không import `src/extensions/`.
- Không push và không mở PR. Commit trong worktree của mình; mỗi ticket ít nhất một commit.
- Kết thúc mỗi ticket, báo cáo theo mẫu ở mục 6.

## 5. Prompt cho từng session

### S1 — Mask & thực thi

```text
Bạn là session S1 (Mask & thực thi) của đợt sửa audit. Worktree ../iip-wt/audit-mask, branch fix/audit-mask.
Đọc: CLAUDE.md; docs/plan/audit-fix-handoff.md (mục 3, 4); docs/plan/chat-studio-tickets.md (R-02, R-03, R-04);
report docs/reports/2026-10-27-backlit-portrait-pipeline-review.md (F9, F12, F13).
Sở hữu: src/agent/executor.py, src/region_engine/*, src/agent/region_overlay.py, endpoint /system, body của process_node.
Không sửa: decide_node, planner, cards, fusion, diff.py, state.py (contract do S0 giữ).

Làm theo thứ tự:
1. R-03: _resolve_subject_region truyền feather_radius của action vào resolve_subject_mask; feather nằm trong cache key.
2. R-02: execute_plan điền report (ExecutedAction) cho mọi action, kể cả action bị skip: requested/actual face_mode,
   backend, fallback_reason, feather requested/actual, strength_scale, mask_hash (sha256 của mask float32 làm tròn), mask_area.
   /system và export trả capabilities (subject_capabilities(), sys.executable, có mobile_sam hay không).
   Việc tay: cài mobile_sam vào .venv (theo scripts/fetch_segmentation_models.py, README), chạy một inference thật
   và ghi kết quả vào báo cáo. Không commit trọng số model.
3. R-04: phân biệt detection_box và edit_mask. Thao tác exposure (gamma_correct, clahe) trên face mặc định dùng
   face_mode=oval; có MobileSAM thì thử SAM trong ROI mặt trước. QA mask tối thiểu (không rỗng, nằm chủ yếu trong
   detection_box, không tràn quá mức, số thành phần hợp lý); QA trượt tính là fail. Fallback sang bbox:
   giảm cường độ về phía trung tính theo hằng số FALLBACK_STRENGTH_SCALE (tạm thời) hoặc skip; ghi strength_scale
   và fallback_reason. Sửa nhánh fallback oval→bbox giữ nguyên cường độ trong executor.
   region_overlay vẽ đúng mask đã thực thi (đối chiếu mask_hash).
Nghiệm thu: theo từng ticket trong chat-studio-tickets.md. Test bắt buộc: feather 30 ≠ 15; hai feather không dùng chung
cache; oval → mask không phải hình chữ nhật; oval lỗi → giảm cường độ hoặc skip, có log; QA trượt → không áp nguyên
cường độ; report đủ trường cho action thành công, action fallback và action bị skip.
```

### S2 — Kế hoạch & đo

```text
Bạn là session S2 (Kế hoạch & đo) của đợt sửa audit. Worktree ../iip-wt/audit-plan, branch fix/audit-plan.
Đọc: CLAUDE.md; docs/plan/audit-fix-handoff.md (mục 3, 4); docs/plan/chat-studio-tickets.md (R-05, R-06, R-07);
report (F1, F2, F3, F7, F8, F10) và docs/reports/reproduce_backlit_hair_case.py.
Sở hữu: planner.py, knowledge/** (cards, rule engine), SYSTEM_PROMPT trong vlm_diagnostician.py, fusion.py,
measurement/diff.py, và trong graph.py chỉ các hàm _needs_verification, _side_effects_check, _targeted_defects,
_track_defects, _decide_tracked.
Không sửa: executor, region_engine, decide_node, evaluate_node, workspace_service, state.py.

Làm theo thứ tự:
1. R-05: _targeted_defects và _needs_verification dùng RegionOperation.defect_ref (fallback về so tên vùng như cũ).
   Card dark-region: defect ở chủ thể người và Perceive có mặt (face mask hoặc face_stats) → recipe nhắm face
   (face_mode=oval, defect_ref=backlit_subject@<region>). Việc có thêm một thao tác nhẹ trên person là cấu hình bật/tắt,
   mặc định tắt; S0 sẽ quyết định ở R-09. Không có mặt → recipe cũ, giảm một bậc, ghi lý do.
   Rule 7 của SYSTEM_PROMPT: cho phép chỉnh vùng con của vùng defect, bắt buộc điền defect_ref; giữ đồng bộ PARAMETER_BOUNDS.
   apply_preserve_guard: face là con của person khi xét phạm vi. Rule-based offline phải đi đúng nhánh.
   Card được validate khi load: test_agent_knowledge.py phải pass.
2. R-06: với backlit_subject@person khi có mask mặt, _track_defects đo thêm vùng mặt. Person resolved nhưng mặt
   không cải thiện tương ứng → status "unverified" (không SHIP vì số đo). Ghi nguyên nhân hết lỗi: độ tối giảm hay
   gap < BACKLIT_MIN_GAP_L. FACE_MID_L chỉ dùng cho hướng thay đổi và sự mâu thuẫn, không phải đích tuyệt đối.
   Nếu cần thêm giá trị DefectChange.status mới, báo S0 (contract).
3. R-07: side effect "black_lift" trong diff.py: lấy pixel tối nhất của ảnh gốc, chỉ trong vùng đã bị thay đổi
   (|ΔL| > 0) và ngoài mask mặt; đo dịch phân vị L* và Δchroma, cả original→candidate và previous→candidate.
   _side_effects_check trả SideEffectCheck đủ hai chiều; exception → status "unavailable", không được trả list rỗng.
   black_lift không gây rollback (dưới SIDE_EFFECT_ROLLBACK_SEVERITY hoặc loại trừ tường minh).
Nghiệm thu: replay session B (reproduce_backlit_hair_case.py; ảnh ngoài repo, skip nếu thiếu) cho thấy black_lift
ở vùng tóc và tracking không ra resolved; test tổng hợp: action face có defect_ref vẫn được track;
ảnh không có vùng tối bị chỉnh thì không báo black_lift; không assert "mặt ≥ 60".
```

### S3 — Outcome & review

```text
Bạn là session S3 (Outcome & review) của đợt sửa audit. Worktree ../iip-wt/audit-outcome, branch fix/audit-outcome.
Đọc: CLAUDE.md; docs/plan/audit-fix-handoff.md (mục 3, 4); docs/plan/chat-studio-tickets.md (R-01, R-08);
report (F4, F5, F6, F7, F11) và docs/reports/2026-10-27-output-evaluation-audit-notes.md.
Sở hữu: decide_node, evaluate_node, should_continue, build_doctor_graph trong graph.py; src/agent/review.py (mới);
src/api/workspace_service.py (record_process, export); hiển thị outcome ở web (web/src/api/types.ts, StatusBadge,
labels.ts, ResultCard) và frontend/app.py; các assert "SHIP" trong tests.
Không sửa: executor, region_engine, planner, cards, fusion, diff.py, _track_defects/_targeted_defects, state.py.

Làm theo thứ tự:
1. R-01: tính Outcome ở cuối graph và xuất qua record_process (run.result.outcome). Chỉ ACCEPTED ↔ final_decision SHIP;
   synthetic đạt reference → ACCEPTED với accepted_by=reference. KEPT_ORIGINAL xác định bằng so pixel với ảnh gốc
   khi intent có fix. Sau rollback, tính lại evaluation_result cho ảnh được trả. Lưu HistoryItem.side_effect_check
   lấy từ _side_effects_check(state) (S2 sở hữu phần ruột). Agent/API ngừng đọc visual_plausibility_passed/vlm_feedback;
   không đổi EvaluationResult. UI hiển thị nhãn: Đã duyệt / Chưa xác minh / Chưa đạt / Giữ ảnh gốc / Không cần chỉnh,
   có cả chữ và biểu tượng. Rà các test assert "SHIP": ảnh thật không có review → UNVERIFIED.
2. R-08: review_candidate(original, previous, candidate, context) -> ReviewResult qua call_gemini_json;
   REVIEW_ENABLED/REVIEW_MODEL/REVIEW_TIMEOUT_MS/REVIEW_MAX_CALLS; tests/conftest.py mặc định tắt (giống VERIFY_ENABLED).
   Input: ảnh gốc, ảnh trước vòng cuối, ẢNH SẼ TRẢ (sau rollback nếu có), crop 100% ghép cặp của vùng đã chỉnh
   (dùng HistoryItem.executed khi có) và crop nghi vấn (side effect black_lift khi có; thiếu thì bỏ qua), intent,
   defect ban đầu, thao tác đã thực thi. Output đúng ReviewResult; prompt yêu cầu mỗi kết luận trỏ tới crop_id,
   cho phép UNCERTAIN, không yêu cầu suy luận dài.
   Rẽ nhánh: node review chạy khi decide cho kết quả dừng, trước END. ACCEPT → ACCEPTED; REVISE → NOT_MET kèm findings;
   REJECT → trả previous_image (UNVERIFIED, hoặc KEPT_ORIGINAL nếu đó là ảnh gốc); UNCERTAIN/lỗi/timeout/không key →
   UNVERIFIED. Chỉ thử lại một lần với lỗi tạm thời. decide_node vẫn là bộ điều phối tất định.
Nghiệm thu: test mock cho từng verdict; review nhận đúng sha256 của ảnh được trả (kể cả sau rollback); lỗi không bao giờ
ra ACCEPTED; số lời gọi ≤ ngân sách; hết vòng không suy ra ACCEPTED; replay session A → KEPT_ORIGINAL; review có trong
export; Vitest và build web pass.
```

### S0 — Integrator (sau bước 0)

```text
Bạn là S0 (Integrator). Sau bước 0 (handoff mục 3):
- Ngày 1: dựng script replay baseline cho session A và B (dựa trên docs/reports/reproduce_backlit_hair_case.py và
  script đo vùng con trong scratchpad của phiên audit; đưa vào docs/reports/ với ROI, công thức, tham số).
  Đọc ảnh từ data/app/users/*/sessions/<id>/ (ngoài git), skip nếu thiếu.
- Cuối ngày 1 và 2: merge S1 → S2 → S3 vào fix/audit-backlit (merge commit), giải quyết xung đột, chạy ruff,
  pytest tests/unit/, Vitest, build web; gửi lại cho các lane danh sách lỗi và yêu cầu đổi contract.
- Ngày 3 (R-09): so 2–3 cấu hình recipe của R-05 trên ca chân dung và các ảnh ngược sáng/chân dung trong data/real/.
  Bảng gồm L* mặt, black_lift tóc/áo, outcome, ảnh ghép. Chọn mặc định tạm thời và ghi rõ là chưa hiệu chỉnh.
  Demo với Gemini thật chỉ khi chủ dự án đồng ý.
- R-10: docs/interfaces.md, mục Pipeline trong CLAUDE.md, release note (đã sửa gì, tham số nào tạm thời, backlog).
Không push; báo chủ dự án khi fix/audit-backlit sẵn sàng mở PR.
```

## 6. Mẫu báo cáo của mỗi session

```text
Ticket: R-xx — done | partial | blocked
Commits: <hash> <message>
Đã làm: …
Test: lệnh đã chạy + kết quả (số pass/fail)
Bằng chứng: replay/ảnh/log (nếu có)
Cần S0 xử lý: đổi contract, file ngoài phần sở hữu, xung đột dự kiến
Chưa làm / rủi ro: …
```

## 7. Điểm dễ xung đột

- `graph.py`: chia theo hàm như bảng mục 1. Không di chuyển, đổi tên hay sắp xếp lại các hàm của lane khác.
- `state.py`: chỉ S0 sửa. Lane cần trường mới thì báo S0.
- `tests/conftest.py`: S3 thêm cờ `REVIEW_ENABLED`; S1/S2 cần cờ khác thì báo S0.
- Web: repo chính đang có thay đổi frontend chưa commit trên `feat/api-analysis-chat`. Phần UI của S3 phải chờ
  chủ dự án commit hoặc gác lại phần đó, nếu không sẽ xung đột.
