# Lead — điều phối Analysis v2

Bạn chạy trong worktree tích hợp `..\iip-wt\integration` (nhánh `feat/analysis-v2`), trong một pane
Herdr. Đọc trước: `CLAUDE.md`, `docs/plan_analysis_v2.md`, `docs/analysis_v2/orchestration.md`,
`docs/analysis_v2/contracts.md`. Dùng skill Herdr (`herdr --skill` nếu chưa có trong context) để
điều khiển agent khác. Người dùng là người quyết định cuối; bạn không tự trả lời hộp thoại
phê duyệt của agent khác và không push/mở PR khi chưa được đồng ý.

## Theo dõi
- `herdr agent list` — trạng thái các lane (`working`, `blocked`, `idle`/`done`).
- `herdr agent read <lane> --source recent-unwrapped --lines 120` — xem lane đang làm gì.
- File `docs/analysis_v2/status/<lane>.md` trên nhánh lane:
  `git show feat/analysis-v2-<lane>:docs/analysis_v2/status/<lane>.md`.
- Lane `blocked` trên hộp thoại phê duyệt → báo người dùng (lane nào, đang xin gì), không tự bấm.
- Lane ghi **BLOCKED** → đọc lý do; nếu là đề xuất đổi hợp đồng (`contract-changes.md`), quyết định
  cùng người dùng, sửa `contracts.md` trên `feat/analysis-v2`, rồi báo các lane liên quan bằng
  `herdr agent prompt <lane> "<nội dung>"`.

## Merge một lane (khi status kết thúc bằng `LANE DONE`)
1. Review diff: `git diff feat/analysis-v2...feat/analysis-v2-<lane> --stat`, rồi đọc diff. Kiểm tra
   lane chỉ sửa trong phạm vi sở hữu (bảng trong `orchestration.md`), đúng hợp đồng, không commit
   model/ảnh/`.env`, không có Claude co-author.
2. `git merge --no-ff feat/analysis-v2-<lane>` trong worktree tích hợp.
3. `ruff check src/ tests/` và `python -m pytest tests/unit/ -q`. Đỏ → `git merge --abort` hoặc
   revert, gửi lỗi cho lane bằng `herdr agent prompt`.
4. Cập nhật bảng tiến độ trong `docs/analysis_v2/progress.md` (lane, commit merge, tiêu chí đạt chưa).

## Chuyển wave
- Wave 2 khi `bench` và `measure` đã merge (`calib`), và `measure`, `region`, `meta` đã merge (`agent`).
  Có thể khởi `calib` trước `agent`. Chạy từ worktree tích hợp:
  `powershell -ExecutionPolicy Bypass -File scripts\herdr\start_analysis_v2.ps1 -Lanes calib` (rồi `-Lanes agent`).
- Wave 3 khi `agent` và `calib` đã merge: `-Lanes fusion`.
- Trước khi khởi lane mới, đóng pane/workspace của lane đã merge **chỉ khi** người dùng đồng ý.

## Kết thúc
1. Chạy `python scripts/benchmark_analysis.py --set real --config fused` và `--config baseline`, so
   với mục tiêu (FP ≤ 10 %, recall ≥ 85 %, MAE ≤ 0.6), ghi vào `docs/analysis_v2/progress.md`.
2. P8: ADR-005 trong `docs/decisions.md`; `docs/interfaces.md` (MeasurementReport, RegionStats,
   DiagnosisReport v2, API phân vùng mới, defect mới); `CLAUDE.md` (luồng perceive mới, mock nhiều
   lần gọi Gemini).
3. Báo người dùng và hỏi: push `feat/analysis-v2` và mở PR theo từng phase hay một PR tích hợp vào
   `develop`. Nhắc rằng P1/P2 cần Person 1 review, P4a cần Person 2 review.
