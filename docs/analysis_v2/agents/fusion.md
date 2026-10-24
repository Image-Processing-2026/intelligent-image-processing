# Lane `fusion` — P6 + P7: kiểm chứng, hợp nhất, phân tích khác biệt (wave 3)

Nhánh `feat/analysis-v2-fusion`, tạo từ `feat/analysis-v2` **sau khi** `agent` và `calib` đã merge.
Đọc trước: `CLAUDE.md`, `docs/plan_analysis_v2.md` §Phase 6, §Phase 7, §7 (D2, D3);
`contracts.md` §C4 §C6; status của `agent` và `calib`.

## P6 — kiểm chứng chéo + hợp nhất
1. `src/agent/fusion.py`: `fuse(hypotheses, measurement, region_stats, priors, intent) -> DiagnosisReport`,
   hoàn toàn tất định, đúng bảng quyết định trong plan; severity từ `severity_from_measurement`.
2. Gọi Gemini `verify` với câu hỏi đóng (schema trong plan), chỉ khi bảng yêu cầu.
3. DiagnosisReport v2 (C6): `confidence`, `evidence_items`, `fixable`, `contraindications`,
   `rejected`, `ask_user`. `defects` chỉ chứa confirmed.
4. D2 — defect mới `motion_blur`, `jpeg_artifacts`, `haze`, `highlight_clipping` (`fixable=False`
   khi toolbox không sửa được): đồng bộ `DEFECT_TYPES`, schema perceive, knowledge cards,
   `SYSTEM_PROMPT`; `test_agent_knowledge.py` phải xanh.
5. Planner đọc `contraindications` (`no_sharpen` → bỏ sharpen); clarify sinh câu hỏi từ `ask_user`
   (tất định trước `interrupt()`).
6. Benchmark: điền config `fused`.

## P7 — vòng ≥ 2
7. `src/analyzer_evaluator/measurement/diff.py`: `compare(original, current, regions) -> DiffReport`
   — trạng thái từng defect (resolved/improved/unchanged/worse) và tác dụng phụ (halo, nhiễu vùng
   tối tăng, banding, cháy kênh tăng, bão hòa quá).
8. `perceive_node` vòng ≥ 2: đo lại + diff; chỉ gọi VLM cho defect ngữ nghĩa còn mở hoặc tác dụng
   phụ cần phân xử. `decide_node`: tác dụng phụ severity ≥ 2 → coi như suy thoái → rollback.

## Test
Mỗi dòng bảng quyết định một test; VLM khai nhiễu trên ảnh sạch → rejected; hoàng hôn + preserve
warm_tone → preserved; contraindication bỏ sharpen; sharpen mạnh → phát hiện halo; gamma 2.0 trên
ảnh tối nhiễu → phát hiện nhiễu vùng tối tăng; xử lý đúng → resolved.

## Tiêu chí xong
`--config fused` trên ảnh thật đạt (hoặc ghi rõ khoảng cách tới) mục tiêu: FP rate ảnh tốt + chủ ý
≤ 10 %, recall ≥ 85 %, MAE severity ≤ 0.6. Toàn bộ test xanh. `docs/analysis_v2/status/fusion.md`
kết thúc bằng `LANE DONE`.
