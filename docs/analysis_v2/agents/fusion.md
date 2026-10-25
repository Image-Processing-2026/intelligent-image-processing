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

## Thêm từ checkpoint sau calib (người dùng duyệt, 2026-10-25)
Đọc `docs/analysis_v2/checkpoint_calib.md` trước. Checkpoint không đạt (FP ảnh sạch 0.54 → 0.39);
fusion là nơi phải kéo FP về ≤ 10 %.

9. **Preserve là việc chính.** `measure_only` báo FP trên **10/10** ảnh `intentional_style`
   (low-key → underexposed, hoàng hôn → warm, high-key → overexposed). Đo không biết chủ ý: bảng
   P6 (đo ≥ 1 + VLM preserve → preserved) phải chặn các ca này.
10. **Ám màu bị bỏ sót 4/4:** VLM không bao giờ báo `color_cast_*` dù số đo có góc lệch nguồn sáng.
    Fusion phải dùng số đo (`severity_from_measurement("color_cast_*")`, sau calib đo tốt hơn:
    vector nguồn sáng MAE 2.0°) làm giả thuyết, rồi verify/preserve (warm_tone, cool_tone).
11. **`backlit_subject` đo được** (11/17 FP còn lại): phạm vi mở rộng sang
    `perception._measured_defects` và `_brightness_level`. Bỏ ngưỡng tay cũ (`_BRIGHTNESS_*`,
    `_*_CLIP_*` import từ `analyzer.py`, `BACKLIT_GAP`, `BACKLIT_MAX_MEAN`); dùng
    `severity_from_measurement` trên `RegionStats` (truyền `long_side`, C4). Lỗi đo được **không
    được phát ra một mình**: phải qua bảng fusion (VLM và đo đồng ý, hoặc verify, hoặc preserve).
12. **D3:** `PERCEIVE_SAMPLES` mặc định đã là 1 (ghi đè bằng env `PERCEIVE_SAMPLES`); code gộp giữ
    nguyên, config benchmark `vlm_only_n3` để so. Đánh giá xem giả thuyết `rejected` có giúp khi làm
    bằng chứng yếu trong fusion không — **chỉ kết luận trên bộ hold-out**, không trên 30 ảnh đã chỉnh.
13. **Bộ hold-out** (`data/holdout_manifest.json`, lead chuẩn bị song song): chỉ dùng để chấm kết
    quả cuối của fusion; lead chạy `--set holdout` sau khi lane ghi `LANE DONE`. **Không** chỉnh prompt, ngưỡng hay luật trên nó; không mở ảnh/đáp án khi
    còn đang chỉnh. Nhãn đang `unreviewed` cho tới khi người dùng duyệt. Phát triển trên 30 ảnh
    `data/real/benchmark` + synthetic, ghi rõ số nào là trên tập đã chỉnh.
14. Người dùng duyệt các sửa ngoài phạm vi của lane `agent` (`contract-changes.md` mục 7).
    Executor bỏ qua `feather_radius` khi dùng resolver chung: giữ, hoặc sửa nếu cần.

## Test
Mỗi dòng bảng quyết định một test; VLM khai nhiễu trên ảnh sạch → rejected; hoàng hôn + preserve
warm_tone → preserved; contraindication bỏ sharpen; sharpen mạnh → phát hiện halo; gamma 2.0 trên
ảnh tối nhiễu → phát hiện nhiễu vùng tối tăng; xử lý đúng → resolved.

## Tiêu chí xong
`--config fused` trên ảnh thật đạt (hoặc ghi rõ khoảng cách tới) mục tiêu: FP rate ảnh tốt + chủ ý
≤ 10 %, recall ≥ 85 %, MAE severity ≤ 0.6. Toàn bộ test xanh. `docs/analysis_v2/status/fusion.md`
kết thúc bằng `LANE DONE`.
