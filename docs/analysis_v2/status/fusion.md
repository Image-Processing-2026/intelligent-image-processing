# Lane `fusion` — trạng thái (P6 + P7, wave 3)

Nhánh `feat/analysis-v2-fusion` (worktree `iip-wt/fusion`), tạo từ `feat/analysis-v2` @ `dec2d3d`.
Brief: `docs/analysis_v2/agents/fusion.md` (mục 1–14). Hợp đồng: `contracts.md` §C4 §C6.
Sửa ngoài phạm vi và thay đổi hành vi: `contract-changes.md` mục 8 (chờ lead duyệt).

## Đã làm (commit `16bc86b`, `439b7ed`, `2e2395f`)

### P6 — kiểm chứng chéo + hợp nhất
- `src/agent/fusion.py`: `fuse(hypotheses, measurement, region_stats, priors, intent, *, verifier,
  long_side, crops_for, use_weak)`, tất định với cùng câu trả lời verify. Bảng quyết định:
  đo ≥ 1 + VLM báo → confirmed (severity = số đo, VLM kéo ±1); đo ≥ 1 + preserve → preserved
  (mức 3 trên chủ thể → `ask_user`); đo ≥ 1, VLM không nhắc → verify (không verify được → bỏ:
  lỗi đo được không phát ra một mình, brief mục 11); đo 0 + VLM báo → verify (not_present/texture
  → rejected; verify lỗi → giữ lời VLM, confidence × 0.5); không đo được → confirmed theo đồng
  thuận; intent áp sau cùng. Không có VLM (luật dự phòng): giữ lỗi toàn ảnh của luật, thêm giả
  thuyết vùng có confidence ≥ 0.5.
- Verify: một lời gọi Gemini/ảnh, schema câu hỏi đóng (`defect|intentional|texture|not_present`),
  ảnh toàn khung + crop 100 % của Perceive. Verify trả `intentional` → thêm preserve tương ứng.
- `backlit_subject` đo được: chủ thể (ưu tiên mặt) so với `background`, nền sáng hơn ≥ 20 L*,
  severity theo độ tối so với mốc (mặt 60 L*, chủ thể khác 50 L*). Ngưỡng prior. Bỏ ngưỡng tay cũ
  (`_BRIGHTNESS_*`, `_*_CLIP_*`, `BACKLIT_GAP`, `BACKLIT_MAX_MEAN`, `_measured_defects`);
  `RegionMetrics.brightness_level` suy từ calibration.
- D2: `DEFECT_TYPES` thêm `motion_blur`, `jpeg_artifacts`, `haze`, `highlight_clipping`
  (`fixable=False`, chỉ báo khi mức ≥ 2). Contraindication `no_sharpen` (jpeg ≥ 1, motion_blur ≥ 2,
  prior EXIF no_sharpen), `no_brighten` (cháy sáng ≥ 2); planner `apply_contraindications`.
  `actionable_defects` bỏ lỗi không sửa được. Card tư vấn `knowledge/cards/limits.yaml`;
  `SYSTEM_PROMPT` và `PERCEIVE_PROMPT` đã đồng bộ; `test_agent_knowledge.py` xanh.
- Clarify sinh câu hỏi tất định từ `ask_user` (ưu tiên trước các câu mơ hồ khác); trả lời "sửa"
  khôi phục đúng lỗi/vùng đã chẩn đoán.
- Benchmark: `fused`, `fused_noverify`, `fused_n3`, `fused_n3_weak`, `--set holdout`,
  `--vlm-cache`.

### P7 — vòng ≥ 2
- `measurement/diff.py`: `compare(original, current, regions, *, defects, original_measurement,
  current_measurement, assess) -> DiffReport` — trạng thái lỗi (resolved/improved/unchanged/worse)
  và tác dụng phụ halo, dark_noise, banding, channel_clipping, oversaturation. Ngưỡng prior, đo
  trên 5 ảnh thật với thao tác toolbox thật (identity/khử nhiễu/gamma ≤ 2 sạch; sharpen 2.0,
  CLAHE 4, CLAHE 2 trên ảnh tối nhiễu → mức ≥ 2).
- `perceive` vòng ≥ 2: đo lại + diff (cache kết quả đo ảnh gốc); chỉ gọi VLM khi còn lỗi ngữ
  nghĩa mở hoặc tác dụng phụ cần phân xử. `decide_node`: tác dụng phụ ≥ 2 → suy thoái → rollback.

### Test
`test_agent_fusion.py` (32: mỗi dòng bảng một test, nhiễu ảo trên ảnh sạch → rejected, hoàng hôn +
warm_tone → preserved, contraindication bỏ sharpen, verify mock qua perceive),
`test_measurement_diff.py` (16: sharpen mạnh → halo, gamma 2.0 ảnh tối nhiễu → dark_noise, sửa
đúng → resolved…), `test_agent_followup.py` (5). `pytest tests/unit/`: **1425 passed**; ruff sạch.

## Số đo (trên 30 ảnh thật đã dùng để chỉnh — lạc quan)

Lần chạy `fused` thứ nhất (`fusion-r1-fused`, trước khi thêm sàn mức 2 cho lỗi không sửa được),
55 lời gọi Gemini:

| | checkpoint calib baseline | measure_only | **fused r1** |
|---|---|---|---|
| FP ảnh sạch (n = 13) | 5 (0.39) | 10 (0.77) | **2 (0.15)** |
| Recall (15) | 5 (0.33) | 5 (0.33) | 7 (0.47) |
| MAE severity | 0.80 | 0.80 | 1.00 |
| Preserve recall | 0.71 | 0 | 0.67 |

Cả 2 FP còn lại là `highlight_clipping` mức 1 do VLM báo (loại mới) → đã thêm sàn mức 2 cho lỗi
không sửa được; dự kiến FP sạch 0/13, **chưa đo lại**.

## Đang dở — dừng do hệ thống thiếu bộ nhớ

Hai lần chạy nền bị hệ thống dừng (low memory), chưa có kết quả, **chưa chạy lại** (chờ người dùng):
1. `fused` lần 2 trên bộ thật (prompt mới → ~32 perceive + ≤ 32 verify): chạy lại bằng
   `python scripts/benchmark_analysis.py --set real --config fused --vlm-cache data/real/vlm_cache --out data/real/benchmark_results/fusion-r2-fused`
   (ảnh đã xong nằm trong cache). Sau đó `vlm_only`, `fused_noverify` dùng lại cache (0 lời gọi).
2. Hồi quy pipeline offline (`run_pipeline`, 3 vòng, 32 ảnh) base `dec2d3d` so với nhánh này.
   Chạy hai lần tuần tự, không song song với benchmark.

## Khoảng cách tới tiêu chí (theo r1)
- FP sạch ≤ 10 %: dự kiến đạt sau sàn mức 2 (cần đo lại).
- Recall ≥ 85 %: **chưa đạt** (7/15). Ám màu 0/3: số đo nguồn sáng cho ~0 trên cả 4 ảnh ám màu
  thật (góc 0–2.6°; color_cast_03 không có pixel trung tính), gray-world không tách được ảnh ám
  khỏi ảnh sạch nên không thêm giả thuyết. Thiếu sáng bị VLM coi là low_key (backlit_02,
  low_light_01) → preserved theo bảng.
- MAE ≤ 0.6: **chưa đạt** (1.00): số đo và VLM đều cho underexposed mức 3 trên ảnh nhãn mức 1;
  thang hiệu chỉnh synthetic lệch so với nhãn thật (reliability underexposed 0.11) — việc của calib.
- D3 (`rejected` làm bằng chứng yếu): đã có config `fused_n3` / `fused_n3_weak`; chưa chạy, chỉ
  kết luận trên hold-out.

Chưa ghi LANE DONE.
