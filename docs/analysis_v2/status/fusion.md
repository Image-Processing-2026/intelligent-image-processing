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

## Lần chạy lại 2026-10-26 (tuần tự, kiểm tra RAM trước mỗi job: 10.5 GB và 11.4 GB trống)

### Job 1 — `fused` r2 trên bộ thật: **BLOCKED do hết quota Gemini**
`fusion-r2-fused`: 51 lời gọi thật + 24 cache hit. Từ ca 16 (`multi_defect_03`) trở đi mọi lời gọi
trả `429 RESOURCE_EXHAUSTED` (quota): **17/32 ca rơi về luật** (toàn bộ `intentional_style`,
`already_good`, `group_small_faces`), nên **FP ảnh sạch chưa đo được** (n = 0 ca sạch được chấm).
Trên 15 ca có VLM thật: precision 1.00 (0 FP), recall 7/15 (noise 4/4, underexposed 3/5; ám màu
0/3, backlit 0/1, blur 0/1, low_contrast 0/1). `backlit_01` hết FP (verify: intentional).
Cache giữ 15 ca đã xong (cả verify): chạy lại cùng lệnh khi có quota chỉ tốn ~17 perceive + ≤ 17
verify.

Lần thử lại (2026-10-26, sau khi reporter thử quota thành công; từng ảnh một, nghỉ 20 s, dừng ở
429 đầu tiên): **dừng ngay ở ảnh đầu tiên `multi_defect_03`, 0/17 ảnh xong**, 1 lời gọi bị từ chối.
Lỗi nguyên văn: `429 RESOURCE_EXHAUSTED … Quota exceeded for metric:
generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 500, model:
gemini-3.5-flash-lite, description: Generate content requests per day limit exceeded on Free Tier …
quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier … Please retry in 5h6m18s`
(retryDelay 18378 s). Đây là **giới hạn theo ngày** (500 request/ngày/project/model, Free Tier),
không phải theo phút: nghỉ giữa các ảnh không giúp được. Lệnh chạy lại khi quota hồi:
`bash` vòng từng ảnh như trên (17 ảnh `multi_defect_03` … `intentional_style_10`), rồi
`python scripts/benchmark_analysis.py --set real --config fused --vlm-cache data/real/vlm_cache --case-retries 0 --out data/real/benchmark_results/fusion-r3-fused`
để gộp (toàn bộ từ cache, 0 lời gọi).

### Job 2 — hồi quy pipeline offline (`run_pipeline`, 3 vòng, 32 ảnh, không key)

| | base `dec2d3d` | nhánh fusion |
|---|---|---|
| Quyết định cuối | SHIP 21, STOP_BEST_EFFORT 11 | SHIP 26, STOP_BEST_EFFORT 6 |
| Plan rỗng ở vòng 1 | 11 | 17 |
| Tổng số vòng / rollback | 56 / 1 | 52 / 3 |
| Thời gian | 372 s | 227 s |

Khác biệt: 6 ảnh (`already_good_02/03`, `group_small_faces_02/03`, `low_light_03`, `noisy_03`) hết
`backlit_subject@face` báo sai do ngưỡng tay cũ → không còn gamma trên mặt, SHIP thay vì STOP.
2 ảnh low-key (`intentional_style_08`, `_10`): luật offline (không có preserve) làm sáng toàn ảnh,
diff P7 thấy nhiễu vùng tối tăng mức 3/2 → rollback, STOP sau 1 vòng thay vì 3 vòng làm sáng.

### Ám màu đo ≈ 0 trên ảnh thật — lỗi đo hay không? (cho lead quyết định lane tiếp)
`measure_color` trên 4 ảnh ám màu (và vài ảnh sạch để so):

| Ảnh | Nhãn | góc nguồn sáng | ln(b/r) | pixel C* < 12 / < 30 |
|---|---|---|---|---|
| color_cast_01 | cool 1 | 2.6° (tin cậy 0.71) | +0.11 | 63 % / 100 % |
| color_cast_02 (ambiguous) | cool 1 | 7.6° (0.48) | −0.29 | 10 % / 45 % |
| color_cast_03 | warm 3 | **không ước lượng** | – | **0 % / 1.1 %** |
| multi_defect_01 | cool 1 | 2.2° (0.66) | −0.08 | 62 % / 86 % |
| intentional_style_05 (hoàng hôn) | – | 21.6° | −1.00 | 55 % / 89 % |
| intentional_style_09 (hoàng hôn) | – | 15.6° | −0.72 | 4 % / 21 % |

Ngưỡng hiệu chỉnh: cool mức 1 khi ln(b/r) ≥ 0.49 (≈ 14°, khớp trên 5 ca synthetic, reliability
0.16); warm mức 1 khi ln(r/b) ≥ 0.35.
1. **Lỗ hổng đo thật (strong cast):** `color_cast_03` không có pixel nào C* < 12 và chỉ 1.1 % C* < 30,
   nên bước chọn pixel trung tính (khởi đầu từ C* < 12, chỉ nhận lại C* gốc < `CAST_MAX_CHROMA`
   = 30) không bắt đầu được → không có ước lượng → `(0, 0.0)` "không đo được". Ám rất mạnh trên ảnh
   thật vượt giả định ≤ 3000 K của synthetic. Đề xuất (lane `measure`/`calib`): khi tập trung tính
   rỗng, nới cổng C* dần hoặc dùng Gray-Edge/Shades-of-Gray trên mọi pixel không cháy với confidence
   thấp, rồi để fusion verify.
2. **Không phải lỗi code (ám nhẹ):** `color_cast_01`, `multi_defect_01` có nhiều pixel gần trung tính
   và ước lượng ổn định ~2–3°, tức ám nhẹ thật hoặc không có; nhãn mức 1 thấp hơn rất xa ngưỡng
   mức 1 của calibration (thang synthetic, chỉ 5 ca cool dương tính). `multi_defect_01` còn ngược
   hướng (hơi ấm theo đo, nhãn cool): có thể nhãn chủ quan (thảm xanh chiếm khung) hoặc nguồn sáng
   thật trung tính. Cần nhãn ám màu mức nhẹ trên ảnh thật/synthetic rồi hiệu chỉnh lại, không phải
   sửa bộ ước lượng.
3. Hoàng hôn đo đúng là ấm mạnh (−1.0, −0.72): bộ ước lượng chạy đúng khi có pixel trung tính; FP
   ám ấm trên ảnh chủ ý được preserve/verify chặn ở fusion.

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

Chưa ghi LANE DONE: FP ảnh sạch (tiêu chí chính) chưa đo được vì hết quota Gemini theo ngày.
