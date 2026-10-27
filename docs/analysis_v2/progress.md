# Analysis v2 — tiến độ (lead)

Nhánh tích hợp `feat/analysis-v2`. Chưa push, chưa mở PR.

## Lane

| Wave | Lane | Phase | Merge | Tiêu chí | Ghi chú |
|---|---|---|---|---|---|
| 1 | bench | P0.2–0.4 | `803a980` | đạt (trừ sai số synthetic → lead đo, xem dưới) | ngoài phạm vi nhưng chấp nhận: `.gitignore`, `data/calibration_sources.json`, `data/download_real_benchmark.py` (tải full-res), `test_build_calibration_set.py`. Chú thích `annotated_by: "agent"` trong manifest chưa được người duyệt. |
| 1 | measure | P1 | `efe536b` | đạt | lệch khỏi plan: quyết định trong `measurement_synthetic.md`; numpy 1.26.4: 70/70 xanh |
| 1 | region | P4a | `d972c22` | đạt | contract-changes 1 (A), 2 (chấp nhận), 3 (chấp nhận). Cần Person 2 review. |
| 1 | meta | P3 | `a908bc0` | đạt | phiên tương tác → lane `agent`; Gradio hoãn |
| 2 | calib | P2 | `526884d` | đạt (MAE synthetic 0.14; FP already_good không tăng) | thêm xác nhận mờ chuyển động, chọn pixel trung tính lặp, D4. contract-changes 5 (`long_side`) chờ người dùng xác nhận; 6.3 giao `fusion`. Cần Person 1 review. |
| 2 | agent | P4b + P5 | `d5dd271` (merge **sau** checkpoint, theo lệnh người dùng) | đạt | sửa ngoài phạm vi chấp nhận (contract-changes 7). Prompt rev 2 chỉnh trên chính bộ thật. N = 3 chưa cho lợi ích đo được (đề nghị xem lại D3 ở `fusion`). |
| 3 | fusion | P6 + P7 | `a20d674` (+ doc `c372ef8`) | một phần: FP sạch đạt (1/13, r3); recall 0.47, MAE 1.00 chưa đạt | mở 2026-10-25 theo quyết định người dùng; brief mục 9–14; contract-changes 8 chấp nhận; xung đột `benchmark_analysis.py` giải sẵn (`lead/fusion-merge-prep`). Ngưỡng backlit/P7 cần Person 1 review. |
| 4 | color | P2 bổ sung | `1cc70fd` | đạt (color_cast_03 warm 3; FP ám màu sạch 4 → 4; hoàng hôn không thành cool) | mở 2026-10-26 (pane `wE:p1`); không dùng hold-out, 0 lời gọi Gemini; contract-changes 9 chấp nhận, key ghi vào §C1; ngưỡng giữ (mức 1 không tách được). Cần Person 1 review `measurement/color.py`. |

Sau mỗi merge: `ruff check src/ tests/` sạch; `pytest tests/unit/ -q`: 1173 → 1243 → 1267 → 1296
passed (sau bench / measure / region / meta), 1301 (+ `benchmark_measurement`), 1344 (calib),
1381 (agent), 1429 (fusion), 1439 (color; coverage `analyzer_evaluator` 97.6 %).

## Số đo

| Bộ | Config | Recall | FP rate ảnh sạch | MAE severity | Nguồn |
|---|---|---|---|---|---|
| real (30 ảnh chấm, 13 sạch/chủ ý) | baseline | 0.47 | 0.54 | 1.00 | `baseline.md` |
| real | vlm_only | 0.53 | 0.62 | 0.88 | `baseline.md` |
| real, sau calib (trước agent) | baseline | 0.33 (5/15) | 0.39 (5/13) | 0.80 | `checkpoint_calib.md` |
| real, sau calib | measure_only | 0.33 (5/15) | 0.77 (10/13) | 0.80 | `checkpoint_calib.md` |
| synthetic, sau calib | measure_only | 0.64 | 0.05 (1/20) | 0.14 | `checkpoint_calib.md` |
| real, lane agent (prompt chỉnh trên bộ này) | vlm_only v2 | 0.47 (7/15) | 0.00 (0/13) | 1.14 | `status/agent.md` |
| real, lane fusion (dev, chỉnh trên bộ này) | fused r3 | 0.47 (7/15) | 0.08 (1/13) | 1.00 | `status/fusion.md` |
| real, lane color | measure_only | 0.40 (6/15) | 0.77 (10/13) | 0.67 | `status/color.md` |
| **hold-out** (20 ảnh đã duyệt, 11 sạch/không sửa) | fused (`3c5af72`+`6f40eb1`) | **0.18 (2/11)** | **0.09 (1/11)** | 1.00 (2 cặp) | mục "Chấm hold-out" |

Mục tiêu cuối: FP ≤ 0.10, recall ≥ 0.85, MAE ≤ 0.6.

Sai số ước lượng P1 trên synthetic (chi tiết `measurement_synthetic.md`): σ nhiễu 5.8 % MAE;
σ defocus 13.1 % (lệch thấp); độ rộng cạnh tương đối 7.4 %; hướng mờ chuyển động 10.9° (sai trên
texture có hướng); nguồn sáng 5.5° (ám mạnh bị đo thiếu).

## Checkpoint (người dùng đặt)

Sau khi merge `calib`: chạy lại benchmark. Nếu FP rate ảnh sạch (already_good + intentional_style)
không giảm rõ từ 0.54 về phía ≤ 0.10 → dừng, báo người dùng trước khi mở wave 3 (`fusion`).

**Kết quả (2026-10-25):** FP ảnh sạch 7 → 5 / 13 (0.54 → 0.39) — không giảm rõ về phía ≤ 0.10.
11/17 FP còn lại là `backlit_subject` origin `measured` từ ngưỡng tay cũ trong `perception`. Đã
merge `agent` theo thứ tự người dùng đặt; **chưa mở wave 3**, chờ người dùng.

## Quyết định sau checkpoint (người dùng, 2026-10-25)

1. Mở wave 3 (`fusion`) ngay, song song với việc chuẩn bị bộ hold-out. Hold-out chỉ dùng để chấm
   kết quả cuối của fusion.
2. Bộ hold-out: 15–20 ảnh thật **mới** (không nằm trong `data/real`, không dùng chỉnh prompt rev 1/2
   hay calib). Nhãn ở dạng mẫu, mọi nhãn `unreviewed` cho tới khi người dùng hoặc thành viên nhóm
   duyệt. Không chỉnh gì trên bộ này.
3. FP `backlit_subject` đo được: phạm vi `fusion` mở rộng sang `perception._measured_defects`;
   lỗi đo được phải qua bảng fusion, không phát ra một mình.
4. Chấp nhận `contract-changes.md` mục 5 (`long_side` tùy chọn, đã ghi vào `contracts.md` §C4).
   Sửa ngoài phạm vi của lane `agent` (`graph.py`, `api/routes.py`, `api/main.py`,
   `frontend/app.py`, `tests/conftest.py`): lead đã review, `pytest tests/unit/` xanh (1381) →
   chấp nhận (mục 7).
5. D3: `PERCEIVE_SAMPLES` mặc định 1, ghi đè bằng env `PERCEIVE_SAMPLES`; giữ code gộp; config
   benchmark `vlm_only_n3`. Fusion đánh giá `rejected` làm bằng chứng yếu, kết luận trên hold-out.

## Review `fusion` trước merge (lead, 2026-10-26; đã merge `a20d674`)

Review tới `0c1a82f`. **Chưa merge**: FP ảnh sạch sau sàn mức 2 chưa đo (lần chạy r2 chờ quota
Gemini). Merge ngay khi `status/fusion.md` kết thúc bằng `LANE DONE`.

**Xung đột.** Thử merge (`git merge-tree`): chỉ `scripts/benchmark_analysis.py`; `contract-changes.md`
tự gộp. Đã giải sẵn trên nhánh cục bộ `lead/fusion-merge-prep` (`77ca85d`, worktree tạm trong
scratchpad), `ruff` sạch, `pytest tests/unit/` 1400 passed + 29 skipped (skip chỉ do worktree tạm thiếu
ảnh mẫu `data/real/*.jpg`). Quy tắc giải:
- `run_benchmark` dùng `load_cases(...)` của nhánh tích hợp: giữ chặn nhãn chưa duyệt và
  `--reviewed-only`. Bản của fusion đọc hold-out trực tiếp, bỏ qua chặn này; không lấy.
- `load_real_cases`: lấy bản fusion (đọc manifest một lần, nhận `image_dir` của manifest).
- CLI: giữ nhóm loại trừ `--allow-unreviewed` / `--reviewed-only` và thêm `--vlm-cache`; đặt
  `VLM_CACHE` trước kiểm tra key; điều kiện cần key giữ `dataset != "synthetic"` (tương đương).
- Docstring: thêm ghi chú hold-out từ chối nhãn chưa duyệt.
Khi merge thật: nếu fusion không sửa `scripts/benchmark_analysis.py` sau `0c1a82f` thì lấy file từ
`lead/fusion-merge-prep`; nếu có sửa thì giải lại theo các quy tắc trên. Sau merge xóa nhánh và
worktree tạm.

**Tài liệu (P8, một phần).** `docs/interfaces.md` và `CLAUDE.md` đã đồng bộ với luồng đã merge
(`55172e8`: đo + hiệu chỉnh, mask chủ thể, crop bằng chứng, perceive v2 với `PERCEIVE_SAMPLES = 1`,
metadata). Phần fusion/P7 nằm sẵn ở commit `f803da6` trên `lead/fusion-merge-prep`; sau khi merge
fusion thì `git cherry-pick f803da6`. ADR-005 vẫn chờ cuối P8.

**contract-changes mục 8: chấp nhận cả 6 điểm** (lead review; báo người dùng qua reporter):
1. `graph.plan_treatment` gọi `apply_contraindications` sau preserve guard: đúng brief mục 5. Chỉ bỏ
   action, không sắp xếp lại. `no_brighten` chặn `gamma > 1` toàn ảnh: đúng quy ước
   `exposure_contrast` (gamma > 1 làm sáng).
2. `retriever.match_cards` lấy mọi lỗi mức ≥ 1: cần cho card tư vấn. Đã kiểm: 4 card trong
   `limits.yaml` đều `auto_apply: false`, không recipe, nên luật dự phòng không sinh action.
3. `fusion_notes` ra khỏi prompt Plan: chấp nhận.
4. `tests/conftest.py` tắt verify mặc định: chấp nhận. Test pipeline cũ vì vậy không đi qua verify;
   verify chỉ được kiểm trong `test_agent_fusion.py`.
5. `scripts/benchmark_analysis.py`: chấp nhận, với cách giải xung đột ở trên.
6. Test lane khác sửa theo hành vi mới: chấp nhận. Test `_measured_defects` cũ bỏ vì hàm bỏ, bảng
   fusion có test riêng mỗi dòng.
Thay đổi hành vi (lỗi không sửa được → plan rỗng → SHIP; tác dụng phụ ≥ 2 → rollback ảnh thực;
vòng ≥ 2 không gọi VLM khi mọi lỗi đo được; `DEFECT_TYPES` thêm 4 loại D2): chấp nhận, ghi vào
`docs/interfaces.md` khi merge.

**Còn mở, không chặn merge:** ngưỡng backlit của fusion (`BACKLIT_MIN_GAP_L = 20`, mốc L* mặt 60 /
chủ thể 50) và ngưỡng tác dụng phụ P7 là prior, chưa hiệu chỉnh (Person 1 review). Số r1 của fusion
đo trên chính 30 ảnh dev đã dùng để chỉnh (lạc quan). D3 (`fused_n3`, `fused_n3_weak`) chỉ kết luận
trên hold-out.

**Merge (2026-10-26).** Fusion chỉ thêm `f66e75f` (status, r3) sau `0c1a82f`, không sửa
`scripts/benchmark_analysis.py` → lấy file từ `lead/fusion-merge-prep`; cây merge khớp `08450b2`
ngoài `progress.md` và `status/fusion.md`. Doc fusion cherry-pick thành `c372ef8`. `ruff` sạch,
`pytest tests/unit/` 1429 passed.

## Merge `color` (lead, 2026-10-26, `1cc70fd`)

Review theo brief: không đọc hold-out, 0 lời gọi Gemini; chỉ 6 ảnh đổi kết quả đo (từ "không ước
lượng" sang có ước lượng), không thêm FP ám màu; FP ám màu ảnh sạch 4/13 trước và sau (is_01, is_05,
is_08, is_09, đều có từ trước); hai hoàng hôn vẫn ấm, không cool. `calibration.yaml` → `p2-20261026`:
mọi ngưỡng giữ, chỉ đổi số khớp (warm reliability 0.49 → 0.64, cool 0.16 → 0.09 vì biến thể cỏ giờ
có ước lượng). Ca dò 5000 K / 7000 K không vào khớp. Xung đột: chỉ `contract-changes.md` (mục 8 và 9
cùng nối cuối), giữ cả hai. `ruff` sạch, `pytest tests/unit/` 1439 passed, coverage
`analyzer_evaluator` 97.6 %.

**contract-changes 9 (key `illuminant_method`, `illuminant_gate_chroma`): chấp nhận**, đã ghi vào
§C1 của `contracts.md`.

**Đề xuất "ám màu có `illuminant_method ≥ 1` đi verify": người dùng duyệt làm trước hold-out; lead
đã làm (xem "Sau merge" bên dưới).** Phân tích ban đầu:
Bảng fusion hiện tại:
- Số đo báo ám mà VLM không nhắc → đã luôn đi verify (mọi `illuminant_method`). Không cần đổi.
- Không có VLM (luật dự phòng) → cần confidence ≥ 0.5; đường 1 (color_cast_03: 0.375) và đường 2
  (≤ 0.2) bị loại. An toàn, không cần đổi.
- Chỗ hở duy nhất: VLM báo ám **và** số đo đường dự phòng ≥ 1 → "vlm + measured agree", xác nhận
  thẳng, severity lấy từ số đo (VLM ±1). Với ước lượng kém tin cậy thì nên hỏi verify (câu hỏi thêm
  vào lời gọi verify sẵn có của ảnh) hoặc để severity theo VLM.
- Cần: fusion đọc `illuminant_method` từ `MeasurementReport.color` (không có trong `RegionStats`, nên
  chỉ dùng được cho vùng `full`), một test offline. Phạm vi `fusion.py`, nhỏ.
- **Thời điểm:** phải quyết trước khi chấm hold-out (đổi luật sau khi xem hold-out là chỉnh trên
  hold-out). Trên dev chỉ một ảnh đi đường dự phòng (color_cast_03), nên không đo được lợi ích; đây
  là luật an toàn, không phải luật chỉnh số. Chờ người dùng chọn: làm trước hold-out, hay đóng băng
  như hiện tại.

**Person 1 review** `src/analyzer_evaluator/measurement/color.py` (đường dự phòng, hằng
`RELAXED_GATES`, `ALL_PIXELS_CONFIDENCE` là prior chưa hiệu chỉnh) cùng P1/P2.

**Nhãn ám màu dev cần người dùng xem lại** (không chặn, không tự sửa): `multi_defect_01` (cool 1, đo
ln(b/r) −0.08, hơi ấm) và `color_cast_02` (ambiguous, cool 1, đo −0.29, hơi ấm). Cả hai nằm giữa
phân bố ảnh sạch (±0.4). Đổi nhãn không đổi ngưỡng (mức 1 vốn không tách được) mà chỉ đổi recall
dev; nếu người dùng duyệt lại, ghi `reviewed_by` trong manifest dev.

## Sau merge (lead, 2026-10-26, trước hold-out)

1. **Nhãn dev (`0f228fb`, người dùng duyệt):** `color_cast_02` ambiguous true → false (ám lạnh mức 1
   xác nhận); `multi_defect_01` ambiguous false → true (không chắc). Không đổi defects. Lưu ý:
   ambiguous bỏ chấm cả ảnh, nên 3 lỗi khác của `multi_defect_01` (underexposed, noise, blur) cũng
   ra khỏi phép chấm dev; số dương tính dev thay đổi, số cũ không so thẳng được.
2. **Fusion, ám màu kém tin cậy:** `Measured.low_trust` = ám màu đo bằng đường dự phòng
   (`color.illuminant_method` ≥ 1, chỉ báo cáo toàn ảnh). VLM báo ám + số đo đó ≥ 1 → verify
   (defect → confirmed, severity theo VLM; intentional → preserved; texture/not_present → rejected;
   verify lỗi → giữ lời VLM, confidence × 0.5). Đường chuẩn giữ dòng 1. 4 test offline mới trong
   `test_agent_fusion.py`. `ruff` sạch, `pytest tests/unit/` 1444 passed.
   Tác động trên dev (offline, không gọi Gemini): chỉ 2 ảnh đi đường dự phòng: `color_cast_03`
   (warm 3) và `intentional_style_08` (cool 1, ảnh sạch). Ở `fusion-r3-fused` VLM không báo ám màu
   trên cả hai, nên với cùng phản hồi VLM nhánh mới không chạy. Luật chỉ đổi "confirmed thẳng"
   thành kết quả verify, nên không thể thêm FP: FP ảnh sạch dự kiến giữ 1/13 (chưa chạy lại, cần
   Gemini). Ghi nhận (không chỉnh): ở r3 VLM giữ `warm_tone` trên `color_cast_03`, nên sau merge
   color, giả thuyết đo warm 3 sẽ bị preserve chặn; ảnh này vẫn có thể là FN.

## Chấm hold-out (lead, 2026-10-26, theo lệnh người dùng)

`python scripts/benchmark_analysis.py --set holdout --config fused --reviewed-only --vlm-cache
data/real/vlm_cache_holdout --out data/real/benchmark_results/holdout-fused` (HEAD `6b7ac00`).
20/20 ảnh `source=vlm`, `gemini-3.5-flash-lite` (cùng model với dev r3), 30 lời gọi, không 429,
không fallback. Chỉ chạy `fused`; `fused_n3*` (D3) chưa chạy.

| | dev r3 (chỉnh trên bộ này) | **hold-out** | mục tiêu |
|---|---|---|---|
| FP ảnh sạch | 1/13 (0.08) | **1/11 (0.09)** | ≤ 0.10 — đạt |
| Recall | 7/15 (0.47) | **2/11 (0.18)** | ≥ 0.85 — không đạt |
| MAE severity | 1.00 | 1.00 (2 cặp) | ≤ 0.6 — không đạt |
| Precision | 0.70 | 0.33 (2/6) | |
| Preserve recall | 0.67 | 1.00 (7/7) | |

Đúng: `noisy_01` noise 2 (khớp mức), `blur_01` blur (đo 1, nhãn 3).
FP ảnh sạch duy nhất: `cast_01` (ca không sửa) underexposed@road 2 (VLM 2 + đo 3 đồng ý).
FP trên ảnh có lỗi: `blur_01` backlit_subject@full 3 (VLM, không đo được), `blur_02` overexposed@cat
2 và highlight_clipping@cat 3 (lấy nét sau mèo bị đọc thành cháy sáng).

**9 lỗi bỏ sót, theo nguyên nhân** (chỉ ghi nhận, không chỉnh trên hold-out):
1. **Preserve chặn lỗi đo được (3/9):** VLM gán ý đồ, preserve thắng số đo. `backlit_02` (đo
   backlit 2) → `silhouette`; `backlit_03` (đo 3) → `silhouette` + `ask_user`; `multi_01` (đo
   underexposed 3) → `low_key`. Bảng fusion cho preserve thắng mọi nguồn, không hỏi verify.
2. **Ám màu đọc thành phong cách (1/9):** `cast_02` (cool 2): đo không ra mức ≥ 1, VLM gán
   `cool_tone`.
3. **Nhiễu/mờ nhẹ không thành giả thuyết (5/9):** noise ở `noisy_02`, `multi_01`, `multi_02`; blur ở
   `multi_02` và `blur_02` (vùng mèo). Số đo dưới mức 1 (hoặc không đo vùng `cats`), VLM không nhắc.
   Trùng kết luận dev: thang hiệu chỉnh synthetic lệch so với nhãn thật mức 1. Ba ảnh này VLM cũng
   gán `low_key`.
4. `ask_user` (`backlit_03`) bị chấm là bỏ sót; trong phiên tương tác ảnh này sẽ hỏi người dùng.

Nhận xét: FP ảnh sạch giữ được ngoài bộ dev, nhưng recall tụt 0.47 → 0.18: số dev lạc quan (prompt
rev 2 chỉnh trên dev), và hệ thống nghiêng hẳn về "giữ nguyên". Mọi thay đổi tiếp theo phải chỉnh
trên dev/synthetic; hold-out này đã dùng, lần chấm sau trên nó không còn là đánh giá độc lập.

## Bộ hold-out (2026-10-25)

19 ảnh Commons mới (`data/holdout_manifest.json`, hướng dẫn `holdout.md`): already_good 3,
intentional_style 6 (low-key ×2, high-key, hoàng hôn ×2, hạt phim), backlit 2, noisy 2 (NIND), blur 2,
color_cast 2, multi_defect 2. Nhãn nháp do lead agent đặt từ thumbnail.

Người dùng (Tiên) đã duyệt cả 19 ảnh ban đầu. `holdout_backlit_01` và `holdout_cast_01` thành ca
"không sửa" (không có lỗi mong đợi). `min_severity` giữ bằng 1 như nháp (bản xuất từ công cụ duyệt
từng đặt nhầm bằng severity, đã hoàn lại). Thêm `holdout_backlit_03` (chân dung ngược sáng rõ, CC0),
đã duyệt (severity 2, min 1). Cả 20 ảnh đã duyệt; chấm bằng `--set holdout --reviewed-only`.
Backlit có hai ảnh có lỗi (`holdout_backlit_02` mức 1, `holdout_backlit_03` mức 2) và một ca
không sửa (`holdout_backlit_01`): vẫn là mẫu mỏng. Chưa chạy config nào trên bộ này; không chỉnh gì theo các nhãn này.

## Hoãn

- Gradio (`frontend/app.py`) chưa nhận metadata: `gr.Image(type="numpy")` bỏ bytes file. Không
  đổi sang `type="filepath"` lúc này (người dùng quyết định, 2026-10-25).
- `measure`: zone histogram 11 vùng, vị trí blob cháy, Tenengrad (xem `measurement_synthetic.md`).
- Ám màu mức 1 không đo được bằng ngưỡng nguồn sáng tuyệt đối (lane `color`): chỉ đo tương đối
  (vòng ≥ 2 qua `measurement/diff.py`, hoặc chủ thể so với nền). Để VLM + verify.

## Review cần người

- P1/P2 (`measure`, `calib`, `color`): Person 1; kèm ngưỡng backlit và tác dụng phụ P7 của `fusion`.
- P4a (`region`): Person 2.
