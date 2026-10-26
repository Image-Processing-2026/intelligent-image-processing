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
| 3 | fusion | P6 + P7 | – | – | mở 2026-10-25 theo quyết định người dùng (sau checkpoint không đạt); brief mục 9–14 |
| 4 | color | P2 bổ sung | – | – | mở 2026-10-26 (pane `wE:p1`), song song `fusion`; brief `agents/color.md`; không dùng hold-out |

Sau mỗi merge: `ruff check src/ tests/` sạch; `pytest tests/unit/ -q`: 1173 → 1243 → 1267 → 1296
passed (sau bench / measure / region / meta), 1301 (+ `benchmark_measurement`), 1344 (calib),
1381 (agent).

## Số đo

| Bộ | Config | Recall | FP rate ảnh sạch | MAE severity | Nguồn |
|---|---|---|---|---|---|
| real (30 ảnh chấm, 13 sạch/chủ ý) | baseline | 0.47 | 0.54 | 1.00 | `baseline.md` |
| real | vlm_only | 0.53 | 0.62 | 0.88 | `baseline.md` |
| real, sau calib (trước agent) | baseline | 0.33 (5/15) | 0.39 (5/13) | 0.80 | `checkpoint_calib.md` |
| real, sau calib | measure_only | 0.33 (5/15) | 0.77 (10/13) | 0.80 | `checkpoint_calib.md` |
| synthetic, sau calib | measure_only | 0.64 | 0.05 (1/20) | 0.14 | `checkpoint_calib.md` |
| real, lane agent (prompt chỉnh trên bộ này) | vlm_only v2 | 0.47 (7/15) | 0.00 (0/13) | 1.14 | `status/agent.md` |
| real | fused | – | – | – | chờ `fusion` |

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

## Review `fusion` trước merge (lead, 2026-10-26; chưa merge)

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
- Lane đề xuất `colorcast` (chưa mở, chờ reporter và người dùng): color_cast_03 không có pixel trung
  tính (C* < 12) nên không đo được; ám màu nhẹ (color_cast_01, multi_defect_01) là lệch thang
  hiệu chỉnh trên ảnh thật (ngưỡng cool mức 1 ln(b/r) ≥ 0.49 khớp từ 5 ảnh synthetic).

## Review cần người

- P1/P2 (`measure`, `calib`): Person 1.
- P4a (`region`): Person 2.
