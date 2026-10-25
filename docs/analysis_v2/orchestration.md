# Analysis v2 — Tổ chức multi-agent (Herdr)

Kế hoạch: `docs/plan_analysis_v2.md`. Hợp đồng interface: `docs/analysis_v2/contracts.md`.
Brief từng agent: `docs/analysis_v2/agents/<lane>.md`. Script khởi động:
`scripts/herdr/start_analysis_v2.ps1`.

## Nhánh

```
develop
  └─ feat/analysis-v2                 ← nhánh tích hợp (lead merge vào đây)
       ├─ feat/analysis-v2-bench      ┐
       ├─ feat/analysis-v2-measure    │ wave 1 (song song)
       ├─ feat/analysis-v2-region     │
       ├─ feat/analysis-v2-meta       ┘
       ├─ feat/analysis-v2-calib      ┐ wave 2 (sau khi measure + bench đã merge)
       ├─ feat/analysis-v2-agent      ┘
       └─ feat/analysis-v2-fusion       wave 3 (sau khi agent + calib đã merge)
```

Mỗi lane có worktree riêng ở `..\iip-wt\<lane>` (cạnh repo chính), một pane Herdr chạy Claude Code.

## Lane

| Wave | Lane | Phase | Sở hữu (chỉ sửa trong phạm vi này) | Phụ thuộc |
|---|---|---|---|---|
| 1 | `bench` | P0.2–0.4 | `scripts/build_calibration_set.py`, `scripts/benchmark_*.py`, `data/real_benchmark_manifest.json`, `tests/unit/test_benchmark_*.py`, `docs/analysis_v2/baseline.md` | — |
| 1 | `measure` | P1 | `src/analyzer_evaluator/measurement/**`, `tests/unit/test_measurement_*.py` | — |
| 1 | `region` | P4a | `src/region_engine/**`, `tests/unit/test_region_*.py`, `tests/integration/test_prompt_segmentation_model.py`, `requirements-semantic.txt`, `scripts/fetch_segmentation_models.py`, `.gitignore`, `tests/fixtures/prompt_segmentation/assets.json` | — |
| 1 | `meta` | P3 | `src/analyzer_evaluator/metadata.py`, `tests/unit/test_metadata*.py`, plumbing `image_metadata` (graph, state `DoctorState`, `src/api/`, `frontend/app.py`) | — |
| 2 | `calib` | P2 | `measurement/calibration.py`, `calibration.yaml`, `scripts/calibrate_severity.py`, ngưỡng trong `analyzer.py`, `knowledge/engine.py` (chỉ chỗ đọc ngưỡng), phần motion của `measurement/sharpness.py`, chọn pixel trung tính trong `measurement/color.py` | bench (bộ synthetic), measure |
| 2 | `agent` | P4b + P5 | `src/agent/perception.py`, `state.py`, `vlm_diagnostician.py`, `executor.py` (resolve mask), `src/agent/evidence.py`, `session.py` + route `create_session` (metadata), test tương ứng | measure, region, meta |
| 3 | `fusion` | P6 + P7 | `src/agent/fusion.py`, `measurement/diff.py`, `state.py`, `perception.py`, `planner.py`, `intent.py`, `graph.py` (`decide_node`), knowledge cards, `SYSTEM_PROMPT` | agent, calib |
| cuối | lead | P8 | ADR-005, `docs/interfaces.md`, `CLAUDE.md` | tất cả |

Lane cần đổi file ngoài phạm vi: ghi vào `docs/analysis_v2/contract-changes.md` và báo lead,
không tự sửa.

## Luồng

```
 wave 1:  bench ──┐   measure ──┐   region ──┐   meta ──┐
                  ▼             ▼            ▼          ▼
          lead review + merge vào feat/analysis-v2 (ruff + pytest tests/unit xanh)
                  │
 wave 2:  calib ──┴── agent
                  ▼
          lead merge; chạy benchmark --config baseline/vlm_only để so
                  │
 wave 3:  fusion (P6 → P7), điền config measure_only + fused vào benchmark
                  ▼
          lead merge; chạy benchmark đầy đủ; P8 docs; hỏi người dùng trước khi push/PR
```

## Quy tắc chung cho mọi lane

1. Đọc `CLAUDE.md`, `docs/plan_analysis_v2.md` (phase của mình), `contracts.md`, brief của mình.
2. Commit nhỏ, Conventional Commits, **không** thêm Claude làm tác giả/co-author.
   Không push, không mở PR, không merge — lead làm.
3. Trước mỗi commit: `ruff check src/ tests/ --fix`, `ruff format src/ tests/`,
   `python -m pytest tests/unit/ -q`. Không commit khi đỏ.
4. Không gọi Gemini thật trong unit test. Lane được phép gọi thật (bench, agent, fusion) chỉ gọi
   trong script benchmark, có `--limit`, và ghi số lần gọi vào báo cáo.
5. Không commit model, ảnh benchmark, `.env`, kết quả benchmark.
6. Không thêm AI sinh pixel (ADR-001), không dependency chỉ chạy CUDA (ADR-004), không nâng numpy.
7. Bị chặn (thiếu quyết định, hợp đồng sai, test của lane khác đỏ): ghi `docs/analysis_v2/status/<lane>.md`
   mục **BLOCKED** rồi dừng, chờ lead.
8. Xong: cập nhật `docs/analysis_v2/status/<lane>.md` (đã làm, tiêu chí đạt chưa, số đo, việc còn lại,
   commit cuối) và dừng ở dòng cuối: `LANE DONE`.

## Môi trường trong worktree

Script khởi động copy vào mỗi worktree: `.env` (key Gemini), `data/real/` (ảnh benchmark), fixture
Penn-Fudan. Model phân vùng không copy: biến môi trường `REGION_DINO_MODEL_PATH` và
`REGION_MOBILE_SAM_CHECKPOINT` trỏ về `models/segmentation/` của repo chính.
