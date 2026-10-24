# Lane `agent` — P4b + P5: nối đo/vùng vào Perceive, bằng chứng cho VLM (wave 2)

Nhánh `feat/analysis-v2-agent`, tạo từ `feat/analysis-v2` **sau khi** `measure`, `region`, `meta` đã
merge. Đọc trước: `CLAUDE.md` (đặc biệt phần mock Gemini), `docs/plan_analysis_v2.md` §Phase 4,
§Phase 5, §7; `contracts.md` §C1 §C2 §C3 §C6; status của các lane wave 1.

## P4b — nối vào Perceive
1. Perceive schema/prompt: `subjects` → `[{name, box_2d}]` (C6); plan prompt vẫn cấm bbox.
2. `perception.measure_regions` → dùng `resolve_subject_mask` + `aggregate`; trả `RegionStats`
   (giữ `RegionMetrics` cũ trong output để UI/API không vỡ). Luôn đo `full` và `background`
   (phần bù hợp các subject). `cache_key` = hash ảnh gốc của lượt chạy, dùng lại ở vòng 2–3.
3. `executor.resolve_region_mask` dùng chung resolver (giữ đường face/full/quadrant cũ khi không có bbox).
4. `DiagnosisReport.measurement`, `.region_stats` được điền; ẩn `maps` khỏi JSON/HistoryItem.
5. Nạp sẵn model phân vùng lúc khởi động API/Gradio khi `capabilities()["semantic_models"]`.

## P5 — bằng chứng cho VLM
6. `src/agent/evidence.py`: `select_evidence(image, measurement, region_masks) -> List[EvidenceCrop]`
   (patch phẳng nhất, nhiễu nhất, cạnh mạnh trên chủ thể và nền, mặt, vùng clip lớn nhất; 512 px,
   100 %, nhãn A/B/…, mô tả số đo) + overlay clip/heatmap nhiễu (chỉ để VLM xem — ADR-001 ok).
7. Prompt perceive v2: tóm tắt bằng chứng có cấu trúc thay dump JSON; prior EXIF (`metadata_priors`)
   đưa vào như gợi ý; VLM trả **giả thuyết** có tham chiếu crop; khuyến khích "không có lỗi".
8. Self-consistency N = 3 (temperature 0.7), giữ defect ≥ 2/3, đồng thuận → confidence; scene và
   preserve theo đa số. Ngân sách: ảnh toàn khung + ≤ 6 crop + 2 overlay.
9. Benchmark: điền config `vlm_only` mới và `measure_only` trong `scripts/benchmark_analysis.py`;
   chạy `--set real` (giới hạn số lần gọi, ghi lại) và so với baseline.
10. Đo và ghi vào status: có cần giữ ảnh toàn khung full-res khi đã có crop không (câu hỏi mở §7).

## Test (offline, mock Gemini theo `_route_gemini` trong `tests/unit/test_vlm_mock.py`)
Crop đúng vị trí trên ảnh synthetic; prompt đủ nhãn crop; gộp self-consistency với 3 phản hồi mock
khác nhau; resolver được gọi với bbox từ perceive; cache trúng ở vòng 2; pipeline rule-based (không
key) vẫn chạy.

## Tiêu chí xong
Toàn bộ `tests/unit/` xanh; số benchmark so với baseline ghi trong
`docs/analysis_v2/status/agent.md`, kết thúc bằng `LANE DONE`.
