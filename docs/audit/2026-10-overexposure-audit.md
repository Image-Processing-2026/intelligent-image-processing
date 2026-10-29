# AI Image Doctor — Read-only Audit Report

Scope: why a heavily overexposed portrait was correctly diagnosed ("cháy sáng nặng toàn bộ khung hình") but produced 3 near-identical, barely-changed outputs with red/cyan fringing, ending in `STOP_BEST_EFFORT`.

Method: static read of `src/`, `frontend/`, `tests/`; one offline run and one live run (7 Gemini calls) of the real pipeline on a synthetic overexposed portrait; micro-probes of each tool. No repository file was modified. Scratch scripts and images are outside the repo (session scratchpad). Statements marked **(inference)** are not directly verified.

Environment note: `pytest tests/unit` → 743 passed. `pyiqa` is installed (BRISQUE/NIQE are computed but never used for decisions, see §7).

---

## 1. Architecture map

Entry points
- Gradio UI: `frontend/app.py` (`process_with_variants`, lines 276-300) calls `run_pipeline(...)` directly (not via the API). It passes `max_iterations` and `num_variants`, but never `intent`, so the UI style choice is not wired into this call (inference: style is picked from the variant gallery or from the interactive session path; I did not read that path in full).
- FastAPI: `src/api/main.py`, routes in `src/api/routes.py` (`/diagnose /process /render /refine /sessions /memory`).
- Pipeline: `src/agent/graph.py::run_pipeline` (555-575) builds a LangGraph `StateGraph` over `DoctorState`.

Modules: `analyzer_evaluator` (metrics + NR-IQA), `region_engine` (masks), `processing_engine` (5 classical ops), `agent` (LangGraph, VLM, planner, executor, variants, intent, memory, refine), `api`, `frontend`. `src/extensions` is deferred and not imported.

```
upload ─► analyze ─► perceive ─► clarify ─► diagnose_and_plan ─► process ─► evaluate ─► decide
          (Module 1  (Gemini #1  (only in   (KB retrieval +      (executor:  (NR-IQA vs    │
           metrics)   + region    interactive Gemini #2 →         mask + 5    PREVIOUS      │
                      measuring)  sessions)   planner clamp →     ops on the  image)        │
                                              preserve guard)     CURRENT img)              │
   ▲                                                                                        │
   └──────────────────────────────── RE_PROCESS ◄──────────────────────────────────────────┤
                                                                                            │ SHIP / STOP_BEST_EFFORT
                       should_continue (graph.py:447) ◄─────────────────────────────────────┘
                            │ num_variants>1: Send ×3
                            ▼
          render_variant (balanced / natural / vivid: re-run the recipe on the ORIGINAL, ≤1024 px preview)
                            ▼
          rank_variants (dedup → Gemini critic #3 or Module-1 score) ─► END
```

Graph wiring: `graph.py:520-552`. Key property: `process_node` runs on `state["current_image"]`, the output of the previous iteration (`graph.py:185-192`), so iterations are cumulative.

---

## 2. AI / brain layer

### 2.1 Models and settings

| Role | Provider / model | Settings |
|---|---|---|
| Diagnosis (Perceive) | Google Gemini via `google-genai`; `DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"` (`vlm_diagnostician.py:41`), override with `GEMINI_MODEL` (not set in `.env`) | structured JSON output only |
| Planning | same model, same function | same |
| Variant critic | same model | same |
| Feedback parsing (chat/refine) | same model, only if the Vietnamese rules fail (`refine.py:172`) | same |
| Intent "chat" | no LLM: deterministic questions (`intent.py`, `build_questions`) | n/a |
| Evaluation of each loop step | no LLM: heuristic score (`no_reference_eval.py`) | n/a |

The single call site:

```python
# src/agent/vlm_diagnostician.py:385-397
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    contents=contents,
    config=genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_json_schema=schema,
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
    ),
)
return _parse_json_text(response.text)
```

No `temperature`, `thinking_config`, `max_output_tokens`, `top_p` or seed is set anywhere, so provider defaults apply. A "flash-lite" model with default sampling serves every role. There is no retry.

### 2.2 System prompts and templates (verbatim)

**PLAN `SYSTEM_PROMPT`** — `vlm_diagnostician.py:45-100`

```text
Bạn là "AI Image Doctor" - một chuyên gia chẩn đoán và xử lý ảnh theo phương pháp kinh điển (Classical Image Processing).
Nhiệm vụ của bạn:
1. Nhìn ảnh và đọc các chỉ số kỹ thuật đo lường (độ sáng, tương phản, nhiễu, độ mờ).
2. Chẩn đoán vấn đề ở từng vùng cụ thể (ví dụ: bầu trời bị chói, khuôn mặt bị tối, nền bị nhiễu).
3. Đề xuất kế hoạch điều trị chỉ sử dụng các công cụ trong Toolbox hợp lệ:
   - denoise (parameters: method in ['gaussian', 'median', 'bilateral', 'nlm'], strength in [0.1, 2.0])
   - gamma_correct (parameters: gamma in [0.5, 2.5]. LƯU Ý: gamma > 1.0 để làm sáng ảnh, gamma < 1.0 để làm tối ảnh)
   - clahe (parameters: clip_limit in [1.0, 4.0])
   - sharpen (parameters: method in ['unsharp_mask', 'laplacian'], amount in [0.2, 2.0])
   - color_correct (parameters: saturation_scale in [0.5, 1.5], temperature_shift in [-1.0, 1.0])
4. Tuân thủ thứ tự ưu tiên: Luôn khử nhiễu TRƯỚC KHI làm nét (denoise before sharpen).
5. Chọn vùng xử lý cho mỗi action bằng "region_type" (có thể bỏ trống để hệ thống tự suy ra từ target_prompt):
   - "full": toàn bộ ảnh (target_prompt: "full").
   - "face": khuôn mặt người (target_prompt: "face"). Tùy chọn face_mode in ['bbox', 'oval'] (mặc định 'bbox').
   - "spatial": một phần cố định của khung hình, kèm quadrant in ['top', 'bottom', 'left', 'right', 'center'].
   - "semantic": một đối tượng cụ thể, target_prompt là danh từ ngắn bằng tiếng Anh (ví dụ: "sky", "person", "dog").
     Tùy chọn instance_selection in ['all', 'largest'] (mặc định 'all').
   - Tùy chọn feather_radius in [5, 50] (độ mềm viền mask, mặc định 15).
   - KHÔNG trả về bbox, binary_mask hay tọa độ pixel.
6. Từ vòng 2 trở đi bạn nhận HAI ảnh: ẢNH GỐC (trước mọi xử lý) và ẢNH HIỆN TẠI (cần chẩn đoán).
   So sánh hai ảnh để thấy các vòng trước đã thay đổi gì; chỉ lập kế hoạch cho ẢNH HIỆN TẠI.
   Nếu ẢNH HIỆN TẠI đã đạt chất lượng tốt, trả về "actions": [] (không xử lý thêm).
7. Nếu có CHẨN ĐOÁN (giai đoạn 1): chỉ lập kế hoạch cho các defect có severity >= 1,
   dùng region của defect làm target_prompt và type của defect làm detected_issue;
   cường độ tham số tỉ lệ với severity (1 nhẹ tay, 3 mạnh tay).
   Dùng region_metrics (số đo thật theo vùng) để chọn tham số cho từng vùng.
   TUYỆT ĐỐI giữ nguyên các đặc điểm trong "preserve"; thao tác vi phạm sẽ bị hệ thống loại bỏ.
8. Nếu có TRI THỨC CHUYÊN MÔN: tham số trong playbook đã được hiệu chỉnh bằng đo đạc trên chính
   toolbox này và đã chọn sẵn theo mức độ của ca hiện tại. Áp dụng ĐỦ các bước của công thức
   phù hợp (ví dụ khử nhiễu VÀ nâng sáng) với đúng giá trị đã chọn; chỉ lệch khi thấy lý do cụ
   thể trên ảnh và phải nêu lý do đó. Tuân thủ các mục "Tránh", và nêu id playbook đã dựa vào
   trong reasoning (ví dụ: "[dark-region]").
9. Nếu có Ý ĐỊNH NGƯỜI DÙNG: đó là ưu tiên cao nhất. Không làm mất điều người dùng muốn giữ,
   phải xử lý điều người dùng muốn sửa, và chọn cường độ hợp với phong cách họ mong muốn.
10. Nếu có KINH NGHIỆM TỪ CA TƯƠNG TỰ: học từ phản hồi cũ của chính người dùng này. Ví dụ ca
   giống trước đây bị chê "tối quá" thì lần này nâng sáng mạnh tay hơn. Không chép nguyên
   phác đồ cũ: ảnh hiện tại vẫn là căn cứ chính.

Viết "reasoning" bằng tiếng Việt có dấu. Trả về kết quả dưới định dạng JSON thuần túy theo cấu trúc:
{
  "iteration": 1,
  "reasoning": "Giải thích lý do lựa chọn thuật toán và tham số...",
  "actions": [
    {
      "region_id": "sky",
      "target_prompt": "sky",
      "region_type": "semantic",
      "detected_issue": "overexposed",
      "operation": "gamma_correct",
      "parameters": {"gamma": 0.8},
      "order": 1
    }
  ]
}
```

Item 8 tells the model to apply the playbook values "exactly" and deviate only for a concrete reason. The JSON example hard-codes `gamma: 0.8` for `overexposed`, the same value the playbook gives. Both anchor the model on 0.8.

**PERCEIVE `PERCEIVE_PROMPT`** — `perception.py:71-91`

```text
Bạn là "AI Image Doctor" - GIAI ĐOẠN 1: QUAN SÁT & CHẨN ĐOÁN (chưa lập kế hoạch xử lý).
Nhìn ảnh, đọc chỉ số kỹ thuật toàn cục và trả về bản chẩn đoán JSON gồm:
1. scene_type: loại cảnh.
2. lighting: mô tả ngắn điều kiện ánh sáng (ví dụ: "ngược sáng", "nắng chiều", "đèn trong nhà").
3. subjects: tối đa 4 vùng quan trọng cần đo riêng, là danh từ ngắn bằng tiếng Anh
   ("face" cho khuôn mặt người, "sky", "person", "food", "dog"...). Không dùng "full".
4. defects: các lỗi kỹ thuật THẬT SỰ cần sửa. Mỗi lỗi gồm type, region ("full" hoặc một
   subject), severity (0 không đáng kể, 1 nhẹ, 2 rõ, 3 nặng) và evidence (bằng chứng ngắn).
   Dùng "backlit_subject" khi chủ thể tối vì nguồn sáng phía sau.
5. preserve: đặc điểm thẩm mỹ CÓ CHỦ Ý cần giữ nguyên (aspect, region, reason). Ví dụ:
   ánh hoàng hôn/đèn vàng ấm áp → warm_tone; ảnh tối tâm trạng → low_key;
   bóng đen nghệ thuật → silhouette; hạt film → film_grain; xóa phông mềm mại → soft_focus;
   tông màu nhạt kiểu film → muted_colors.
   Đặc điểm đã nằm trong preserve thì KHÔNG được liệt kê thành defect.
6. summary: tóm tắt chẩn đoán bằng tiếng Việt, 1–3 câu.
Lưu ý: chỉ số toàn cục là trung bình cả ảnh. Một vùng nhỏ (ví dụ khuôn mặt ngược sáng)
có thể tối dù brightness_level toàn cục là "normal" → hãy nhìn ảnh và liệt kê vùng đó.
Từ vòng 2 trở đi bạn nhận ẢNH GỐC và ẢNH HIỆN TẠI: chỉ chẩn đoán lỗi CÒN LẠI trên ẢNH HIỆN TẠI.
Viết lighting, evidence, reason và summary bằng tiếng Việt có dấu.
```

Schema: `perception.py:93-127` (`scene_type`, `lighting`, `subjects ≤4`, `defects[{type, region, severity 0-3, evidence}]`, `preserve[{aspect, region, reason}]`, `summary`).

**CRITIC prompt** — `variants.py:253-261`

```text
Bạn là "AI Image Doctor" - GIÁM KHẢO. Bạn nhận ẢNH GỐC và vài PHIÊN BẢN đã được xử lý bằng
thuật toán kinh điển (không có AI tạo ảnh). Hãy xếp hạng các phiên bản từ tốt nhất đến kém nhất:
1. Đã sửa được các lỗi trong chẩn đoán chưa (sáng, tương phản, nhiễu, màu…)?
2. Có tự nhiên không: không cháy sáng, không quầng, không bệt da, không nhiễu bị khuếch đại?
3. Có giữ được các đặc điểm cần giữ (preserve) và không khí của ảnh gốc không?
Trả về JSON: "ranking" (mọi id, tốt nhất trước), "recommended" (id nên chọn), và "notes"
(mỗi phiên bản một nhận xét ngắn bằng tiếng Việt có dấu, nêu ưu và nhược điểm thấy được trên ảnh).
```

**FEEDBACK prompt** — `refine.py:131-136`

```text
Bạn chuyển góp ý của người dùng về một bức ảnh đã chỉnh thành các điều chỉnh có cấu trúc.
Mỗi điều chỉnh gồm kind, region ("full" hoặc tên vùng ngắn bằng tiếng Anh như "face", "sky")
và strength (1 nhẹ, 2 rõ). Chỉ dùng các kind cho phép. Nếu góp ý không yêu cầu thay đổi gì
về ảnh, trả về danh sách rỗng.
```

**Dynamic blocks in the Plan user message** (`vlm_diagnostician.py:335-358`, `_build_history_feedback` 212-280, `intent.py:274-289`, `memory.py:262-281`):
"Chỉ số kỹ thuật hiện tại: {metrics JSON}", "CHẨN ĐOÁN (giai đoạn 1, kèm region_metrics đo thực tế theo vùng)", the KB block "TRI THỨC CHUYÊN MÔN (…)", optional "Ý ĐỊNH NGƯỜI DÙNG (ưu tiên cao nhất, ghi đè chẩn đoán)", optional "KINH NGHIỆM TỪ CA TƯƠNG TỰ", then "Vòng lặp: N". From iteration 2 it also adds "--- PHẢN HỒI TỪ CÁC VÒNG TRƯỚC ---" (last 2 iterations: actions, deltas of brightness/contrast/noise/sharpness, decision, reasoning), ending with "KHÔNG lặp lại cùng thao tác với cùng tham số nếu chỉ số không cải thiện." The real iteration-1 text from the live run is summarised in §8.2.

### 2.3 What is sent to the model

```python
# vlm_diagnostician.py:307-332
def to_vlm_image(image):   # longest side <= VLM_MAX_SIDE = 1024 (line 43), cv2.INTER_AREA
def image_parts(image, iteration, original_image):
    if iteration > 1 and original_image is not None:
        parts += ["ẢNH GỐC (trước mọi xử lý):", to_vlm_image(original_image), "ẢNH HIỆN TẠI (cần chẩn đoán):"]
    parts.append(to_vlm_image(image))
```

- Iteration 1: 1 image. Iteration ≥ 2: 2 images (original + current). Critic: the original plus every variant (4 images in the live run). All are plain downscaled full frames. No crops, overlays, masks or histogram images.
- Observed sizes: 512×512 (the test image is 512 px; real photos are capped at 1024 px on the long side).
- Text: global `TechnicalMetrics` JSON (§3), the Stage-1 diagnosis with measured `region_metrics`, the KB block, and history. The Perceive call gets only metrics, history and images (642 characters of text on iteration 1).
- Calls per iteration: Perceive 1 + Plan 1. Plan is skipped, with no Gemini call, when no defect has severity ≥ 1 (`graph.py:124-130`). One critic call when ≥2 distinct variants exist.

### 2.4 Output schema, parsing, failure handling

- `PLAN_RESPONSE_SCHEMA` is generated from `PARAMETER_BOUNDS` (`vlm_diagnostician.py:103-164`): one `anyOf` variant per operation with min/max-bounded numbers.
- `_plan_from_vlm_json` (167-209): per-action `clamp_region_fields`, keeps only the parameters that belong to the action's operation, drops invalid actions with a warning; if all are invalid → `ValueError`.
- Any exception (network, parse, validation) → rule-based plan from the KB `auto_apply` cards, `source="vlm_fallback"` (443-453). No key → `source="rule_based"`. A valid `actions: []` is a real empty plan and becomes `SHIP` (`graph.py:386-387`).
- The planner then clamps parameters, reorders and runs the preserve guard (§5).
- Perceive failures fall back to `diagnose_from_metrics` (`perception.py:404-406`).

---

## 3. Analyzer (`src/analyzer_evaluator/analyzer.py`)

All Module 1 metrics are **global**. Per-region brightness is added later, for the "subjects" only (`perception.compute_region_metrics`).

Thresholds (`analyzer.py:24-49`):

```python
_BRIGHTNESS_LOW = 70.0;  _BRIGHTNESS_HIGH = 185.0
_SHADOW_CLIP_THRESH = 5; _HIGHLIGHT_CLIP_THRESH = 250
_SHADOW_CLIP_RATIO_THRESH = 0.25; _HIGHLIGHT_CLIP_RATIO_THRESH = 0.20
_CONTRAST_LOW = 40.0; _CONTRAST_HIGH = 80.0
_NOISE_CLEAN = 3.0; _NOISE_LOW = 8.0; _NOISE_MEDIUM = 15.0
_BLUR_SEVERE = 100.0; _BLUR_MILD = 300.0
_LAB_GREEN_A_THRESH = 110.0; _LAB_WARM_B_THRESH = 140.0; _LAB_COOL_B_THRESH = 118.0
_RGB_CHANNEL_DIFF = 20.0; _RGB_GREEN_DIFF = 15.0
```

| Metric | Formula / code | Scope |
|---|---|---|
| `brightness_mean` | mean of the **grayscale** image (cv2 RGB2GRAY), from a 256-bin histogram (`_compute_gray_hist_stats`, 141-202) | global |
| `brightness_level` | `underexposed` if mean<70 or shadow_ratio>0.25; `overexposed` if mean>185 **or** highlight_ratio>0.20 (`analyzer.py:395-400`) | global |
| `highlight_clip_ratio` | `hist[250:].sum()/N` on **luma** (≥250, not =255) (line 183) | global |
| `shadow_clip_ratio` | `hist[:6].sum()/N` | global |
| `contrast_std` | std of luma; low <40, high >80 | global |
| `noise_variance` | despite the name, a **sigma**: Immerkær `π/2 · 1/(6(w−2)(h−2)) · Σ|I*K|`, K=[[1,−2,1],[−2,4,−2],[1,−2,1]] (113-138) | global |
| `sharpness_laplacian_var` | `cv2.Laplacian(gray, CV_32F).var()`; <100 severe, <300 mild | global |
| Tenengrad | computed, log-only | global |
| `color_cast` | mean LAB a*/b* plus mean RGB: greenish / warm (b*>140 or R−B>20) / cool / none (216-264) | global |
| `histogram_stats` | gray min/max/skew, dynamic_range (P99−P1), per-channel RGB mean/std, dominant HSV hue | global |

Gaps that matter here: no per-channel clipping ratio, no `==255` ratio, no clipped-area map, and no estimate of how much highlight information is recoverable. The live input had 33.7% of RGB samples at exactly 255, but the analyzer reported only `highlight_clip_ratio = 0.2993` (luma ≥ 250).

---

## 4. Region engine

- Faces: MediaPipe (`face_detector.py`; models in `models/mediapipe/`), bbox expanded by `expand_ratio` (default 0.15). `face_mode` is bbox by default (oval needs a Face Landmarker model).
- Semantic regions ("sky", "person", "suit"…): optional GroundingDINO + MobileSAM (`segmentation_backend.py`). **The model directory is missing in this checkout** (`models/` has only `mediapipe/`). The live run logged: `GroundingDINO model directory was not found at …/models/segmentation/grounding-dino-tiny`. The executor then (decision D1) either falls back to a keyword quadrant ("sky"→`top`, "ground"→`bottom`, "center"; `executor.py:30-36,155-176`; the live run mapped "background"→`bottom`, "sky"→`top`) or **skips the action** ("person", "suit", "helmet" were skipped).
- Spatial regions (top/bottom/left/right/center): rectangular quadrant masks.
- Masks are applied: `apply_region_op` runs the op on the whole frame, then `blend_regions(image, processed, mask)` = `mask*processed + (1−mask)*original` (`base.py:13-36`, `mask_utils.py:96-141`).
- Feathering: `create_soft_mask` Gaussian blur, kernel size = `feather_radius` (made odd), `sigma = kernel/3`, reflect border (`mask_utils.py:60-77`). The planner bounds `feather_radius` to [5, 50], default 15 (`planner.py:59`). On a ≤1024 px image, 15 px is a tight feather (inference: this can leave a visible edge around a local face adjustment).
- `executor._validate_mask` (39-62) rejects non-float32, all-zero, NaN and out-of-range masks. An invalid mask means the action is skipped (not run on the full image) (`executor.py:186-193`).
- `region_type == "full"` uses `mask=None`, so the op runs globally (`executor.py:134-135,198-199`). In the live run both "full" gamma actions ran this way.

---

## 5. Toolbox / processing engine

Operation names → functions: `denoise→apply_denoise`, `gamma_correct→apply_gamma`, `clahe→apply_clahe`, `sharpen→apply_sharpen`, `color_correct→apply_color_balance`. All accept a soft mask through `apply_region_op`. All work on uint8 RGB. The planner enforces the order denoise 10 → gamma 20 → clahe 30 → sharpen 40 → color 50 (`planner.py:24-30,253-265`), overwriting any model-supplied order.

### 5.1 Parameter table (`planner.py:33-52`)

| Tool | Parameter | Min | Max | Default | Notes |
|---|---|---|---|---|---|
| denoise | strength | 0.1 | 2.0 | 1.0 | |
| denoise | method | gaussian / median / bilateral / nlm | | bilateral | |
| gamma_correct | gamma | **0.5** | 2.5 | 1.2 | `<1` darkens |
| clahe | clip_limit | 1.0 | 4.0 | 2.0 | tile grid fixed at 8×8, not exposed to the plan |
| sharpen | amount | 0.2 | 2.0 | 1.0 | |
| sharpen | method | unsharp_mask / laplacian | | unsharp_mask | |
| color_correct | saturation_scale | 0.5 | 1.5 | 1.0 | |
| color_correct | temperature_shift | −1.0 | 1.0 | 0.0 | |

Clamping: `clamp_parameters` (180-227) clamps with a warning and fills defaults. Unknown operations are dropped (`sanitize_actions`, 230-250).

### 5.2 Implementations and color space

```python
# exposure_contrast.py:14-21 — gamma: per-channel LUT directly on RGB (no luminance / linear-light handling)
inv_gamma = 1.0 / max(0.01, gamma)
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
return cv2.LUT(image, table)
```
```python
# exposure_contrast.py:44-49 — CLAHE on the LAB L channel only (colour-safe), 8x8 tiles
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB); ... l_enhanced = clahe_op.apply(l_ch)
```
```python
# sharpen.py:20-24 — unsharp mask applied independently to each RGB channel, sigma fixed at 2.0
blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
# sharpen.py:29-34 — laplacian: 3x3 kernel with centre 1+4*amount, also per RGB channel
```
```python
# color.py:21-33 — saturation in HSV (uint8 round trip); temperature = raw R/B offsets
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
img_f[:, :, 0] = np.clip(img_f[:, :, 0] + temperature_shift * 20.0, 0, 255)   # R +
img_f[:, :, 2] = np.clip(img_f[:, :, 2] - temperature_shift * 20.0, 0, 255)   # B -  (G untouched)
```
`denoise.py:14-44`: gaussian / median / bilateral (d = max(5, 5·s), σ = 50·s) / NLM (h = 10·s). All run per RGB channel (NLM uses `fastNlMeansDenoisingColored`).

### 5.3 Capability checklist

| Capability | Exists? | Evidence |
|---|---|---|
| Tone curve | **No** | only a fixed power function |
| Levels / black-white point | **No** | none |
| Gamma | Yes (global or masked), **range 0.5–2.5** | above |
| Highlight compression / recovery | **No** | none; gamma is the identity at 255 |
| Exposure adjustment (EV / linear gain) | **No** | none; only gamma |
| Local (masked) exposure | Indirect: `gamma_correct` with a region mask | semantic masks need the missing DINO model → quadrant guess or skipped |
| Saturation / vibrance | Saturation only (HSV scale); no vibrance | `color.py` |
| White balance | Crude temperature (R±, B∓); no gray-world/auto WB, no tint axis | `color.py:27-33` |
| Local contrast | CLAHE (L channel) | |
| Luminance-only sharpening | **No**; sharpening is per RGB channel | `sharpen.py` |

For an overexposed image the only brightness-lowering tool is a power function with exponent in [1, 2], and it cannot move a pixel that is already 255 (§9).

---

## 6. Style variants (Tự nhiên / Cân bằng / Đậm nét)

They are **post-hoc parameter multipliers applied to one shared recipe**: not separate plans, and not new VLM calls.

```python
# variants.py:59-90
"balanced": StyleProfile(...)                                   # recipe unchanged
"natural":  intensity={"gamma_correct":0.7,"clahe":0.5,"denoise":0.8,"color_correct":0.6}, drop_operations=("sharpen",)
"vivid":    intensity={"gamma_correct":1.1,"clahe":1.4,"sharpen":1.2,"color_correct":1.3},
            extra_saturation=1.12, extra_clahe=1.5
```
```python
# variants.py:129-148  (k = intensity)
gamma:      1.0 + k*(gamma-1.0)           clip_limit: 1.0 + k*(clip-1.0)
denoise:    k*strength                    sharpen:    k*amount
color:      sat = 1.0 + k*(sat-1.0),      temp = k*temp
# every result is clamped by clamp_parameters
```
- Recipe = the actions of all iterations that were not rolled back (`treatment_recipe`, 98-105).
- Rendering re-runs the recipe on the **original** downscaled to ≤1024 px (`render_variant`, 201-221), after `apply_preserve_guard`, and scores it with `evaluate_no_reference(image, previous_image=original)`.
- `vivid` adds a full-image CLAHE (1.5) and saturation (1.12) if the recipe lacks them (CLAHE is skipped for noisy images).
- `_deduplicate` (224-233) drops byte-identical variants. **If the recipe is empty (e.g. iteration 1 rolled back, or no actions), all three renders equal the original and only one survives** (from reading the code; not triggered in my runs).

Numbers from the live run (recipe = `gamma 0.8` full, then `gamma 0.85` on the face):

| Variant | Actions rendered | Mean \|Δ\| vs input (0-255) | Module-1 score |
|---|---|---|---|
| balanced | gamma 0.8, gamma 0.85 (face) | 11.18 | 89.50 |
| natural | gamma 0.86, gamma 0.895 (face) | 7.60 | 89.08 |
| vivid | gamma 0.78, gamma 0.835 (face), clahe 1.5, sat 1.12 | 26.03 | 87.76 |

For an overexposed image, "Tự nhiên" and "Cân bằng" differ by about 0.06 in gamma, and both push in the same direction. They share the same 255 plateau (§9). Only "Đậm nét" is visibly different, and it gets there through CLAHE and saturation, not exposure.

---

## 7. Evaluator and loop control

### 7.1 Metrics and what they are compared against

Real images (no ground truth) use `evaluate_no_reference` (`graph.py:213-218`):

- `estimated_quality_score` is the only score the loop uses. It is a heuristic composite, **absolute, no reference** (`no_reference_eval.py:71-129`):

```python
_W_NOISE=0.35; _W_BLUR=0.30; _W_EXPOSURE=0.20; _W_CONTRAST=0.15
noise_penalty    = min(noise_sigma/30, 1)*100
blur_penalty     = (1 - min(laplacian_var/300, 1))*100
exposure_penalty = min(|brightness_mean-128|/128, 1)*100       # grayscale mean only
contrast_penalty = min(distance_to_[45,75]/45, 1)*100
score = 100 - (0.35*noise + 0.30*blur + 0.20*exposure + 0.15*contrast)
```
  Clipping is not part of the score. BRISQUE/NIQE from pyiqa are computed and stored, but `estimated_quality_score` is the heuristic only (line 388).
- `quality_improved` and `delta_metrics` compare the **current image with the previous iteration's image** (`previous_image` is the image before this iteration's process step, or the original on iteration 1; `graph.py:199-203`). After iteration 1 the original is never the reference.
- PSNR/SSIM/MSE (`evaluate_reference`) run **only** when `is_synthetic` and a ground truth exists (`graph.py:206-212`). The UI sets `is_synthetic = ground_truth is not None` (`app.py:92`). They compare against the **ground truth, not the input**, so "PSNR/SSIM against the input penalizes any change" is **not** the failure mode for ordinary uploads. (If a user supplied a "ground truth" equal to the input, any change would be penalized; inference, not exercised.)

`quality_improved` (`no_reference_eval.py:218-298`):

```python
if delta_noise > 10.0: return False
if curr_clip > 0.15 and curr_clip > prev_clip*1.5: return False       # new highlight clipping
denoise_improved    = delta_noise < -1.5 and sharpness_drop_ratio < 0.30
sharpen_improved    = delta_sharpness > 20.0 and delta_noise < 3.0
brightness_improved = delta_brightness > 0.0                           # |prev-128| - |curr-128| > 0
return denoise_improved or sharpen_improved or brightness_improved
```
A step that changes nothing (Δbrightness = 0) is `quality_improved = False` (observed: live iteration 3).

### 7.2 Decision logic (`graph.py:328-444`) and constants (`graph.py:49-58`)

`PSNR_THRESHOLD=28, SSIM_THRESHOLD=0.88, PSNR_DEGRADATION=1.5, REAL_TARGET_SCORE=85.0, REAL_MIN_GAIN=1.0, VERIFY_SEVERITY=2`. `max_iterations` defaults to 3 in code; the UI passes its slider value.

For a real image, `decide_node` checks in this order:

1. `unchanged`: the plan has actions and `np.array_equal(current, previous)` → `STOP_BEST_EFFORT`, no rollback (379-383). Only exact equality counts; a 0.5-gray-level change passes.
2. `degraded`: has actions, not unchanged, and `_is_degraded_real` → `STOP_BEST_EFFORT` **with rollback** to the previous image (376-377, 413-419):
```python
# graph.py:242-253
if eval_result.get("quality_improved") is False: return True
if history: return current_quality < prev_quality     # ANY drop of the heuristic score, even 0.01
```
3. Empty plan → `SHIP` (386-387).
4. Otherwise `_decide_real` (272-296): degraded → `STOP_BEST_EFFORT`; **score ≥ 85 → `SHIP`**; score − prev_score < 1.0 → `STOP_BEST_EFFORT` (reached only when score < 85); else `RE_PROCESS`.
5. Real image + `SHIP` + `iteration < max_iters` + `_needs_verification` (a severity ≥2 defect on a touched region or on full) → downgraded to `RE_PROCESS` (399-405).
6. `RE_PROCESS` and `iteration >= max_iters` → `STOP_BEST_EFFORT` (407-408).

So `STOP_BEST_EFFORT` comes from one of four paths: **(A)** degraded: `quality_improved False` or any score drop, with rollback; **(B)** pixels exactly unchanged; **(C)** score < 85 and gain < 1.0; **(D)** the iteration cap reached while the decision was still `RE_PROCESS`. Rollback happens only in (A).

Best-so-far: there is no best-of-N selection across iterations. The final image is the last non-rolled-back one; a rollback restores only the single previous image (`graph.py:411-419`). Variants are ranked by the Module-1 score with `SCORE_TIE_TOLERANCE = 0.5` (`variants.py:236-250`) or by the Gemini critic.

Visual judge: Gemini acts as a visual judge only in the variant critic (§2.2), after the loop. It never accepts or rejects a loop step. Each iteration's Perceive step does re-inspect the image, and it can end the loop by returning no defects, which makes the plan empty and the decision `SHIP`.

---

## 8. Live trace

Run and logged. Input: `skimage.data.astronaut()` (512×512 portrait) with `img + 80` clipped. The other suggested transform, `1-(1-img)**3`, was also run offline. Model `gemini-3.5-flash-lite`, 7 Gemini calls in total, `max_iterations=3`, `num_variants=3`, no intent, case memory off.

### 8.1 Input histogram

| Input | R ≥250 | R =255 | G ≥250 | G =255 | B ≥250 | B =255 | luma ≥250 | luma =255 |
|---|---|---|---|---|---|---|---|---|
| `+80` (heavy) | 52.23% | 50.11% | 29.54% | 27.05% | 26.43% | 23.80% | 29.93% | 24.71% |
| `1-(1-x)^3` | 42.60% | 0.15% | 20.25% | 0.11% | 17.36% | 0.10% | 21.09% | 0.11% |

The cubic transform mostly squeezes values into 250–254; the `+80` shift produces large truly clipped areas, closer to "cháy nặng". The channels clip at different rates (R 50%, G 27%, B 24%), so large areas are clipped in R but not in G/B (inference: this is what makes hue shift and fringing worse after per-channel operations).

Input metrics (`+80`): brightness_mean 185.49 (overexposed), contrast_std 65.19, noise 2.43, sharpness 655, color_cast warm, highlight_clip_ratio 0.2993.

### 8.2 Live per-iteration log

| Iter | Diagnosis | Plan | Change vs prev (mean \|Δ\|, mean luma) | Score (prev → now), `quality_improved`, Δbrightness | Decision |
|---|---|---|---|---|---|
| 1 | VLM: overexposed@full sev 2, color_cast_warm; preserve `warm_tone@full`. **Measured** (added by code): overexposed@face sev 2 (face mean 205, 38% clipped) | VLM; KB ids `global-overexposed`, `scene-portrait`…; `gamma_correct full gamma=0.8` | 10.67; 184.5 → 173.8 | input 88.18 → 89.46; True; +10.59 | `SHIP` by score, downgraded to **RE_PROCESS** (face sev-2 defect needs verification) |
| 2 | VLM: overexposed@full sev 2; measured overexposed@face sev 2 (mean 196.7, 37% clipped) | VLM; `gamma_correct face gamma=0.85` (face bbox, ~7% of frame) | **0.51**; 173.8 → 173.3 | 89.46 → 89.50; True; +0.49 | RE_PROCESS (same downgrade) |
| 3 | VLM: overexposed@full sev 2; measured overexposed@face sev 2 and overexposed@sky sev 2 (heuristic `top` quadrant, 48% clipped) | VLM returns `"actions": []` ("đã đạt chất lượng cân bằng, không còn lỗi nghiêm trọng") | **0.00** | 89.50 → 89.50; **False**; 0.0 | **SHIP** (empty plan) |

Final: `SHIP` after 3 iterations, final mean luma ≈173, highlight_clip_ratio 0.29 (input 0.2993). My run ended in `SHIP`, not `STOP_BEST_EFFORT` (see §8.3 for the offline runs that did).

Raw Gemini output, iteration-1 Plan:

```json
{"iteration": 1,
 "reasoning": "Dựa trên tri thức chuyên môn [global-overexposed] và [scene-portrait], ảnh chân dung đang bị dư sáng toàn cục (brightness_mean = 185.49, highlight_clip_ratio cao). Áp dụng gamma_correct toàn ảnh với gamma=0.8 để hạ độ sáng tổng thể, ...",
 "actions": [{"region_id": "full_image", "target_prompt": "full", "region_type": "full", "feather_radius": 15,
              "detected_issue": "overexposed", "operation": "gamma_correct", "parameters": {"gamma": 0.8}}]}
```

That Plan prompt included the KB card: "Công thức: gamma_correct trên toàn ảnh (mức 2): gamma=0.8 [mức 2→0.8, mức 3→0.7]", "Tránh: gamma < 0.6: vùng tối bị dìm, ảnh bết và mất chi tiết bóng." and "Tránh: Vùng đã cháy trắng… không khôi phục được chi tiết". The model copied the card value, as `SYSTEM_PROMPT` item 8 instructs.

Critic (call 7) ranked balanced > natural > vivid, noting that balanced "xử lý khéo léo vùng dư sáng, khôi phục chi tiết trang phục và nền". (inference: this does not match the images, where the white background is unchanged.)

### 8.3 Offline (rule-based, no key) traces

| Input | Iter | Plan | Score | Decision |
|---|---|---|---|---|
| `1-(1-x)^3` | 1 | gamma 0.8 full (mean \|Δ\| 6.96) | 84.44 (improved True) | RE_PROCESS (score < 85) |
| | 2 | gamma 0.8 on face (mean \|Δ\| 0.57) | 84.49, gain 0.05 < 1.0 | **STOP_BEST_EFFORT** (path C) |
| `+80` | 1 | gamma 0.8 full | 89.46 | RE_PROCESS (verification downgrade) |
| | 2 | gamma 0.8 full again (mean \|Δ\| 10.8) | 88.80 < 89.46 → degraded | **STOP_BEST_EFFORT + rollback** (path A) |

In the `+80` offline case iteration 2 had `quality_improved True` (Δbrightness +10.78) and was rolled back only because the absolute heuristic score fell by 0.66.

### 8.4 Toolbox probe on the `+80` image (measured)

| Operation | luma mean | % samples ≥250 | % samples =255 | Score | quality_improved |
|---|---|---|---|---|---|
| input | 185.5 | 36.1 | **33.7** | **88.18** | n/a |
| gamma 0.8 | 174.9 | 35.6 | 33.7 | 89.46 | True |
| gamma 0.7 | 168.6 | 35.1 | 33.7 | 89.28 | True |
| gamma 0.5 (floor) | 153.3 | 34.7 | **33.7** | **87.87** | True |
| gamma 0.5 ×3 cumulative | 108.9 | 33.7 | 33.7 | 81.89 | True |

Chroma side effects: mean and p99 change of Lab (a*, b*) caused by one operation on this image:

| Operation | Mean | p99 |
|---|---|---|
| gamma 0.8 | 2.55 | 10.63 |
| unsharp 1.0 | 2.24 | 15.56 |
| unsharp 2.0 | 4.22 | 28.65 |
| laplacian 1.0 | 5.08 | 31.83 |
| CLAHE 2.0 (Lab L only) | 0.11 | 3.16 |
| saturation 1.12 | 2.04 | 8.60 |
| temperature ±0.5 | 6.0–6.2 | 8.54 |

Gamma on a partly clipped pixel: `(255,200,170) → (255,188,153)` at γ 0.8 and `→ (255,156,113)` at γ 0.5. R stays clipped while G and B fall, so the pixel turns more saturated orange/red.

---

## 9. Red flags (ranked by likelihood)

### (a) Outputs barely change

1. **The toolbox cannot fix clipped highlights, and the only exposure tool is a bounded power function.** Gamma is the identity at 255, so 33.7% of samples stayed at 255 even at the minimum gamma 0.5 (§8.4). There is no gain, levels, tone curve or highlight compression (§5.3). A near-white background cannot be "fixed" with this toolbox, so every output keeps the white plateau. (confirmed)
2. **The playbook picks a mild gamma and the prompt makes the model follow it.** Card `global-overexposed`: severity 2 → 0.8, severity 3 → 0.7 (`cards/exposure.yaml:27-46`); `SYSTEM_PROMPT` item 8 says to apply the chosen values. Live: the model returned exactly 0.8 and the mean moved 185 → 174. The rule engine assigns severity 3 only if brightness > 215 (`engine.py:53`), and in the live run the VLM rated the problem severity 2. The card's own "Tránh: gamma < 0.6" also discourages going lower. (confirmed)
3. **The evaluator rates the input as good already, and does not reward real exposure fixes.** The input scores 88.18, above `REAL_TARGET_SCORE = 85` (§7.2 step 4). The exposure term is `0.20·|mean−128|/128`, so 10 gray levels is worth about 1.6 points; clipping is not scored; contrast and sharpness penalties are already 0. A stronger correction (gamma 0.5) scored lower (87.87) than the input (88.18), and three cumulative γ 0.5 scored 81.89 (§8.4). (inference: the drop comes from the contrast term as `contrast_std` leaves the 45–75 band; not verified per term.) Differences between candidate treatments (≤ 1.3 points) are comparable to `REAL_MIN_GAIN = 1.0` and `SCORE_TIE_TOLERANCE = 0.5`. (numbers confirmed, cause inferred)
4. **Follow-up iterations do almost nothing.** Iteration 2 touched only the face bbox (mean |Δ| 0.51) and iteration 3 changed nothing (0.00). `perception._measured_defects` adds an `overexposed` defect of fixed severity 2 whenever a region's highlight_clip_ratio ≥ 0.20 (`perception.py:261-263`). It cannot be cured because the pixels are clipped, so the loop keeps verifying until the cap or until the VLM gives up. (confirmed)
5. **Semantic region backend is missing in this checkout.** "person", "suit", "helmet" are skipped and "sky"/"background" become quadrant guesses (`executor.py:155-176`); errors are swallowed per action (`executor.py:90-98`). (confirmed here; the user's machine may differ)
6. **The VLM sees only downscaled global images and its own previous output.** It has no clipping map and no per-channel clipping stats. A flash-lite model with default sampling saw a 10-level mean change and answered "đã đạt chất lượng cân bằng" in iteration 3 (§8.2). (confirmed in the trace; cause is inference)

### (b) Three variants nearly identical

1. All three are scalar rescalings of the same one-action recipe (§6): γ 0.78 / 0.80 / 0.86. Mean |Δ| vs input: 11.2 / 7.6 / 26.0 (live), 7.5 / 5.3 / 19.4 (offline cubic). In exposure terms the gap between "Tự nhiên" and "Cân bằng" is tiny relative to the clipped plateau gamma cannot move. (confirmed)
2. "Đậm nét" differs mostly through added CLAHE and saturation, not exposure; "Tự nhiên" uses gamma k = 0.7, which for an overexposed image means a *weaker* fix. (confirmed)
3. If iteration 1 is rolled back or the plan is empty, the recipe is empty and all three renders equal the original, and `_deduplicate` collapses them (`variants.py:98-105,224-233`). The user saw 3 images, so their recipe was non-empty; the "≈ input" symptom matches a recipe of one weak gamma. (partly inference)
4. The ranking tie-break favours `balanced` (`STYLE_ORDER`), and the critic is the same flash-lite model; in the live run it preferred the weakest-looking "Cân bằng" and credited it with recovering detail that cannot be recovered. (inference)

### (c) Red/cyan color fringing on hair and clothes

Not visibly reproduced in my 512 px crops (the test image has little fine hair detail), so this ranking is based on measured chroma shifts plus code reading:

1. **Per-RGB-channel sharpening.** Unsharp mask and laplacian run on each channel independently (`sharpen.py:20-34`), not on luminance. Lab chroma p99 shifts: 15.6 (amount 1.0), 28.7 (amount 2.0), 31.8 (laplacian 1.0). Where one channel is clipped and another is not (R 50% clipped vs B 24%), per-channel overshoot produces opposite-signed R and B halos, i.e. red/cyan fringes. (shift measured; fringe mechanism is inference)
2. **Per-channel gamma LUT on partly clipped pixels** changes hue because the clipped channel does not move while the others fall (`(255,200,170) → (255,188,153)`). The imbalance grows every iteration and in every variant, since iterations are cumulative. (confirmed numerically)
3. **Temperature shift** adds +20·t to R and −20·t to B, leaves G alone and does not preserve luminance (`color.py:27-33`), so it produces a global red/blue (cyan-leaning) tint. The preserve guard neutralizes it only for `warm_tone`/`cool_tone` (`planner.py:274-277`). (confirmed in code; not triggered in my run)
4. **Chroma subsampling in the source JPEG** plus clipped channels creates coloured edge pixels that sharpen/gamma then amplify. (inference)
5. CLAHE is not a suspect (Lab L only; chroma shift 0.11). `blend_regions` is also unlikely: the fringes are on subject edges rather than on mask borders, and "full" actions use no mask. (inference)

### (d) Hitting max iterations / `STOP_BEST_EFFORT`

The four paths are in §7.2 (A rollback on a score drop or `quality_improved False`; B pixels identical; C score < 85 and gain < 1; D iteration cap). In my traces: C (offline cubic) and A (offline `+80`).
1. The score sits in a narrow ~84–90 band and moves by tenths per step, so "gain < 1.0" (C) or a 0.01 drop (A) triggers almost immediately. (confirmed)
2. Re-applying the same gamma to the already-processed `current_image` compounds, and the heuristic score then *falls* (88.80 < 89.46) and causes a rollback (A). (confirmed offline)
3. Verification downgrades `SHIP` to `RE_PROCESS` for any persistent severity-2 defect (`graph.py:399-405`). The face defect is permanent (clipped), which pushes the loop into iterations 2 and 3 with nothing meaningful left to try. (confirmed)
4. The loop never compares against the original or against a clipping-based target. `previous_image` is the last step only, so a series of tiny steps each look like "no gain". (confirmed)

---

## 10. Misc

Config / env
- `.env.example`: `GEMINI_API_KEY`, `PORT`, `HOST`, `DEBUG`, `MAX_ITERATIONS`. I found no consumer of `MAX_ITERATIONS` in `src/` (inference: unused; the UI slider and API pass `max_iterations`). Region env vars: `REGION_DINO_MODEL_PATH`, `REGION_MOBILE_SAM_CHECKPOINT`. Optional `GEMINI_MODEL`. `VLM_MAX_SIDE` is a constant (1024), not an env var.
- `.env` is saved with a BOM, so `python-dotenv` misses `GEMINI_API_KEY` (documented in `src/api/main.py:17` and `CLAUDE.md`); scripts read it with `utf-8-sig`. If the key is lost the pipeline silently runs rule-based (`source=rule_based`).
- Dependencies (`requirements.txt`): fastapi, uvicorn, pydantic≥2.6, python-dotenv, opencv-python-headless≥4.9, scikit-image, numpy<2, pillow, pyyaml, langgraph≥0.0.30, google-genai≥2.0, mediapipe, gradio. Dev extras: pytest, pytest-asyncio, ruff, pyiqa, scipy, matplotlib; optional `segmentation` extra (torch etc.). CPU-only target (ADR-004).
- Tests: 743 unit tests pass; integration tests exist in `tests/integration` (`test_pipeline_loop.py`, `test_full_pipeline_real.py`, …). Unit tests use the offline rule-based path or mocked Gemini (`tests/unit/test_vlm_mock.py`). I found no test asserting that a heavily overexposed image is visibly corrected, or that the three variants are visually distinct beyond dedup.
- TODO/FIXME: a grep for `TODO|FIXME|XXX|HACK` in `src`, `frontend`, `tests`, `scripts` finds none in code. `CLAUDE.md` lists GroundingDINO/MobileSAM as a TODO, and `src/extensions` (deblur/inpaint/super-res) is deliberately deferred.
- Untracked files in the working tree: `CLAUDE.md`, `generate_visual_comparison.py`, `list_gemini_models.py`, `run_gemini_test.py`, `scripts/download_real_samples.py`, `scripts/prepare_e2e_images.py`, `scripts/run_e2e_full.py`.
- Naming quirk: `noise_variance` holds a standard deviation (Immerkær sigma), not a variance (`analyzer.py:113-138`).
- `data/real/` has underexposed, noisy and warm-cast samples but no overexposed one, so this failure class has no regression fixture.

---

## Summary of likely root causes

1. **Toolbox gap:** no exposure gain, levels, tone curve or highlight compression. The only darkening tool is gamma in [0.5, 1), which is the identity on clipped pixels. 33.7% of samples stayed at 255 even at the gamma floor (measured).
2. **Weak, enforced parameters:** the playbook gives γ 0.8 (severity 2) / 0.7 (severity 3) and the prompt tells the model to use those values. Live: γ 0.8, mean luma 185 → 174.
3. **Evaluator is blind to the defect:** the heuristic score is global, uses the grayscale mean, ignores clipping, and rates the overexposed input 88 (> `REAL_TARGET_SCORE` 85). Treatment differences (~1 point) are at or below the decision thresholds, and a stronger fix scored lower than the input.
4. **Loop stops on noise:** `STOP_BEST_EFFORT` fires on score gain < 1.0, on any score drop (rollback), or on identical pixels, and after the first mild gamma there is nothing meaningful left to try.
5. **Unfixable "measured" face defect** (clipped ≥ 20%, fixed severity 2) forces verification iterations that apply a tiny face-only gamma (mean |Δ| 0.5, then 0.0).
6. **Variants are scalar multipliers of that same weak recipe** (γ 0.78/0.80/0.86 plus CLAHE/saturation), so they converge on the same white plateau; with an empty recipe they collapse to the original.
7. **Fringing sources:** per-channel RGB sharpening, per-channel gamma on partly clipped pixels, and an R±/B∓ temperature shift that does not preserve luminance (chroma shifts measured; visible fringes not reproduced on the test image).
8. **Missing region backend** (GroundingDINO absent): semantic actions are skipped or turned into quadrant guesses.
9. **Flash-lite VLM with default sampling and no highlight information:** it accepted the image after a ~10-level change, and the critic ranked the weakest variant first.

## Open questions

- Which tools and parameters were in the failing user run's plan and recipe (sharpen? color_correct? temperature)? That decides whether the fringing came from sharpening, the temperature shift or gamma on clipped channels. I could not reproduce visible red/cyan fringes on the test image.
- Which exit path produced the user's `STOP_BEST_EFFORT` (A/B/C/D)? The run's per-iteration `history[*].decision`, `eval_score` and `rolled_back` would tell.
- Does the user's machine have the GroundingDINO/MobileSAM models (`REGION_DINO_MODEL_PATH`), and was `GEMINI_API_KEY` actually picked up (BOM issue), or did that run use the rule-based path?
- Did the VLM rate the overexposure severity 2 or 3 in the user's run (γ 0.8 vs 0.7)?
- How is the UI style selector (Tự nhiên / Cân bằng / Đậm nét) wired in the user's running version? `frontend/app.py` does not pass `intent` to `run_pipeline`, and I did not read the interactive-session UI path in full.
- Are pyiqa BRISQUE/NIQE meant to feed decisions? They are computed and stored but unused.
