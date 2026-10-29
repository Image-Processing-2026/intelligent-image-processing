# Kế Hoạch Test Toàn Diện: Module 1 (Evaluator) ⟷ Module 4 (Agent Orchestrator)

> **Mục tiêu:** Kiểm chứng trọn vẹn logic tương tác giữa Module 1 và Module 4, đảm bảo hệ thống
> hoạt động đúng ý tưởng thiết kế của project — từ chẩn đoán, xử lý, đánh giá, đến ra quyết định.
>
> **Ngày tạo:** 2026-10-01
> **Phạm vi:** Chỉ Module 1 + Module 4 (Module 2 & 3 dùng baseline OpenCV trên nhánh `develop`)

---

## Mục Lục

1. [Tổng quan hiện trạng](#1-tổng-quan-hiện-trạng)
2. [Ma trận lỗ hổng test (Gap Analysis)](#2-ma-trận-lỗ-hổng-test-gap-analysis)
3. [Kế hoạch test chi tiết](#3-kế-hoạch-test-chi-tiết)
4. [Bảng tham chiếu ngưỡng & hằng số](#4-bảng-tham-chiếu-ngưỡng--hằng-số)
5. [Hướng dẫn chạy test](#5-hướng-dẫn-chạy-test)

---

## 1. Tổng Quan Hiện Trạng

### Tests đã có

| File | Số tests | Phạm vi |
|------|----------|---------|
| `tests/unit/test_analyzer_evaluator.py` | 81 | Schema, analyzer, reference eval, no-reference eval, synthetic loader, performance |
| `tests/unit/test_agent_planner.py` | 33 | Plan validation, clamping, decision logic, INT-01/02/03 unit tests |
| `tests/unit/test_agent_executor.py` | 12 | Mask validation, execution, fault tolerance, face detection |
| `tests/unit/test_vlm_mock.py` | 10 | Mock VLM responses, fallback engine, multi-turn closed-loop |
| `tests/integration/test_pipeline_loop.py` | 1 | Synthetic noisy image full pipeline |
| `tests/integration/test_pipeline_synthetic_benchmark.py` | 4 | Underexposed, noisy, degradation guard, zero-redundant compute |
| `tests/integration/test_api_routes.py` | 5 | FastAPI endpoints |
| **Tổng** | **146** | |

---

## 2. Ma Trận Lỗ Hổng Test (Gap Analysis)

Sau khi rà soát toàn bộ codebase và bộ test hiện có, các lỗ hổng test quan trọng được xác định:

### 🔴 Ưu tiên cao (Ảnh hưởng logic cốt lõi)

| # | Lỗ hổng | Giải thích | Liên quan |
|---|---------|------------|-----------|
| G1 | **Rollback về ảnh tốt nhất khi suy thoái** | Spec `04_decision_and_evaluation.md` mô tả: Nếu vòng k làm ảnh xấu đi, hệ thống phải rollback về ảnh vòng k-1. Hiện tại `decide_node` chỉ set `STOP_BEST_EFFORT` mà **không thay `current_image` về `previous_image`**. Cần test xác nhận hành vi này (và có thể cần sửa code). | `graph.py` L145-180 |
| G2 | **Vòng lặp multi-turn thực tế (2-3 vòng)** | Chưa có test E2E chạy đủ 2-3 vòng lặp, kiểm tra: (a) history entries có đúng iteration, (b) previous_image chuyển đúng giữa các vòng, (c) delta_metrics tích lũy đúng | `graph.py`, `state.py` |
| G3 | **ADR-002 routing enforcement** | `evaluate_quality()` phải route đúng: synthetic+GT → reference; real → no-reference. Cần test integration xác nhận pipeline không bao giờ gọi PSNR/SSIM cho ảnh real | `__init__.py` |
| G4 | **Fallback rule-based khi không có API key** | Hiện tại test chỉ mock VLM. Cần test E2E **thực sự** chạy qua fallback (không set `GEMINI_API_KEY`) với từng loại ảnh để xác nhận plan sinh ra đúng | `vlm_diagnostician.py` |

### 🟡 Ưu tiên trung bình (Edge cases & robustness)

| # | Lỗ hổng | Giải thích |
|---|---------|------------|
| G5 | **Color cast detection → color_correct action** | Analyzer phát hiện color cast nhưng fallback engine không sinh action `color_correct`. Cần test xác nhận hành vi này và document |
| G6 | **Plan với multi-region composite** | Chưa test plan có nhiều vùng khác nhau (face + sky + background) chạy qua executor trong cùng 1 vòng |
| G7 | **Ảnh cực đoan qua pipeline** | Ảnh toàn đen, toàn trắng, 1×1 px, panorama 1×1000 chạy qua toàn pipeline — phải không crash |
| G8 | **evaluate_node trả technical_metrics đầy đủ cho decide_node** | Zero-redundant compute: test rằng `evaluation_result["technical_metrics"]` luôn có đầy đủ các key cần thiết |

### 🟢 Ưu tiên thấp (Nice-to-have)

| # | Lỗ hổng | Giải thích |
|---|---------|------------|
| G9 | **Overexposed image E2E** | Có fixture `create_overexposed_image` nhưng chưa có test E2E cho nó |
| G10 | **Salt & pepper noise E2E** | Có fixture nhưng chưa test E2E |
| G11 | **Blurred image E2E** | Có fixture nhưng chưa test E2E (blur level → diagnose → sharpen plan) |
| G12 | **Heuristic score monotonicity qua pipeline** | Test rằng khi pipeline xử lý thành công, heuristic score tăng dần qua các vòng |

---

## 3. Kế Hoạch Test Chi Tiết

### 📦 Phase 1: Unit Tests — Logic Quyết Định & Handshake (Ưu tiên cao)

> **File:** `tests/unit/test_decision_logic.py`
> **Mục tiêu:** Test isolated logic của `_decide_synthetic`, `_decide_real`, và `decide_node`

#### Test 1.1: `test_decide_synthetic_ship_when_targets_met`
- **Input:** `eval_result = {"psnr": 30.0, "ssim": 0.92}`, `history = []`
- **Assert:** Trả về `"SHIP"`
- **Ý nghĩa:** Khi PSNR ≥ 28 VÀ SSIM ≥ 0.88, Agent phải dừng và giao ảnh

#### Test 1.2: `test_decide_synthetic_reprocess_when_below_targets`
- **Input:** `eval_result = {"psnr": 22.0, "ssim": 0.75}`, `history = []`
- **Assert:** Trả về `"RE_PROCESS"`
- **Ý nghĩa:** Chưa đạt ngưỡng → Agent tiếp tục lặp

#### Test 1.3: `test_decide_synthetic_stop_on_psnr_degradation`
- **Input:** `eval_result = {"psnr": 20.0, "ssim": 0.70}`, `history` chứa entry trước có `psnr = 25.0`
- **Assert:** Trả về `"STOP_BEST_EFFORT"` (vì `25.0 - 20.0 = 5.0 > PSNR_DEGRADATION (1.5)`)
- **Ý nghĩa:** Agent phát hiện xử lý làm ảnh xấu đi → dừng ngay

#### Test 1.4: `test_decide_synthetic_continue_on_minor_dip`
- **Input:** `eval_result = {"psnr": 24.0, "ssim": 0.80}`, `history` chứa entry trước có `psnr = 25.0`
- **Assert:** Trả về `"RE_PROCESS"` (vì `25.0 - 24.0 = 1.0 < 1.5`)
- **Ý nghĩa:** Sụt giảm nhỏ trong ngưỡng cho phép → tiếp tục

#### Test 1.5: `test_decide_real_stop_on_quality_improved_false`
- **Input:** `eval_result = {"quality_improved": False, "estimated_quality_score": 40}`
- **Assert:** Trả về `"STOP_BEST_EFFORT"`
- **Ý nghĩa:** Module 1 phát hiện cháy sáng/bùng nhiễu → INT-03 Degradation Guard kích hoạt

#### Test 1.6: `test_decide_real_stop_on_quality_score_drop`
- **Input:** `eval_result = {"quality_improved": True, "estimated_quality_score": 45}`, `history` có entry trước `estimated_quality_score = 55`
- **Assert:** Trả về `"STOP_BEST_EFFORT"`
- **Ý nghĩa:** Quality score giảm qua các vòng → dừng

#### Test 1.7: `test_decide_real_continue_on_improvement`
- **Input:** `eval_result = {"quality_improved": True, "estimated_quality_score": 65}`, `history` có entry trước `estimated_quality_score = 55`
- **Assert:** Trả về `"RE_PROCESS"`

#### Test 1.8: `test_decide_node_max_iterations_hard_cap`
- **Setup:** `state["iteration"] = 3`, `state["max_iterations"] = 3`
- **Assert:** `decision = "STOP_BEST_EFFORT"` bất kể eval result
- **Ý nghĩa:** Hard-cap 3 vòng phải luôn được tôn trọng

#### Test 1.9: `test_decide_node_empty_plan_means_ship`
- **Setup:** VLM trả về plan rỗng (0 actions)
- **Assert:** `decision = "SHIP"`
- **Ý nghĩa:** VLM cho rằng ảnh đã tốt → Agent tin tưởng và giao ảnh

---

### 📦 Phase 2: Integration Tests — Full Pipeline Scenarios

> **File:** `tests/integration/test_mod1_mod4_integration.py`
> **Mục tiêu:** Test luồng E2E qua LangGraph với các kịch bản ảnh thực tế

#### Test 2.1: `test_underexposed_pipeline_synthetic_improvement`
```
Input: checkerboard × factor=0.3 (underexposed) + ground_truth
Pipeline: run_pipeline(is_synthetic=True, max_iterations=3)
Assert:
  - history[0].plan chứa action "gamma_correct" (fallback engine)
  - eval_result có psnr > 0 và ssim > 0
  - Nếu PSNR ≥ 28 & SSIM ≥ 0.88 → decision = "SHIP"
  - Nếu chưa đạt → decision = "STOP_BEST_EFFORT" (hết vòng)
  - history đầy đủ metrics_before và metrics_after
```

#### Test 2.2: `test_noisy_pipeline_synthetic_denoising`
```
Input: flat_128 + gaussian noise (std=30) + ground_truth
Pipeline: run_pipeline(is_synthetic=True, max_iterations=3)
Assert:
  - history[0].plan chứa action "denoise" (noise_level sẽ là "severe")
  - metrics_after["noise_variance"] ≤ metrics_before["noise_variance"]
  - delta_metrics có delta_psnr, delta_ssim, delta_mse
```

#### Test 2.3: `test_overexposed_pipeline_no_reference`
```
Input: checkerboard × factor=2.5 (overexposed), KHÔNG có ground_truth
Pipeline: run_pipeline(is_synthetic=False, max_iterations=2)
Assert:
  - eval_result KHÔNG có psnr/ssim (ADR-002 compliance)
  - eval_result CÓ estimated_quality_score
  - eval_result CÓ quality_improved (bool)
```

#### Test 2.4: `test_blurred_image_pipeline`
```
Input: checkerboard bị GaussianBlur(kernel=15)
Pipeline: run_pipeline(is_synthetic=False, max_iterations=2)
Assert:
  - metrics_before["blur_level"] là "severe_blur" hoặc "mild_blur"
  - Pipeline không crash
  - decision là một trong ["SHIP", "STOP_BEST_EFFORT"]
```

#### Test 2.5: `test_multi_turn_previous_image_tracking`
```
Input: noisy image, max_iterations=3
Pipeline: run_pipeline(is_synthetic=True, max_iterations=3)
Assert:
  - Vòng 1: previous_image PHẢI là original_image
  - Vòng 2+: previous_image PHẢI là ảnh đã xử lý ở vòng trước
  - len(history) == max_iterations (hoặc ít hơn nếu SHIP sớm)
  - Mỗi history entry có iteration tăng dần: 1, 2, 3
```

#### Test 2.6: `test_multi_turn_delta_metrics_accumulation`
```
Input: noisy image, max_iterations=2
Pipeline: run_pipeline(is_synthetic=True, max_iterations=2)
Assert:
  - history[0].eval_score có "delta_metrics" (có thể rỗng nếu vòng 1)
  - history[1].eval_score["delta_metrics"] PHẢI có delta_psnr, delta_ssim
  - delta_metrics phản ánh sự thay đổi so với vòng liền trước
```

#### Test 2.7: `test_extreme_images_no_crash` (Parametrized)
```
Tham số hóa cho: toàn đen, toàn trắng, 1×1 px, 3×3 px, panorama 1×500
Pipeline: run_pipeline(max_iterations=1)
Assert:
  - Không raise Exception
  - state["decision"] là string hợp lệ
  - state["current_image"] là np.ndarray hợp lệ
```

---

### 📦 Phase 3: Integration Tests — Cơ Chế Bảo Vệ & Contracts

> **File:** `tests/integration/test_mod1_mod4_guards.py`
> **Mục tiêu:** Test các cơ chế bảo vệ và hợp đồng giữa 2 module

#### Test 3.1: `test_degradation_guard_stops_immediately`
```
Setup: Mock evaluate_no_reference trả về quality_improved=False
Pipeline: run_pipeline(is_synthetic=False, max_iterations=3)
Assert:
  - Pipeline dừng SAU ĐÚNG 1 vòng (không lặp tiếp)
  - decision = "STOP_BEST_EFFORT"
  - len(history) == 1
```

#### Test 3.2: `test_rollback_behavior_on_degradation`
```
Setup: Chuẩn bị ảnh mà vòng 2 làm xấu đi
Assert:
  - Kiểm tra state["current_image"] sau khi pipeline kết thúc
  - Nếu decision = "STOP_BEST_EFFORT" → ảnh cuối PHẢI là phiên bản
    trước khi suy thoái (previous_image), KHÔNG phải ảnh đã bị hỏng
  - ⚠️ LƯU Ý: Hiện tại code CHƯA implement rollback.
    Test này sẽ FAIL → dùng làm TDD để sửa code nếu cần.
```

#### Test 3.3: `test_adr002_no_psnr_ssim_for_real_images`
```
Input: Ảnh thực (is_synthetic=False), KHÔNG có ground_truth
Pipeline: run_pipeline(is_synthetic=False)
Assert:
  - eval_result.get("psnr") is None
  - eval_result.get("ssim") is None
  - eval_result.get("mse") is None
  - eval_result["estimated_quality_score"] is not None
```

#### Test 3.4: `test_adr002_psnr_ssim_for_synthetic_images`
```
Input: Ảnh synthetic (is_synthetic=True) + ground_truth
Pipeline: run_pipeline(is_synthetic=True)
Assert:
  - eval_result["psnr"] is not None
  - eval_result["ssim"] is not None
  - eval_result["psnr"] > 0
  - eval_result["ssim"] > -1
```

#### Test 3.5: `test_zero_redundant_compute_in_decide_node`
```
Setup: Mock analyze_image, chạy pipeline
Assert:
  - analyze_image KHÔNG bao giờ được gọi từ decide_node
  - Tổng số lần gọi analyze_image = (1 từ analyze_node) + (N từ evaluate_node)
  - Không có lần gọi thừa
```

#### Test 3.6: `test_plan_validator_rejects_invalid_operations`
```
Setup: Force VLM trả về plan chứa operation "magic_enhance" (không hợp lệ)
Assert:
  - Sau validate_and_sort_plan, action "magic_enhance" bị loại bỏ
  - Các action hợp lệ vẫn được giữ nguyên
```

#### Test 3.7: `test_plan_validator_enforces_denoise_before_sharpen`
```
Setup: Plan chứa [sharpen (order=1), denoise (order=2)]
Assert:
  - Sau validate_and_sort_plan: denoise (priority=10) trước sharpen (priority=40)
```

---

### 📦 Phase 4: Integration Tests — Fallback Engine & VLM Handshake

> **File:** `tests/integration/test_fallback_vlm_pipeline.py`
> **Mục tiêu:** Test hệ thống VLM fallback rule-based trong pipeline thực tế

#### Test 4.1: `test_fallback_underexposed_generates_gamma_correct`
```
Input: Ảnh thiếu sáng (brightness_mean < 70)
Điều kiện: Không set GEMINI_API_KEY
Assert:
  - diagnose_and_plan trả về plan có action "gamma_correct"
  - gamma = 1.3 (giá trị mặc định fallback)
  - reasoning chứa "Fallback Rule-Based"
```

#### Test 4.2: `test_fallback_noisy_generates_denoise`
```
Input: Ảnh nhiễu (noise_level = "severe")
Điều kiện: Không set GEMINI_API_KEY
Assert:
  - Plan có action "denoise" với method="bilateral", strength=1.0
```

#### Test 4.3: `test_fallback_low_contrast_generates_clahe`
```
Input: Ảnh contrast_std < 40, brightness bình thường
Assert:
  - Plan có action "clahe" với clip_limit=2.0
```

#### Test 4.4: `test_fallback_clean_image_generates_no_actions`
```
Input: Ảnh sạch (noise=clean, brightness=normal, contrast=normal)
Assert:
  - Plan có 0 actions
  - decide_node sẽ trả về "SHIP" (kế hoạch rỗng = ảnh đã tốt)
```

#### Test 4.5: `test_fallback_noisy_underexposed_generates_both_actions`
```
Input: Ảnh vừa thiếu sáng VÀ nhiễu
Assert:
  - Plan có 2 actions: ["denoise", "gamma_correct"]
  - Sau validate_and_sort: denoise (order=10) TRƯỚC gamma_correct (order=20)
```

#### Test 4.6: `test_history_feedback_influences_vlm_prompt`
```
Setup: Tạo history giả lập, gọi _build_history_feedback
Assert:
  - Output chứa thông tin operations, delta metrics, decision của vòng trước
  - Output chứa cảnh báo "KHÔNG lặp lại cùng thao tác"
```

---

### 📦 Phase 5: Stress Tests — Edge Cases & Robustness

> **File:** `tests/integration/test_pipeline_edge_cases.py`

#### Test 5.1: `test_salt_pepper_noise_pipeline`
```
Input: create_salt_pepper_image(flat_128, prob=0.05)
Pipeline: run_pipeline(max_iterations=2)
Assert: Không crash, có decision hợp lệ
```

#### Test 5.2: `test_color_cast_detection_in_pipeline`
```
Input: create_color_cast_image(flat_128, cast_type="warm")
Pipeline: run_pipeline(max_iterations=1)
Assert:
  - metrics["color_cast"] == "warm"
  - Pipeline không crash
  - ⚠️ Lưu ý: Fallback engine hiện KHÔNG sinh action color_correct cho color cast
```

#### Test 5.3: `test_combined_degradation_pipeline`
```
Input: Ảnh vừa noisy, vừa underexposed, vừa blurred
Pipeline: run_pipeline(max_iterations=3)
Assert:
  - Pipeline xử lý tuần tự, history ghi nhận đầy đủ
  - Mỗi vòng có thể xử lý 1 hoặc nhiều vấn đề
```

#### Test 5.4: `test_thumbnail_dimensions_in_intermediate_images`
```
Input: Ảnh lớn (1000×1000 px)
Pipeline: run_pipeline(max_iterations=2)
Assert:
  - Tất cả ảnh trong intermediate_images có max(H,W) ≤ 512
  - Đúng số lượng thumbnail = số vòng đã chạy
```

#### Test 5.5: `test_pipeline_determinism_with_seed`
```
Input: Ảnh noisy cùng seed=42
Chạy: 2 lần run_pipeline với cùng input
Assert: Kết quả quyết định giống nhau (deterministic)
```

---

## 4. Bảng Tham Chiếu Ngưỡng & Hằng Số

### Module 1 — Analyzer

| Metric | Ngưỡng | Phân loại |
|--------|--------|-----------|
| `brightness_mean` | < 70 | `underexposed` |
| `brightness_mean` | > 185 | `overexposed` |
| `shadow_clip_ratio` | > 0.25 | `underexposed` |
| `highlight_clip_ratio` | > 0.20 | `overexposed` |
| `contrast_std` | < 40 | `low` |
| `contrast_std` | > 80 | `high` |
| `noise_variance` | < 3.0 | `clean` |
| `noise_variance` | [3.0, 8.0) | `low` |
| `noise_variance` | [8.0, 15.0) | `medium` |
| `noise_variance` | ≥ 15.0 | `severe` |
| `sharpness_laplacian_var` | < 100 | `severe_blur` |
| `sharpness_laplacian_var` | [100, 300) | `mild_blur` |
| `sharpness_laplacian_var` | ≥ 300 | `sharp` |

### Module 1 — No-Reference Quality Logic

| Điều kiện | Kết quả `quality_improved` |
|-----------|---------------------------|
| `delta_noise > 10.0` | `False` (bùng nổ nhiễu) |
| `curr_highlight_clip > 0.15 AND > prev × 1.5` | `False` (cháy sáng mới) |
| `delta_noise < -1.5 AND sharpness_drop < 30%` | `True` (khử nhiễu OK) |
| `delta_sharpness > 20 AND delta_noise < 3.0` | `True` (tăng nét OK) |
| `delta_brightness > 0` | `True` (sáng gần 128) |

### Module 4 — Decision Thresholds

| Hằng số | Giá trị | Sử dụng |
|---------|---------|---------|
| `PSNR_THRESHOLD` | 28.0 dB | SHIP nếu PSNR ≥ |
| `SSIM_THRESHOLD` | 0.88 | SHIP nếu SSIM ≥ |
| `PSNR_DEGRADATION` | 1.5 dB | STOP nếu ΔPSNR > |
| `MAX_ITERATIONS` | 3 (default) | Hard cap |

### Module 4 — Planner Parameter Bounds

| Operation | Parameter | Min | Max | Default |
|-----------|-----------|-----|-----|---------|
| `denoise` | `strength` | 0.1 | 2.0 | 1.0 |
| `denoise` | `method` | — | — | `bilateral` |
| `gamma_correct` | `gamma` | 0.5 | 2.5 | 1.2 |
| `clahe` | `clip_limit` | 1.0 | 4.0 | 2.0 |
| `sharpen` | `amount` | 0.2 | 2.0 | 1.0 |
| `sharpen` | `method` | — | — | `unsharp_mask` |
| `color_correct` | `saturation_scale` | 0.5 | 1.5 | 1.0 |
| `color_correct` | `temperature_shift` | -1.0 | 1.0 | 0.0 |

---

## 5. Hướng Dẫn Chạy Test

### Chạy toàn bộ test suite
```bash
pytest tests/ -v --tb=short
```

### Chạy riêng từng phase
```bash
# Phase 1: Unit tests decision logic
pytest tests/unit/test_decision_logic.py -v

# Phase 2: Integration pipeline scenarios
pytest tests/integration/test_mod1_mod4_integration.py -v

# Phase 3: Guards & contracts
pytest tests/integration/test_mod1_mod4_guards.py -v

# Phase 4: Fallback & VLM
pytest tests/integration/test_fallback_vlm_pipeline.py -v

# Phase 5: Edge cases
pytest tests/integration/test_pipeline_edge_cases.py -v
```

### Chạy test với coverage
```bash
pytest tests/ -v --cov=src/agent --cov=src/analyzer_evaluator --cov-report=term-missing
```

### Lưu ý quan trọng
- **Không cần** `GEMINI_API_KEY` — toàn bộ test dùng fallback rule-based hoặc mock
- **Module 2 & 3** dùng baseline OpenCV (đã có trên `develop`), không cần model AI
- Nếu test 3.2 (Rollback) FAIL → đó là **expected** — dùng làm TDD để implement rollback

---

> [!IMPORTANT]
> Tổng cộng **~40 test cases mới** cần viết, chia 5 phases.
> Ưu tiên triển khai: **Phase 1 → Phase 3 → Phase 2 → Phase 4 → Phase 5**
> (Logic quyết định → Guards → E2E scenarios → Fallback → Edge cases)

> [!TIP]
> Sau khi merge PR Module 2 & 3, cần chạy lại Phase 2 & Phase 5
> vì kết quả xử lý ảnh thực tế sẽ thay đổi so với baseline.
