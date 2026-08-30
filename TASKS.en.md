# Intelligent Image Processing — Module Handbook & Task Roadmap

This handbook is designed for **all 4 team members** and their AI coding assistants. Each member can refer to their dedicated module section to understand:
- The module's role within the global diagnosis and treatment loop.
- Cross-module data contracts (Input received from whom, Output handed to whom).
- Required classical Image Processing algorithms, mathematical formulas, and libraries.
- The concrete task checklist.
- Technical pitfalls and critical caveats to avoid.

> **Progress Tracking:** Update checkboxes from `[ ]` (Todo) → `[/]` (In Progress) → `[x]` (Done) as implementation proceeds.

---

## 🔁 Global System Architecture & Feedback Loop

```
Image → [Module 1: Analyze] → [VLM: Diagnose] → [Module 2: Locate Regions] → [Module 4: Plan]
                                                                                      │
                                              ┌──────────────────────────────────────────┘
                                              ▼
                                   [Module 3: Process] → [Module 1: Evaluate] → [Module 4: Decide]
                                                                                      │
                                                                   ┌──────────────────┴──────────────────┐
                                                              Quality Bar Met                      Not Yet Met
                                                                   │                                    │
                                                                   ▼                                    ▼
                                                                 SHIP                         Re-Plan (Max 3 iterations)
```

**Core Philosophy:** The AI is the orchestrating brain (perception, diagnosis, tool planning, loop decision). All pixel manipulation is performed by explicit classical image processing algorithms (OpenCV, scikit-image) — this forms the graded substance of the Image Processing course.

---

## 👥 Team Ownership & Inter-Module Data Flow

| Module | Core Responsibility | Analogy | Assignee | Primary Directory |
|---|---|---|---|---|
| **Module 1** | Technical Metric Extraction & Quality Evaluation | 🔬 **Sensory / QC** | **Person 1** | `src/analyzer_evaluator/` |
| **Module 2** | Spatial / Semantic Localization & Soft-Mask Blending | 👁️ **Eyes** | **Person 2** | `src/region_engine/` |
| **Module 3** | Classical Image Processing Algorithms Execution | ✋ **Hands** | **Person 3** | `src/processing_engine/` |
| **Module 4** | LangGraph Orchestration, VLM Diagnostician & Backend | 🧠 **Brain** | **Person 4** | `src/agent/`, `src/api/`, `frontend/` |

### Inter-Module Data Flow Diagram

```
Person 4 (Agent) ─── calls ──→ Person 1 (Analyzer)    → returns TechnicalMetrics
Person 4 (Agent) ─── calls ──→ Person 2 (Region)      → returns Soft-Mask float32 [0.0, 1.0]
Person 4 (Agent) ─── calls ──→ Person 3 (Processing)  → returns Processed Image
Person 4 (Agent) ─── calls ──→ Person 1 (Evaluator)   → returns EvaluationResult
Person 3 (Processing) ── uses ─→ Person 2 (mask_utils.blend_regions) → Alpha-blend output
```

---
---

## 📋 Module 1: Image Analyzer & Evaluator — 🔬 Sensory & Quality Control

**Assignee:** Person 1  
**Directory:** `src/analyzer_evaluator/`  
**Key Files:** `analyzer.py`, `reference_eval.py`, `no_reference_eval.py`

---

### 1.1 Role in the System

Module 1 executes at **two distinct stages** in the loop:
1. **Loop Start (`analyze_image`):** Computes quantitative technical metrics (brightness, contrast, noise, blur, color balance) so the VLM knows *what* is wrong.
2. **Loop End (`evaluate_reference` / `evaluate_no_reference`):** Measures the output quality to decide whether to `SHIP` or `RE_PROCESS`.

---

### 1.2 Input / Output Contracts

| Function | Called By | Input Data | Output Data | Handed To |
|---|---|---|---|---|
| `analyze_image(image)` | Module 4 (Agent) | RGB `np.uint8` array `(H, W, 3)` | `TechnicalMetrics` object | Module 4 (Agent → VLM prompt) |
| `evaluate_reference(current, ground_truth)` | Module 4 (Agent) | 2× RGB `np.uint8` arrays | `{"psnr": float, "ssim": float, "mse": float}` | Module 4 (Agent: decision node) |
| `evaluate_no_reference(current, previous?)` | Module 4 (Agent) | 1–2× RGB `np.uint8` arrays | `{"brisque": float, "delta_contrast": float, ...}` | Module 4 (Agent: decision node) |

**Primary Pydantic Schema — `TechnicalMetrics`:**
```python
class TechnicalMetrics(BaseModel):
    brightness_mean: float          # Mean luminance [0, 255]
    brightness_level: str           # "underexposed" | "normal" | "overexposed"
    contrast_std: float             # Standard deviation of luminance
    contrast_level: str             # "low" | "normal" | "high"
    noise_variance: float           # Estimated noise variance
    noise_level: str                # "clean" | "low" | "medium" | "severe"
    sharpness_laplacian_var: float  # Variance of Laplacian operator
    blur_level: str                 # "sharp" | "mild_blur" | "severe_blur"
    color_cast: str                 # "warm" | "cool" | "greenish" | "none"
    histogram_stats: dict           # Histogram statistics (min, max, skewness)
```

---

### 1.3 Image Processing Algorithms to Implement

#### A. Technical Image Analysis (`analyzer.py`)

| Metric | Algorithm / Mathematical Formula | Library |
|---|---|---|
| **Brightness** | Convert to grayscale, compute `np.mean(gray)`. Categorize (<70: underexposed, >185: overexposed). | NumPy |
| **Contrast** | Standard deviation of grayscale values: `np.std(gray)`. Low < 40, High > 80. | NumPy |
| **Blur / Sharpness** | Variance of the 2nd-order differential **Laplacian operator** $\nabla^2 f$: `cv2.Laplacian(gray, CV_64F).var()`. Low variance indicates lack of sharp edges. | OpenCV |
| **Noise Estimation** | Residual difference between raw grayscale image and a 3×3 **Median Filter**: `cv2.absdiff(gray, cv2.medianBlur(gray, 3))`. | OpenCV |
| **Histogram Distribution** | Compute 3-channel histogram `cv2.calcHist()` to inspect dynamic range and clipping. | OpenCV |
| **Color Cast** | Compare mean channel intensities ($\bar{R}, \bar{G}, \bar{B}$). Significant offset identifies warm/cool cast. | NumPy |

#### B. Full-Reference Benchmark Evaluation (`reference_eval.py`)

| Metric | Formula | Library |
|---|---|---|
| **MSE** | $\text{MSE} = \frac{1}{N}\sum_{i=1}^N (I_{\text{ref}}[i] - I_{\text{cur}}[i])^2$ | NumPy |
| **PSNR** | $\text{PSNR} = 20 \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right)\text{ dB}$ | scikit-image / NumPy |
| **SSIM** | Multi-scale structural similarity (luminance, contrast, structural correlation). | scikit-image |

#### C. No-Reference Perceptual Evaluation (`no_reference_eval.py`)

| Metric | Description | Library |
|---|---|---|
| **BRISQUE** | Blind/Referenceless Image Spatial Quality Evaluator (lower score = higher quality, 0–100). | `pyiqa` or OpenCV |
| **NIQE** | Naturalness Image Quality Evaluator (lower score = more natural image statistics). | `pyiqa` |
| **Delta Metrics** | Iteration-over-iteration metric variance tracking ($\Delta\text{contrast}, \Delta\text{sharpness}, \Delta\text{noise}$). | Custom |

---

### 1.4 Task Checklist

- [ ] **Technical Metrics Extraction (`analyzer.py`)**
  - [ ] Implement mean brightness & classification (`underexposed`, `normal`, `overexposed`).
  - [ ] Implement contrast standard deviation & dynamic range calculation.
  - [ ] Implement noise variance estimation using median filter residual.
  - [ ] Implement blur detection via Laplacian variance & Tenengrad gradient.
  - [ ] Implement RGB/HSV histogram distribution analysis & color cast detection.
- [ ] **Reference-Based Evaluation (`reference_eval.py`)**
  - [ ] Implement PSNR, SSIM, and MSE metrics using `scikit-image`.
  - [ ] Build synthetic dataset loader (paired clean + artificially degraded benchmark set).
- [ ] **No-Reference Quality Evaluation (`no_reference_eval.py`)**
  - [ ] Implement BRISQUE / NIQE metric calculation (via `pyiqa` or OpenCV).
  - [ ] Implement delta-metric comparison across iterations (tracking improvement/degradation).
- [ ] **Unit Tests (`tests/unit/test_analyzer_evaluator.py`)**

---

### 1.5 Critical Caveats & Gotchas

> [!CAUTION]
> **DO NOT compute PSNR/SSIM on real user images.** Real-world photos (old damaged prints, blurry mobile snapshots) have no pristine ground truth. Computing PSNR against an arbitrary reference is mathematically invalid. Restrict PSNR/SSIM exclusively to `data/synthetic/`.

> [!IMPORTANT]
> Ensure all functions handle single-channel grayscale `(H, W)` and 3-channel RGB `(H, W, 3)` inputs gracefully without raising dimension errors.

---
---

## 📋 Module 2: Region Engine & Mask Synthesis — 👁️ Spatial Eyes

**Assignee:** Person 2  
**Directory:** `src/region_engine/`  
**Key Files:** `detector.py`, `face_detector.py`, `spatial.py`, `mask_utils.py`

---

### 2.1 Role in the System

Module 2 acts as the **"Eyes"** answering the question **"Where?"**:
- When the Agent dictates: *"Correct exposure on the sky and denoise the background"*, Module 2 generates pixel-accurate **continuous soft-masks** localizing the "sky" and "background".
- Module 3 uses these masks to isolate modifications without affecting the rest of the image.

**This is the most critical module for visual realism:** imperfect masks create harsh visible boundary seams.

---

### 2.2 Input / Output Contracts

| Function | Called By | Input Data | Output Data | Handed To |
|---|---|---|---|---|
| `segment_by_prompt(image, text_prompt)` | Module 4 (Agent/Executor) | RGB `uint8` image + string (e.g. `"sky"`) | Soft-mask `float32 [0.0, 1.0]` shape `(H, W)` | Module 3 & Module 4 |
| `detect_faces(image)` | Module 4 (Agent/Executor) | RGB `uint8` image | List of facial soft-masks `float32 [0.0, 1.0]` | Module 3 & Module 4 |
| `create_soft_mask(binary_mask, feather_radius)` | Internal Module 2 | Binary mask `uint8` (0 or 255) | Soft-mask `float32 [0.0, 1.0]` | Module 2 exports |
| `blend_regions(original, processed, soft_mask)` | Module 3 (Processing Engine) | 2× RGB `uint8` images + soft-mask | Blended RGB `uint8` image | Module 4 (Agent) |

**Strict Mask Specifications:**
- Data Type: `np.float32`
- Value Range: `[0.0, 1.0]` continuous values (`0.0` = unaffected background, `1.0` = full effect).
- Spatial Shape: `(H, W)`, strictly identical to the target image dimensions.

---

### 2.3 Image Processing Algorithms to Implement

#### A. Lightweight Face Localization (`face_detector.py`)

| Technique | Details | Library |
|---|---|---|
| **MediaPipe Face Detection** | CPU/iGPU-optimized detector. Returns normalized relative bounding boxes. Expand bounding box by ~15% to fully cover forehead and chin. | `mediapipe` |
| **BBox to Mask Conversion** | Render binary rectangle mask, then pass to `create_soft_mask()` for edge feathering. | NumPy + OpenCV |

#### B. Open-Vocabulary Semantic Segmentation (`detector.py`)

| Technique | Details | Library |
|---|---|---|
| **GroundingDINO** | Takes raw text prompt (e.g. "sky", "vegetation") and returns object bounding boxes. | `groundingdino` |
| **MobileSAM / SAM2** | Takes bounding boxes and produces pixel-accurate instance masks. | `mobile_sam` |
| **Heuristic Fallback** | Fallback when model weights are not loaded (e.g. "sky" → top quadrant mask). | NumPy |

#### C. Geometric & Spatial Masks (`spatial.py`)

| Technique | Details |
|---|---|
| **Bounding Box Mask** | Generates rectangular mask from coordinates `(xmin, ymin, xmax, ymax)`. |
| **Quadrant Mask** | Generates directional mask for `"top"`, `"bottom"`, `"left"`, `"right"`, `"center"`. |

#### D. Edge Feathering & Alpha Blending (`mask_utils.py`) — ⭐ CORE SUBSTANCE

| Technique | Mathematical Formula / Implementation | Library |
|---|---|---|
| **Gaussian Edge Feathering** | Smooths binary mask boundaries into a continuous transition gradient using a 2D Gaussian kernel: $\text{Mask}_{\text{soft}} = \text{Mask}_{\text{bin}} * G_{\sigma}$. Kernel size must be odd. | OpenCV |
| **Alpha Compositing** | $I_{\text{out}} = \text{Mask}_{\text{soft}} \odot I_{\text{processed}} + (1.0 - \text{Mask}_{\text{soft}}) \odot I_{\text{original}}$ — Linear pixel interpolation across all 3 color channels. | NumPy |

---

### 2.4 Task Checklist

- [ ] **Face Detection (`face_detector.py`)**
  - [ ] Integrate MediaPipe Face Detection for CPU execution.
  - [ ] Convert bounding boxes to binary masks with 15% margin expansion.
  - [ ] Apply `create_soft_mask()` to feather facial mask borders.
- [ ] **Semantic Segmentation (`detector.py`)**
  - [ ] Integrate GroundingDINO + MobileSAM/SAM2 for prompt-based segmentation.
  - [ ] Ensure model executes smoothly on CPU/iGPU without CUDA crashes.
  - [ ] Implement heuristic fallback for offline testing.
- [ ] **Spatial & Grid Utilities (`spatial.py`)**
  - [ ] Implement bounding box and quadrant mask generators.
- [ ] **Soft-Mask & Alpha Compositing (`mask_utils.py`)**
  - [ ] Implement Gaussian edge feathering (`float32 [0.0, 1.0]`).
  - [ ] Implement 3-channel `blend_regions()` alpha compositing function.
- [ ] **Unit Tests (`tests/unit/test_region_engine.py`)**

---

### 2.5 Critical Caveats & Gotchas

> [!CAUTION]
> **Masks MUST be exported as `np.float32` in `[0.0, 1.0]`.** Exporting hard binary `0/255 uint8` masks causes visible edge seams after processing — the most severe visual flaw in regional enhancement.

> [!WARNING]
> **All models must run on CPU / AMD iGPU.** The primary development hardware has no NVIDIA CUDA. Use MobileSAM or TinySAM instead of SAM-ViT-H, and MediaPipe instead of heavyweight neural networks. Single-image inference latency of 1–3 seconds is completely acceptable.

> [!TIP]
> Visually verify mask smoothness during development: `plt.imshow(soft_mask, cmap='gray')` — borders must display a smooth gradient transition from black to white.

---
---

## 📋 Module 3: Classical Image Processing Engine — ✋ Hands

**Assignee:** Person 3  
**Directory:** `src/processing_engine/`  
**Key Files:** `base.py`, `denoise.py`, `exposure_contrast.py`, `sharpen.py`, `color.py`

---

### 3.1 Role in the System

Module 3 is the **"Hands"** of the system — containing the core classical image processing algorithms graded in the course. It **never** decides what to do on its own; it receives explicit operation commands and parameters from Module 4 (Agent) and applies them accurately to the specified region.

---

### 3.2 Input / Output Contracts

**All processing operations implement a standardized wrapper signature:**

```python
def apply_<operation>(
    image: np.ndarray,              # RGB uint8 (H, W, 3)
    mask: np.ndarray | None = None, # Soft-mask float32 [0.0, 1.0] (H, W) from Module 2
    **params                        # Algorithm-specific hyperparameters
) -> np.ndarray:                    # Blended RGB uint8 output
```

| Function | Key Parameters | Mask Source |
|---|---|---|
| `apply_denoise(image, mask, method, strength)` | `method`: "gaussian"/"median"/"bilateral"/"nlm", `strength`: 0.1–2.0 | Module 2 |
| `apply_gamma(image, mask, gamma)` | `gamma`: 0.5–2.5 (>1 brighter, <1 darker) | Module 2 |
| `apply_clahe(image, mask, clip_limit, tile_grid_size)` | `clip_limit`: 1.0–4.0 | Module 2 |
| `apply_sharpen(image, mask, method, amount)` | `method`: "unsharp_mask"/"laplacian", `amount`: 0.2–2.0 | Module 2 |
| `apply_color_balance(image, mask, saturation_scale, temperature_shift)` | `saturation_scale`: 0.5–1.5, `temperature_shift`: -1.0–1.0 | Module 2 |

**Execution Logic within each function:**
1. Apply the mathematical operation across the entire image (spatial filters require surrounding pixel context to avoid boundary corruption).
2. If `mask` is provided: call `blend_regions(original, processed, mask)` to alpha-blend the edited region back into the original image.
3. If `mask is None`: return the globally processed image.

---

### 3.3 Image Processing Algorithms to Implement

#### A. Denoising Filters (`denoise.py`)

| Algorithm | Mechanism & Formula | OpenCV API | Pros & Cons |
|---|---|---|---|
| **Gaussian Blur** | Convolution with 2D Gaussian kernel $G_\sigma(x, y)$ | `cv2.GaussianBlur(img, (k,k), σ)` | Fast, uniform smoothing — blurs edges |
| **Median Filter** | Replaces each pixel with local neighborhood median | `cv2.medianBlur(img, k)` | Excellent for salt-and-pepper noise |
| **Bilateral Filter** | Non-linear filter combining spatial domain distance and photometric range intensity differences | `cv2.bilateralFilter(img, d, σ_color, σ_space)` | Preserves sharp edges while smoothing textures |
| **Non-Local Means (NLM)** | Averages pixels based on patch similarity across the whole image | `cv2.fastNlMeansDenoisingColored(img, h, ...)` | State-of-the-art detail preservation — slower |

#### B. Brightness & Contrast Enhancements (`exposure_contrast.py`)

| Algorithm | Principle & Formula | Color Space |
|---|---|---|
| **Gamma Correction** | Non-linear intensity scaling: $I_{\text{out}} = 255 \cdot \left(\frac{I_{\text{in}}}{255}\right)^{1/\gamma}$. Implemented via Lookup Table (LUT) for maximum speed. | RGB |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization. Divides image into contextual tiles, clips histogram slope to prevent noise amplification, equalizes each tile, and applies bilinear interpolation. | **LAB** — strictly applied to L (Luminance) channel only, preserving original A & B chrominance channels |

#### C. Sharpening Operations (`sharpen.py`)

| Algorithm | Mechanism & Formula |
|---|---|
| **Unsharp Masking** | Adds high-frequency detail back into the original: $I_{\text{sharp}} = I + \alpha \cdot (I - G_\sigma * I)$ |
| **Laplacian Sharpening** | Convolves with 2nd-order 3×3 differential kernel: $\begin{bmatrix} 0 & -1 & 0 \\ -1 & 4+\alpha & -1 \\ 0 & -1 & 0 \end{bmatrix}$ |

#### D. Color Balance & Adjustment (`color.py`)

| Algorithm | Mechanism | Color Space |
|---|---|---|
| **Gray World White Balance** | Scales R, G, B channels under the assumption that average scene reflectance is achromatic gray. | RGB |
| **HSV Saturation Adjustment** | Multiplies the S (Saturation) channel in HSV space to boost or desaturate color vibrancy. | HSV |
| **Temperature Shift** | Adjusts R vs. B channel offsets to render warmer or cooler tones. | RGB |

---

### 3.4 Task Checklist

- [ ] **Base Region Wrapper (`base.py`)**
  - [ ] Implement `apply_region_op(image, mask, op_func, **kwargs)` with automatic alpha blending.
- [ ] **Denoising Algorithms (`denoise.py`)**
  - [ ] Implement Gaussian, Median, Bilateral, and NLM filters parameterized by `strength`.
- [ ] **Exposure & Contrast (`exposure_contrast.py`)**
  - [ ] Implement Gamma Correction using fast 8-bit LUT.
  - [ ] Implement CLAHE in LAB color space (operating strictly on L-channel).
- [ ] **Sharpening Operations (`sharpen.py`)**
  - [ ] Implement Unsharp Masking with tunable `amount` factor.
  - [ ] Implement Laplacian kernel sharpening.
- [ ] **Color Correction (`color.py`)**
  - [ ] Implement Gray World white balancing.
  - [ ] Implement HSV Saturation scaling and Temperature Shift.
- [ ] **Unit Tests (`tests/unit/test_processing_engine.py`)**

---

### 3.5 Critical Caveats & Gotchas

> [!CAUTION]
> **CLAHE MUST operate in LAB color space on the L-channel.** Applying CLAHE directly on RGB channels causes severe chromatic distortion (e.g., human faces turning unnatural greenish-purple).

> [!WARNING]
> **DO NOT crop images before filtering.** Convolving on a cropped bounding box destroys boundary pixel context. Always filter the complete frame and composite with `blend_regions()`.

> [!TIP]
> Always enforce `np.clip(result, 0, 255).astype(np.uint8)` after float arithmetic to prevent integer wrap-around artifacts (overflow/underflow).

---
---

## 📋 Module 4: AI Agent, VLM & Backend — 🧠 Orchestrating Brain

**Assignee:** Person 4  
**Directory:** `src/agent/`, `src/api/`, `frontend/`  
**Key Files:** `state.py`, `graph.py`, `vlm_diagnostician.py`, `planner.py`, `executor.py`, `src/api/main.py`, `frontend/app.py`

---

### 4.1 Role in the System

Module 4 serves as the **"Brain & Conductor"** coordinating the closed-loop workflow:
1. **Perceive:** Receive input, call Module 1 to measure metrics, call Gemini Multimodal VLM for clinical diagnosis.
2. **Plan:** Synthesize structured region-based operations (what, where, parameters, priority order).
3. **Execute:** Call Module 2 to generate masks, call Module 3 to execute operations.
4. **Evaluate:** Call Module 1 to evaluate output quality.
5. **Decide:** `SHIP` or `RE_PROCESS` (loop back with updated state).
6. **Interface:** Expose RESTful FastAPI endpoints and interactive Gradio UI.

---

### 4.2 Input / Output Contracts

| Component | Input | Output |
|---|---|---|
| **LangGraph Pipeline** | Raw RGB `uint8` image (+ optional ground truth) | `DoctorState` with final image, history trace, and decision |
| **VLM Diagnostician** | Image array + `TechnicalMetrics` JSON (from Module 1) | `TreatmentPlan` JSON schema |
| **Plan Validator** | Raw `TreatmentPlan` from VLM | Validated & sorted `TreatmentPlan` (restricted to toolbox) |
| **Executor** | Validated `TreatmentPlan` + Current Image | Resulting image after executing all actions |
| **FastAPI Backend** | HTTP POST multipart/form-data | JSON diagnosis & base64 processed image |
| **Gradio Demo UI** | Browser file upload | Before/After visual comparison & execution logs |

**Treatment Plan Schema — `TreatmentPlan`:**
```json
{
  "iteration": 1,
  "reasoning": "Sky is overexposed, background contains moderate Gaussian noise, facial region is underexposed.",
  "actions": [
    {
      "region_id": "background",
      "target_prompt": "background",
      "detected_issue": "noise",
      "operation": "denoise",
      "parameters": {"method": "bilateral", "strength": 1.0},
      "order": 10
    },
    {
      "region_id": "sky",
      "target_prompt": "sky",
      "detected_issue": "overexposed",
      "operation": "gamma_correct",
      "parameters": {"gamma": 0.8},
      "order": 20
    },
    {
      "region_id": "face",
      "target_prompt": "face",
      "detected_issue": "underexposed",
      "operation": "gamma_correct",
      "parameters": {"gamma": 1.3},
      "order": 21
    }
  ]
}
```

**Fixed Agent Toolbox (The agent may ONLY call these verified operations):**
```
analyze_image()     → Module 1
detect_regions()    → Module 2
create_soft_mask()  → Module 2
denoise()           → Module 3
gamma_correct()     → Module 3
clahe()             → Module 3
sharpen()           → Module 3
color_correct()     → Module 3
evaluate_quality()  → Module 1
```

---

### 4.3 Technical Architecture & Components

#### A. LangGraph State Machine (`state.py`, `graph.py`)

| Graph Node | Module Called | Responsibility |
|---|---|---|
| `analyze_node` | Module 1: `analyze_image()` | Measures current image metrics. |
| `diagnose_and_plan_node` | Gemini VLM + `validate_and_sort_plan()` | Diagnoses flaws and generates a validated JSON plan. |
| `process_node` | Module 2 + Module 3 via `execute_plan()` | Resolves masks and executes image processing operations. |
| `evaluate_node` | Module 1: `evaluate_reference()` / `evaluate_no_reference()` | Quantifies quality improvements. |
| `decide_node` | Internal Logic | Decides `SHIP`, `RE_PROCESS`, or `STOP_BEST_EFFORT`. |

**Stop Conditions:**
- `SHIP`: Quality criteria satisfied or no further issues detected.
- `RE_PROCESS`: Improvement needed and remaining iterations available.
- `STOP_BEST_EFFORT`: Reached hard limit (`MAX_ITERATIONS = 3`) or detected quality degradation.

#### B. VLM Prompt Engineering (`vlm_diagnostician.py`)

| Aspect | Detail |
|---|---|
| **Model** | Gemini 1.5 Flash (cost-effective multimodal reasoning for repeated in-loop calls). |
| **Input Payload** | System Prompt (role, toolbox, order constraints) + PIL Image + `TechnicalMetrics` JSON. |
| **Output** | Strict `TreatmentPlan` JSON structure. |
| **Fallback** | Rule-based heuristic fallback if `GEMINI_API_KEY` is missing or network fails. |

#### C. Plan Validation & Priority Sorting (`planner.py`)

| Validation Rule | Rationale |
|---|---|
| Strip invalid operations | Prevents the VLM from hallucinating unsupported tools. |
| Enforce Order: **denoise (10) → gamma (20) → clahe (30) → sharpen (40) → color (50)** | **Denoising MUST precede sharpening.** Sharpening first severely amplifies high-frequency noise. |

---

### 4.4 Task Checklist

- [ ] **LangGraph State Graph (`state.py`, `graph.py`)**
  - [ ] Define `DoctorState` TypedDict (original/current images, history, metrics, iteration counter).
  - [ ] Construct LangGraph nodes: analyze → diagnose_and_plan → process → evaluate → decide.
  - [ ] Enforce bounded execution (`MAX_ITERATIONS = 3`) to prevent infinite loops.
  - [ ] Maintain full audit history per iteration (`HistoryItem`).
- [ ] **VLM Diagnostician (`vlm_diagnostician.py`)**
  - [ ] Author multimodal system prompt for Gemini (specifying toolbox and ordering constraints).
  - [ ] Implement Gemini Multimodal API invocation (PIL Image + JSON metrics).
  - [ ] Parse and validate JSON response into `TreatmentPlan` model.
  - [ ] Implement rule-based fallback mode.
- [ ] **Plan Validator & Tool Dispatcher (`planner.py`, `executor.py`)**
  - [ ] Filter out non-toolbox operations.
  - [ ] Enforce execution order (denoise before sharpen).
  - [ ] Implement `execute_plan()` dispatching actions to Module 2 and Module 3.
- [ ] **FastAPI Backend (`src/api/`)**
  - [ ] `/api/v1/health` status endpoint.
  - [ ] `/api/v1/diagnose` diagnostic analysis endpoint.
  - [ ] `/api/v1/process` full closed-loop execution endpoint.
- [ ] **Gradio Demo UI (`frontend/app.py`)**
  - [ ] Build interactive UI: image upload, max iteration slider, before/after slider, JSON log view.
- [ ] **Integration Tests (`tests/integration/test_pipeline_loop.py`)**

---

### 4.5 Critical Caveats & Gotchas

> [!CAUTION]
> **A hard iteration ceiling (`MAX_ITERATIONS = 3`) is MANDATORY.** Never write an unbounded `while not perfect:` loop. An LLM may repeatedly find minor subjective imperfections, exhausting API quotas.

> [!WARNING]
> **The VLM may propose unsupported tools** (e.g., "face_beautify", "super_resolution"). The Plan Validator must strip them before execution to prevent pipeline crashes.

> [!IMPORTANT]
> **Execution order is critical:** Always Denoise BEFORE Sharpening. Sharpening first amplifies noise variance, degrading visual quality. The Plan Validator must enforce this priority order regardless of the VLM's suggested order.

---
---

## 🚫 Deferred Items (Out of MVP Scope)

The following capabilities are located in `src/extensions/` and **MUST NOT be imported** into the MVP pipeline:

- [ ] Deblurring: Wiener deconvolution, Richardson–Lucy restoration (`src/extensions/deblur.py`)
- [ ] Inpainting / Scratch Removal: Telea, Navier–Stokes (`src/extensions/inpaint.py`)
- [ ] Super-Resolution: SRCNN, Real-ESRGAN (`src/extensions/super_res.py`)

---

## 📌 Team-Wide Coding Conventions

| Category | Standard |
|---|---|
| **Log Messages & Exceptions** | Write strictly in **English** |
| **Code Comments & Docstrings** | Write in **Vietnamese with proper diacritics** (`tiếng Việt có dấu`) |
| **PowerShell Scripts (`.ps1`)** | Save as UTF-8 with BOM |
| **Type Annotations** | Python 3.10+ syntax (`float \| None`, `tuple`, `dict`) |
| **Data Schemas** | Pydantic v2 `BaseModel` |
| **Personal Workspaces** | Place scratch files, prompts, and notes in `personal/<your_name>/` (gitignored) |
| **Shared Technical Contracts** | `docs/interfaces.md`, `docs/decisions.md` |
