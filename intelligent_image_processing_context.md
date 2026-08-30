# AI Image Doctor — Project Context

**Course:** Image Processing (Xử Lý Ảnh)
**Team size:** 4
**Status:** Pre-implementation — architecture and stack decided, MVP scope locked

This document is the single source of truth for the project. It should be loaded in full by any AI coding agent before it plans, scaffolds, or writes code for this repository.

---

## 1. What this project is

AI Image Doctor is an **agentic, closed-loop image processing system**. It does not simply transform an image on request ("upload → prettify"). Instead it behaves like a diagnostic system: it analyzes the technical condition of an image, identifies *what* is wrong and *where*, plans a sequence of image-processing operations to fix it, executes that plan region by region, evaluates the result, and decides whether to ship the output or re-plan and try again — up to a fixed iteration limit.

Core philosophy:

> **The AI is the orchestrating brain. The Image Processing Engine is the tool that actually does the work.**

This distinction matters because the project is for an Image Processing course, not a generative-AI course. The classical/traditional image-processing techniques (denoising, CLAHE, gamma correction, sharpening, inpainting, etc.) are the graded substance of the work. AI (a VLM and an orchestration agent) is only responsible for perception, reasoning, planning, and decision-making — never for directly hallucinating pixels as a shortcut around the actual processing.

### The loop

```
Image → Analyze → Diagnose → Identify Regions → Plan → Process → Evaluate
                                                                       │
                                                          ┌────────────┴────────────┐
                                                       Good enough?              Not yet
                                                          │                          │
                                                          ▼                          ▼
                                                        Ship                    Re-plan → Process → Evaluate (loop, bounded)
```

---

## 2. Goals

1. Analyze the technical condition of an input image (brightness, contrast, noise, blur, sharpness, histogram, color).
2. Identify the specific problems present.
3. Localize *which regions* of the image contain each problem (not just "the image has noise" but "the background has noise").
4. Propose a processing pipeline appropriate to the diagnosis.
5. Execute the corresponding image-processing operations.
6. Support **region-aware processing** — operations applied selectively, not uniformly across the whole image.
7. Evaluate the quality of the result after each iteration.
8. Automatically decide `SHIP` or `RE-PROCESS`.
9. Bound the loop with a maximum iteration count to avoid infinite processing.
10. Support before/after comparison and an explainable processing history.

## 3. Explicit non-goal

Do **not** build `Image → Generative AI → Beautiful Image`. That approach defeats the purpose of an Image Processing course submission. The correct shape is:

```
Image → AI Analysis → AI Planning → Traditional/AI Image Processing → Evaluation → AI Decision
```

AI touches perception, planning, and decision-making. The actual pixel manipulation is done by classical (or well-understood, explainable) image-processing algorithms, implemented and understood by the team — not hidden behind an opaque generative call.

---

## 4. High-level architecture

```
USER → Image Upload
          │
          ▼
  IMAGE ANALYZER            (brightness, contrast, noise, blur, sharpness, histogram, color)
          │
          ▼
  VISION AI / VLM           (visual understanding + diagnosis in natural language)
          │
          ▼
  REGION ENGINE             (detection, segmentation, mask generation)
          │
          ▼
  AI AGENT                  (reasoning, planning, tool selection, decision-making)
          │
          ▼
  IMAGE PROCESSING ENGINE   (denoise, deblur, gamma, CLAHE, sharpen, color correct, inpaint, super-res)
          │
          ▼
  EVALUATOR                 (PSNR, SSIM, no-reference quality, visual/semantic check)
          │
          ▼
  AGENT DECISION → SHIP  or  RE-PROCESS (loop back to AI AGENT, bounded by MAX_ITERATIONS)
```

Conceptual roles (useful for the presentation/report):

| Component | Role | Analogy |
|---|---|---|
| AI Agent | Reasoning, planning, decision | Brain |
| Vision AI / Region Engine | Understanding the image, locating regions | Eyes |
| Image Processing Engine | Actually modifying pixels | Hands |
| Evaluator | Judging whether the result is acceptable | Quality control |

---

## 5. Modules

### Module 1 — Image Analyzer
Computes objective technical metrics: brightness (too dark/bright/normal), contrast (low/normal/high), noise (Gaussian-like vs. salt-and-pepper vs. general level), blur (motion vs. out-of-focus vs. general sharpness loss), sharpness, histogram distribution, and color (cast, saturation, white balance, fading).

### Module 2 — Vision AI / VLM
Provides: (a) visual understanding of image content, (b) natural-language quality diagnosis (e.g. "the background is noisy and the face is underexposed"), (c) image-processing domain knowledge (mapping a problem → cause → suitable algorithm → expected effect), and (d) reasoning about *processing order* (e.g. denoise before sharpen, never the reverse, since sharpening first amplifies noise).

### Module 3 — Region Engine (detection / segmentation)
Not every operation should apply to the whole image (e.g. sky is overexposed, face is underexposed, background is noisy — each needs a different, spatially-limited fix). Supports:
- **Spatial** regions (top-left, center, arbitrary area)
- **Semantic** regions (face, person, sky, background, vegetation, building, text)
- **Problem-based** regions (noisy / blurred / overexposed / underexposed / damaged / scratched region)
- **User-defined** regions (brush, rectangle, polygon, mask)

Produces masks — binary (`0` = skip, `1` = process) or **soft masks** (`0.0`–`1.0` continuous), the latter needed to blend region boundaries smoothly and avoid seam artifacts.

**Separation of concerns:** the VLM does not need to produce pixel-perfect masks itself. It answers "what should be fixed and where, semantically" (e.g. "the sky needs exposure correction"); a dedicated segmentation model turns that semantic target into an actual pixel mask.

### Module 4 — AI Agent
The orchestrator. Responsibilities: understand (input, analysis, regions, intent), diagnose (region + problem + severity), plan (region → operation → parameters → order), execute (call tools), evaluate (read Evaluator output), decide (`SHIP` / `RE-PROCESS`).

The agent must **only** call functions from a fixed toolbox — it cannot invent arbitrary operations. This keeps the system predictable, debuggable, and explainable in the report:

```
analyze_image()
detect_regions()
create_mask()
denoise() / deblur() / gamma_correct() / clahe() / sharpen() / color_correct() / inpaint()
evaluate_quality()
compare_versions()
```

The agent outputs a **structured plan**, not a single natural-language sentence, e.g.:

```json
{
  "regions": [
    { "target": "sky",        "issue": "overexposure",  "operation": "exposure_correction", "strength": 0.6 },
    { "target": "face",       "issue": "underexposure", "operation": "local_brightness",     "strength": 0.4 },
    { "target": "background", "issue": "noise",         "operation": "denoise",              "strength": 0.7 }
  ]
}
```

The agent must retain **processing history** across iterations (what was tried, what changed, what degraded), so it doesn't repeat a failed operation.

### Module 5 — Image Processing Engine
The graded core of the project. Classical/well-understood algorithms:
- **Denoising:** mean / median / Gaussian / bilateral / non-local means filters
- **Deblurring** *(optional extension)*: Wiener filter, Richardson–Lucy
- **Brightness:** gamma correction, intensity transformation
- **Contrast:** histogram stretching, histogram equalization, CLAHE
- **Sharpening:** Laplacian, unsharp masking
- **Color:** white balance, HSV-based adjustment, color correction
- **Inpainting** *(optional extension)*: Telea, Navier–Stokes
- **Super resolution** *(optional extension)*: SRCNN, Real-ESRGAN or similar

Every operation must support the region-aware signature `operation(image, mask, parameters)`, not just `operation(image)`. Difficulty varies: brightness/gamma/simple color adjustments are easy to apply per-region; sharpening/CLAHE are moderate; kernel-based filters (Gaussian/median) and frequency-domain operations are hardest near mask boundaries and need careful blending to avoid artifacts.

### Module 6 — Evaluator
**This module has a correctness trap the team must design around explicitly — see §7.**

Objective metrics: PSNR, SSIM, MSE, noise level, sharpness, contrast, brightness, histogram distribution. Visual/semantic evaluation via the VLM: unnatural artifacts, oversharpening, loss of detail, color problems, general visual plausibility. Higher PSNR is not automatically "better" — for enhancement/restoration tasks, perceptual/visual quality can matter more than pixel-level similarity to some reference.

---

## 6. Agent loop and stop conditions

```
Analyze → Plan → Process → Evaluate → Decision
                                          │
                     ┌────────────────────┴────────────────────┐
                  acceptable                                 not acceptable
                     │                                           │
                     ▼                                           ▼
                   SHIP                              RE-PROCESS (adjust plan, loop)
```

- `MAX_ITERATIONS` must be a hard cap (e.g. 3).
- `SHIP` when quality score ≥ threshold, no major artifacts, and task requirements are satisfied.
- `RE-PROCESS` when quality is below threshold *and* further improvement looks plausible given history.
- `STOP / BEST-EFFORT` when the iteration cap is hit, or there's no meaningful improvement between iterations, or the last change caused degradation.
- Never implement an unbounded `while image_not_perfect: process()` loop.

---

## 7. Architecture decisions made for this implementation

These decisions were made after reviewing the original spec and should be treated as binding unless the team explicitly revisits them.

### 7.1 Evaluator: reference vs. no-reference metrics (fixes a conceptual gap in the original spec)
PSNR/SSIM require a ground-truth reference image. For real user-uploaded photos (an old damaged photo, a blurry snapshot), **no such ground truth exists**, so PSNR/SSIM cannot legitimately be applied there. The evaluator must branch:
- **Synthetic test set** (clean images the team artificially degrades in a controlled way): ground truth exists → use PSNR/SSIM to benchmark individual algorithms.
- **Real user-submitted images**: no ground truth → use **no-reference quality metrics** (BRISQUE, NIQE, or NIMA via the `pyiqa` package) and/or iteration-over-iteration self-comparison, never a comparison against a non-existent "original."

This distinction must be stated explicitly in the report and reflected in the code (two separate evaluation code paths, not one that silently assumes a reference exists).

### 7.2 Region-boundary artifacts
Kernel-based filters (Gaussian, median, CLAHE) applied through a hard binary mask will produce visible seams at region boundaries. Use **soft masks** (continuous 0.0–1.0 values, feathered near edges) and alpha-blend the processed region back into the image, rather than hard-compositing a binary mask. This is real engineering work and should be owned by one person, not left as a "fix later if time permits" item.

### 7.3 Technology stack

| Layer | Choice | Rationale |
|---|---|---|
| Backend | **FastAPI** (Python) | Required in practice — the CV ecosystem (OpenCV, scikit-image, PyTorch) is Python-first |
| Agent orchestration | **LangGraph** | The Analyze→Plan→Process→Evaluate→Decide loop maps directly onto a LangGraph state graph; processing history is a natural fit for LangGraph state |
| Vision AI / VLM | **Gemini** (multimodal) | Cost-effective for repeated in-loop calls; multimodal reasoning is sufficient for diagnosis and planning |
| Region Engine | **GroundingDINO + MobileSAM/SAM2** for open-vocabulary text-prompted segmentation; **MediaPipe** for face detection | Implements the exact separation in §5/Module 3: the VLM names a semantic target in text (e.g. "sky"), GroundingDINO+SAM turns that into a pixel mask. MediaPipe is lightweight and CPU-friendly for faces specifically |
| Image Processing Engine | **OpenCV + scikit-image** | Classical algorithms — the graded core of the course project; implement operations explicitly rather than hiding them behind a black-box call |
| Evaluator | **scikit-image** (PSNR/SSIM) + **pyiqa** (BRISQUE/NIQE, no-reference) | See §7.1 |
| Frontend | **Next.js** if the team wants a polished demo UI; **Gradio/Streamlit** if grading doesn't weight UI heavily and time is tight | Trade development time against demo polish |

**Hardware note:** target dev machine uses an AMD Radeon iGPU (no CUDA). SAM/GroundingDINO/YOLO-family models default to expecting CUDA; on this hardware they must run on CPU or via lightweight variants (MobileSAM, YOLOv8-nano). Since this system processes single images (not real-time video), a few seconds of latency per image on CPU is acceptable — no real-time optimization is required.

### 7.4 Team ownership (4 people, aligned to the toolbox contract in §5/Module 4)

1. **Image Analyzer + Evaluator** — shared metric code (histogram, sharpness, PSNR/SSIM, no-reference quality).
2. **Region Engine** — GroundingDINO+SAM pipeline, MediaPipe faces, soft-mask/feathering logic (§7.2). Hardest module technically.
3. **Image Processing Engine** — classical algorithms plus the region-aware `operation(image, mask, params)` wrapper for each.
4. **AI Agent + Vision AI/VLM + backend integration** — LangGraph orchestration, Gemini diagnosis calls, structured plan generation/validation, processing history, stop conditions, and the FastAPI service that ties every module together.

### 7.5 Coding conventions for this repo
- Log messages: English.
- Code comments: Vietnamese, with correct diacritics.
- Any PowerShell (`.ps1`) scripts: save as UTF-8 with BOM for Windows PowerShell 5.1 compatibility.

---

## 8. MVP scope (build this first, and only this, until it works end-to-end)

- **Analysis:** histogram, brightness, contrast, noise, blur, sharpness.
- **Region processing:** basic semantic regions, masks (including soft masks), user-selected regions.
- **Processing:** denoising, gamma correction, CLAHE, sharpening, basic color correction.
- **AI:** image understanding, diagnosis, processing planning, parameter recommendation.
- **Agent:** process → evaluate → re-process → ship loop with bounded iterations.
- **Evaluation:** PSNR/SSIM (synthetic test set only, per §7.1), no-reference metrics (real images), visual assessment.

## 9. Deferred — do not start until the MVP ships end-to-end

Deblurring, inpainting/old-photo restoration, scratch removal, super-resolution, face enhancement, AI beautification, photography/aesthetic-intent reasoning, user-defined masks beyond basic shapes, multiple enhancement profiles.

---

## 10. Positioning statement (for the report/presentation)

> AI Image Doctor is a system that automatically diagnoses the technical condition of an image, identifies the regions that need intervention, plans a treatment using image-processing techniques, evaluates the result, and adjusts the process — repeating until a quality bar is met or an iteration limit is reached.

Framed as four questions the pipeline answers in order: **What** is wrong (AI diagnosis) → **Where** is it wrong (vision/segmentation) → **How** to fix it (agent planning) → **Execute** (image processing engine) → **Is it good?** (evaluator) → **Ship or retry?** (agent decision).
