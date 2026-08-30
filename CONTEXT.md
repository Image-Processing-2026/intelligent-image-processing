# CONTEXT.md — Agent Instructions for Intelligent Image Processing

## 1. Project Overview & Core Mission
**Intelligent Image Processing** (AI Image Doctor) is an **agentic, closed-loop image processing system** for an Image Processing university course.

### Core Philosophy
> **The AI is the orchestrating brain. The Image Processing Engine is the tool that actually does the work.**

- **Explicit Non-Goal:** Do **NOT** use Generative AI to directly synthesize/beautify pixels (no DALL-E, Midjourney, Stable Diffusion direct image-to-image magic).
- **Substance of the Course:** Classical/traditional algorithms (OpenCV, scikit-image) modified region-by-region form the graded substance.
- **AI Role:** Vision AI (Gemini VLM) + Orchestrator (LangGraph) handle technical diagnosis, region targeting, tool planning, evaluation, and deciding whether to ship or re-process.

### The Feedback Loop
```
Image → Analyze → Diagnose (VLM) → Locate Regions (SAM/DINO) → Plan (Agent) → Process (OpenCV) → Evaluate
                                                                                                    │
                                                                           ┌────────────────────────┴────────────────────────┐
                                                                      Quality OK?                                         Not yet
                                                                           │                                                 │
                                                                           ▼                                                 ▼
                                                                         SHIP                                  Re-Plan & Loop (Max 3 iters)
```

For comprehensive context and architectural rationale, always refer to [intelligent_image_processing_context.md](file:///intelligent_image_processing_context.md).

---

## 2. Coding Conventions & Language Rules
- **Log Messages & Exceptions:** Write strictly in **English** (e.g. `logger.info("Executing CLAHE on region 'sky' with clip_limit=2.0")`).
- **Code Comments & Docstrings:** Write in **Vietnamese with proper diacritics** (`tiếng Việt có dấu`). Explain algorithm reasoning, math, and parameter selection in Vietnamese.
- **PowerShell Scripts (`.ps1`):** Must be saved with **UTF-8 with BOM** encoding for Windows PowerShell 5.1 compatibility.
- **Type Annotations & Validation:** Use Python 3.10+ type hints (`float | None`, `tuple`, `dict`) and Pydantic v2 schemas for all cross-module data structures.

---

## 3. Team Ownership & Architecture Boundaries
The codebase is partitioned into 4 distinct ownership modules:

1. **`src/analyzer_evaluator/` (Person 1):**
   - Technical image quality analysis (histogram, brightness, contrast, noise, blur, sharpness).
   - Dual-path evaluation: Ground-truth reference metrics (`PSNR`, `SSIM` on synthetic data) vs. No-reference metrics (`BRISQUE`, `NIQE` on real user images).
2. **`src/region_engine/` (Person 2):**
   - Semantic text-prompted segmentation (GroundingDINO + MobileSAM/SAM2) & lightweight face detection (MediaPipe on CPU).
   - Soft mask generation (`float32 [0.0, 1.0]`) with edge feathering and alpha blending.
3. **`src/processing_engine/` (Person 3):**
   - Classical image processing operations: Denoise, CLAHE, Gamma, Sharpen, Color adjustment.
   - Every operation MUST conform to the unified signature: `operation(image: np.ndarray, mask: np.ndarray | None, **params) -> np.ndarray`.
4. **`src/agent/` & `src/api/` (Person 4):**
   - LangGraph state machine, Gemini VLM prompt construction, JSON plan validation, history tracking, loop bounding (`MAX_ITERATIONS = 3`), and FastAPI backend routes.

---

## 4. The Fixed Agent Toolbox
The LangGraph agent can **only** call validated functions from this fixed set. Never invent new arbitrary tools during planning:
```python
analyze_image(image) -> TechnicalMetrics
detect_regions(image, target_prompts) -> dict[str, Mask]
create_soft_mask(binary_mask, blur_radius) -> SoftMask
denoise(image, mask, method, strength, ...) -> np.ndarray
gamma_correct(image, mask, gamma) -> np.ndarray
clahe(image, mask, clip_limit, grid_size) -> np.ndarray
sharpen(image, mask, method, amount) -> np.ndarray
color_correct(image, mask, method, params) -> np.ndarray
evaluate_quality(current_img, original_img=None, is_synthetic=False) -> EvalScore
```

---

## 5. Scope Isolation (MVP vs. Deferred)
- **In MVP Scope:** Classical denoising, gamma correction, CLAHE, unsharp masking, basic color balance, semantic masks, soft-mask blending, dual-path evaluation, LangGraph loop.
- **Deferred (`src/extensions/`):** Deblurring (Wiener/Richardson-Lucy), Inpainting (Telea/Navier-Stokes), Super-Resolution (SRCNN/Real-ESRGAN). **Do NOT import from `src/extensions/` into the MVP pipeline.**

---

## 6. Key Commands
- **Run FastAPI Backend:** `uvicorn src.api.main:app --reload --port 8000`
- **Run Demo Frontend:** `python frontend/app.py`
- **Run Unit & Integration Tests:** `pytest tests/`
- **Code Linting:** `ruff check src/ tests/`

---

## 7. Personal Workspaces & AI Custom Rules (`personal/`)
- Each teammate has different development habits, prompt techniques, and AI tools (Claude Code, Cursor, Windsurf, Copilot, etc.).
- Store all individual scratchpads, custom AI prompt files, `.cursorrules`, personal planning docs, and temporary experimental test scripts inside **`personal/<your_name>/`**.
- This directory is completely ignored by Git (configured in `.gitignore`), preventing merge conflicts and keeping the shared repository clean.

