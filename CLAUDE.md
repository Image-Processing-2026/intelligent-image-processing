# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Agentic closed-loop image processing for a university Image Processing course ("AI Image Doctor").
Gemini VLM + LangGraph **diagnose and orchestrate**; classical CV (OpenCV, scikit-image, NumPy) does **all** pixel work.

**Hard constraint (ADR-001):** never use generative AI (Stable Diffusion, GANs, DALL-E, image-to-image models) to produce or modify pixels. This is a course requirement. Do not add it, even as an "optional" path.

Background docs: `CONTEXT.md` (mission, ownership), `docs/interfaces.md` (cross-module contracts), `docs/decisions.md` (ADR-001…004), `CONTRIBUTING.md` (git workflow). `personal/` holds individual team notes and is not authoritative.

## Commands

```bash
pip install -r requirements-dev.txt          # includes pytest-cov, pyiqa, ruff

# Lint/format (CI runs only `ruff check src/ tests/`)
ruff check src/ tests/ --fix && ruff format src/ tests/ && ruff check src/ tests/

# Tests (CI runs tests/unit/ only; integration tests are local)
pytest tests/unit/ -v
pytest tests/unit/test_agent_planner.py::test_name -v   # single test
pytest tests/                                            # unit + integration
pytest --cov=src/analyzer_evaluator tests/unit/          # coverage gate fail_under=90 (analyzer_evaluator only)

# Run
uvicorn src.api.main:app --reload --port 8000           # FastAPI, routes under /api/v1 (/health, /diagnose, /process, /render, /refine, /sessions, /memory)
python frontend/app.py                                  # Gradio on :7860, calls run_pipeline() directly (not via API)
```

Run from the repo root. `pythonpath = ["."]` means every import uses the `src.` prefix (`from src.agent.graph import run_pipeline`). Tests and the pipeline work without `GEMINI_API_KEY` (see VLM fallback below). The key goes in `.env` (see `.env.example`).

## Architecture: 4 owned modules

| Module | Owner | Role |
|---|---|---|
| `src/analyzer_evaluator/` | Person 1 | `analyze_image()` → `TechnicalMetrics`; local measurement + severity calibration (`measurement/`); EXIF priors (`metadata.py`); dual-path evaluation |
| `src/region_engine/` | Person 2 | Soft masks: `resolve_subject_mask` (subject masks), `segment_by_prompt`, `detect_faces` (MediaPipe), spatial/quadrant masks, `blend_regions` |
| `src/processing_engine/` | Person 3 | Classical ops: denoise, gamma, CLAHE, sharpen, color balance |
| `src/agent/` + `src/api/` | Person 4 | LangGraph loop, VLM diagnosis, plan validation, executor, FastAPI |
| `src/extensions/` | — | **DEFERRED** (deblur, inpaint, super-res). Never import it into MVP code. Ruff excludes it. |

Cross-module changes must respect `docs/interfaces.md`. Keep edits inside the module you were asked to touch. Region names are normalised by `src/agent/regions.canonical_region` (face/full synonyms); executor, planner and perception all use it.

### Pipeline (`src/agent/graph.py::run_pipeline`)

LangGraph `StateGraph` over `DoctorState` (`src/agent/state.py`):

```
analyze → perceive → clarify → diagnose_and_plan → process → evaluate → decide ─┬─ SHIP / STOP_BEST_EFFORT → END
   ↑                                                                  │
   └──────────────────────────── RE_PROCESS ──────────────────────────┘   (max_iterations, default 3)
```

- `analyze` calls `analyze_image()`. Its `*_level` fields and `defect_severity` come from the calibrated measurement (Analysis v2 D4): `measurement.measure_image()` measures per 64 px tile (exposure L\*/clipping, Liu σ noise, defocus σ/edge width/motion, neutral-pixel illuminant, blockiness, dark channel), and `measurement.calibration.severity_from_measurement()` maps a measurement to severity 0–3 using thresholds fitted on the synthetic set (`calibration.yaml`, `calibration_version()`). Never add hand-tuned thresholds elsewhere; refit with `scripts/calibrate_severity.py`.
- `perceive` (`perception.perceive`, Perceive v2) is stage 1. It measures the image, sends Gemini the full frame plus up to 6 evidence crops at 100 % (`evidence.select_evidence`), overlays, the measurement summary and EXIF priors (`metadata_priors`), and gets a `DiagnosisReport` (scene, subjects with `box_2d`, `defects` with severity 0–3, `preserve` = intentional style). It drops defects that conflict with `preserve`, builds subject masks with `region_engine.resolve_subject_mask` (the same masks the executor uses later, cached by `cache_key`), stores per-region `region_stats` (`measurement.aggregate`) and `region_metrics`, then runs fusion. `PERCEIVE_SAMPLES` (default 1, env `PERCEIVE_SAMPLES`) Gemini samples are merged by self-consistency; hypotheses below quorum go to `rejected`.
- Fusion (`fusion.fuse`, Analysis v2 P6) is a deterministic decision table over VLM hypotheses × calibrated measurement × `preserve` × EXIF priors × intent. Measured and reported by the VLM → confirmed, severity from the measurement (VLM moves it ±1); a colour cast measured through the illuminant fallback (`color.illuminant_method` ≥ 1, `Measured.low_trust`) is verified instead and keeps the VLM severity. Measured but preserved → preserved (severity 3 on a subject → `ask_user`). Measured but not mentioned, or reported but not measured → one closed-question Gemini verify call per image (`defect|intentional|texture|not_present`, full frame + crops); `not_present`/`texture` → `rejected`. A measured defect is never emitted on its own, and a failed verify keeps a VLM claim at half confidence. Measured `backlit_subject` compares the subject (face first) with `background` in L\*. Each decision is logged in `fusion_notes`. `perception.FUSION_ENABLED` / `VERIFY_ENABLED` switch it off; `tests/conftest.py` disables verify for every test (old tests mock an exact number of Gemini responses), and fusion tests turn it back on.
- `motion_blur`, `jpeg_artifacts`, `haze`, `highlight_clipping` (D2) are diagnosed with `fixable=False`, only from severity 2. `actionable_defects` excludes them, so an image with only these ships an empty plan. They add contraindications (`no_sharpen`, `no_brighten`) and advisory cards (`knowledge/cards/limits.yaml`, `auto_apply: false`, no recipe).
- From iteration 2 Perceive re-measures and compares with the original (`measurement.diff.compare` → `DiagnosisReport.diff`: per-defect resolved/improved/unchanged/worse plus side effects). It calls the VLM again only when a semantic defect is still open or a side effect needs judging.
- `diagnose_and_plan` calls `graph.plan_treatment()` (stage 2). If no defect has severity ≥ 1, it returns an empty plan without calling the VLM. Otherwise it runs `diagnose_and_plan()` → `validate_and_sort_plan()` → `planner.apply_preserve_guard()`, which deterministically neutralises or drops actions that break `preserve`, → `planner.apply_contraindications()` (`no_sharpen` drops every sharpen; `no_brighten` drops full-image `gamma_correct` with gamma > 1).
- `process` calls `executor.execute_plan()`. It builds a mask with the region engine and then dispatches to processing-engine functions. It also records one `ExecutedAction` per action (`DoctorState.executed_actions`, copied to `HistoryItem.executed`): requested vs actual face mode, backend, feather, `strength_scale`, `mask_hash`. Face exposure ops (`gamma_correct`, `clahe`) default to `face_mode=oval` when the plan sets none (checked with `model_fields_set`; an explicit `bbox` is respected). An oval/SAM failure or a failed mask QA (`face_mask_qa`) falls back to bbox at reduced strength (`FALLBACK_STRENGTH_SCALE` = 0.5, temporary, uncalibrated BL-10), never at full strength.
- `evaluate_node` does its own ADR-002 routing: reference eval only when `is_synthetic` and a ground truth exists. It must store `eval_metrics.model_dump()` (a plain dict), never the Pydantic model. The model breaks `HistoryItem` and the JSON API response.
- `decide_node`, in order: degraded → `STOP_BEST_EFFORT` + rollback (on real images, a processing side effect of severity ≥ `SIDE_EFFECT_ROLLBACK_SEVERITY` (2) from `measurement.diff.side_effects` — halo, dark_noise, banding, channel_clipping, oversaturation — also counts as degraded; stored in `HistoryItem.side_effects`); empty plan → `SHIP`; `_decide_synthetic` (PSNR ≥ 28 and SSIM ≥ 0.88) or `_decide_real` (score ≥ `REAL_TARGET_SCORE` → `SHIP`, gain < `REAL_MIN_GAIN` → `STOP_BEST_EFFORT`). For real images a score-based `SHIP` becomes `RE_PROCESS` when the plan acted on a defect with severity ≥ `VERIFY_SEVERITY` (on its region or on the full image) and iterations remain, so the next Perceive can verify the fix. Only after that does the iteration cap turn a remaining `RE_PROCESS` into `STOP_BEST_EFFORT`.
- On real images `decide_node` also asks the reviewer (`_judge_iteration` → `review_candidate`, once per iteration with actions and changed pixels): `ACCEPT` → `SHIP` and clears a soft-rule regression (`quality_improved`/score; measured `worse` defects and side-effect rollbacks are not overridden), `REVISE` → `RE_PROCESS` (the iteration cap still applies), `REJECT` → rollback. `UNCERTAIN`, no key or a failed call fall back to the rules above. The verdict is stored in `HistoryItem.review`/`DoctorState.review`, and `review_node` reuses it when it saw the very image being returned.
- After `decide_node` ends the loop, `review.review_candidate` (Gemini reads original, previous image and candidate with paired crops; `review.REVIEW_ENABLED`, off in tests) and `review.derive_outcome` set `DoctorState.outcome` (`ACCEPTED`, `UNVERIFIED`, `NOT_MET`, `KEPT_ORIGINAL`, `NO_CHANGE`) separate from `decision`. A real image is `ACCEPTED` only when the review accepts the returned image; no key or a failed call gives `UNVERIFIED`. Temporary thresholds are listed in `docs/reports/2026-10-28-audit-fix-release-note.md` (BL-10).
- Real-image tracking (`graph._track_defects`) measures each targeted defect on its own region against the original; an action carries `RegionOperation.defect_ref` (`type@region`) so a face action still tracks the person's `backlit_subject`. The dark-region card's `face_when_person` step targets `face` (`face_mode=oval`) when Perceive located a face. A person defect that measures `resolved` while the face did not brighten becomes `unverified` (`DefectChange.cause`) and `_decide_tracked` returns `STOP_BEST_EFFORT`. `_side_effects_check` returns a `SideEffectCheck` (previous→candidate and original→candidate; `unavailable` on a measurement error, never an empty list). The `black_lift` side effect (dark areas outside the face lifted to grey) has severity ≤ 1 and never rolls back. New thresholds here are temporary, uncalibrated (BL-10).
- `clarify` (Phase 4, `intent.py`) runs only in interactive sessions (`session.py`, with a `MemorySaver` checkpointer). On iteration 1 it calls `interrupt()` with one deterministic question (`approach`: follow the AI recommendation, built from the diagnosis, or do it another way described in the chat; no question when there is no fixable defect). On resume the node re-runs from the top, so everything before `interrupt()` must stay deterministic. The resulting `IntentProfile` overrides every iteration's diagnosis (`apply_intent` in `perceive_node`) and adds an "Ý ĐỊNH NGƯỜI DÙNG" block to the Plan prompt. `vlm_diagnostician` receives that block as text (`intent_text`), because importing `intent` there would create an import cycle through `perception`.
- Feedback (`refine.py`) is stateless: Vietnamese rules, then Gemini, then edits to the treatment, then a re-render from the original. It bypasses the preserve guard because the user asked explicitly.
- Case memory (`memory.py`, Phase 5): a SQLite store of past runs, choices and feedback, with no images. It is disabled unless `configure_case_memory()` is called (the API and the Gradio `__main__` do so). `tests/conftest.py` disables it for every test; a test that needs it configures a `tmp_path` DB. `diagnose_and_plan_node` adds similar cases to the iteration-1 Plan prompt, and `rank_variants_node` recommends the user's preferred style when the intent gives none. Record runs at the API/UI layer (`remember_run`), not inside the graph.
- Variants (`variants.py`): with `num_variants > 1`, `should_continue` fans out one `Send` per style (balanced/natural/vivid) to `render_variant`, which renders the treatment of the non-rolled-back iterations on a ≤1024 px preview. `rank_variants` then deduplicates and ranks them, using the VLM critic or the Module 1 score. Render results accumulate in `variant_candidates` (an `operator.add` reducer); the ranked list goes in `variants`. Never write ranked results back into the reducer key. Re-render a chosen variant with `planner.sanitize_actions`, never `validate_and_sort_plan`, because reordering changes the result.

### VLM behaviour (`src/agent/vlm_diagnostician.py`)

- SDK is `google-genai` (`genai.Client(...).models.generate_content`) with `response_json_schema=PLAN_RESPONSE_SCHEMA`, generated from the planner bounds. Model defaults to `DEFAULT_GEMINI_MODEL`, overridable via `GEMINI_MODEL`. Images are sent at full resolution (PNG, or JPEG q95 when over the `VLM_INLINE_IMAGE_BUDGET` inline budget) with `media_resolution=HIGH`; Gemini still resamples them to its own token budget. From iteration 2 the original image is sent too.
- Knowledge Base (`src/agent/knowledge/`, see its `README.md`): YAML playbook cards (symptoms → recipe using the 5 tools → contraindications) and markdown principles. Retrieval is structured filtering plus BM25, with no embeddings. `format_context()` injects the matched cards into the Plan prompt, with parameters already resolved for the case's severity. `TreatmentPlan.knowledge` records the ids used.
- No `GEMINI_API_KEY` → rule-based plan (`TreatmentPlan.source == "rule_based"`), produced by the KB rule engine from the `auto_apply` cards. All tests depend on this offline path, so keep it working.
- Key set but the call, parse or validation fails → the same rule-based plan with `source == "vlm_fallback"`. Only a valid VLM response with `actions: []` yields an empty plan (= `SHIP`).
- Perceive uses its own `PERCEIVE_PROMPT` and schema (`perception.py`). Subject boxes are `box_2d` in Gemini's convention (ymin, xmin, ymax, xmax) on 0..1000. They are kept in `DiagnosisReport.subject_boxes` and never sent to the Plan prompt. `PERCEIVE_FULL_FRAME_MAX_SIDE` optionally downscales the full frame; crops stay at 100 %.
- `.env` is saved with a BOM, so `load_dotenv()` misses `GEMINI_API_KEY`; scripts read it with `encoding="utf-8-sig"`.
- `SYSTEM_PROMPT` is in Vietnamese and lists the toolbox and parameter ranges. Keep it in sync with `planner.py` (gamma hallucination was a past bug).
- Mock Gemini in tests with `patch.dict("os.environ", {"GEMINI_API_KEY": "fake-..."})` plus patching `src.agent.vlm_diagnostician.genai` and setting `mock_genai.Client.return_value.models`. Both stages call Gemini through `call_gemini_json`, so a pipeline test needs perceive and plan responses (`_route_gemini` in `tests/unit/test_vlm_mock.py` routes them by schema). See `tests/unit/test_vlm_mock.py`). Never hit the real API in unit tests.

## Contracts that are easy to break

**Agent toolbox is fixed at 5 operations** (`planner.ALLOWED_OPERATIONS`): `denoise`, `gamma_correct`, `clahe`, `sharpen`, `color_correct`. Do not invent new operations. Unknown operations are silently dropped. Toolbox names differ from function names: `gamma_correct → apply_gamma` and `color_correct → apply_color_balance`. Changing an operation or its parameters means updating all of these together:
1. `planner.PARAMETER_BOUNDS` (clamping by `clamp_parameters`) and the `priority_map` in `validate_and_sort_plan`. Order is denoise(10) → gamma(20) → clahe(30) → sharpen(40) → color(50), and the planner overwrites any VLM-supplied order.
2. The dispatch branch and defaults in `executor.execute_plan`.
3. The VLM `SYSTEM_PROMPT`. The structured-output schema is generated from `PARAMETER_BOUNDS`.
4. The playbook cards in `src/agent/knowledge/cards/*.yaml`. They are validated against `ALLOWED_OPERATIONS` and `PARAMETER_BOUNDS` when loaded, so a stale card fails `test_agent_knowledge.py`.
5. `docs/interfaces.md`.

**Processing engine signature:** `apply_<op>(image, mask: Optional[np.ndarray] = None, **params) -> np.ndarray`. Implement the pure algorithm as `_raw_<op>(image, **params)`, then wrap it with `base.apply_region_op(image, mask, _raw_<op>, ...)`, which handles soft-mask blending. Do not blend by hand inside an operation.

**Images:** inter-module format is RGB `uint8` `(H, W, 3)`. Grayscale input must also work (`test_grayscale_support.py`). Convert back to RGB after any OpenCV BGR work. RGB/BGR swaps have caused real bugs here.

**Masks (ADR-003):** `np.float32` in `[0.0, 1.0]`, shape `(H, W)` or `(H, W, 1)` matching the image, Gaussian-feathered (`mask_utils.create_soft_mask`). Never use hard binary masks for blending. `executor._validate_mask` rejects anything else (wrong dtype, all zeros, NaN, out of range) and **silently falls back to full-image processing**.

**Executor swallows errors:** each action runs in its own `try/except` that logs a warning and skips the action. A broken operation will not fail the pipeline. Assert on output or logs in tests, not on exceptions.

**Subject masks:** `resolve_subject_mask(image, name, box_2d)` chains face → MediaPipe; with the models (`REGION_DINO_MODEL_PATH`, `REGION_MOBILE_SAM_CHECKPOINT`, see `scripts/fetch_segmentation_models.py`) → GroundingDINO matched to the Gemini box → MobileSAM; without models → GrabCut on the Gemini box; no box → best DINO detection (confidence ≤ 0.5) → quadrant keyword → full. `subject_capabilities()` reports what is available without loading models. The executor uses this resolver for semantic regions that have a Perceive `box_2d` (`DiagnosisReport.subject_boxes`); other regions go through the controller (face/full/quadrant keywords, including Vietnamese, and semantic prompts). `target_prompt` values `face`/`khuôn mặt` go to MediaPipe. A face action with no detected face is skipped. `resolve_subject_mask` takes `feather_radius` (the action's feather; part of the cache key). Mask QA failing counts as a failure, not a pass.

**Dual-path evaluation (ADR-002):** PSNR/SSIM/MSE (`evaluate_reference`) only for synthetic images with ground truth. Real images use `evaluate_no_reference` (heuristic score + optional `pyiqa` BRISQUE/NIQE + deltas vs. the previous iteration). `analyzer_evaluator.evaluate_quality()` is the unified router. `graph.evaluate_node` duplicates that routing, so keep both consistent.

**Dependencies:** `numpy<2.0.0` is pinned for compatibility, so do not bump it. Use Pydantic v2 APIs (`model_dump()`, not `.dict()`). ADR-004: CPU-only target (AMD iGPU, no CUDA), so do not add CUDA-only dependencies.

## Conventions

- **Bilingual:** docstrings and code comments in **Vietnamese with diacritics**; log messages, exception messages, identifiers and commit messages in **English**.
- Full type annotations on every function. Ruff: line length 100, rules `E,F,I,W` (import sorting enforced, `E501` ignored).
- PowerShell `.ps1` scripts must be saved as UTF-8 **with BOM** (Windows PowerShell 5.1).
- Tests: unit tests in `tests/unit/`, shared synthetic-image fixtures in `tests/conftest.py` and `tests/fixtures/synthetic_images.py`. Sample real images are in `data/real/`.

## Git workflow

- Branch from `develop`: `feat/`, `fix/`, `refactor/`, `docs/`, `test/<name>`. PR into `develop` (only release PRs go to `main`). Never commit directly to `main`.
- Conventional Commits: `<type>(<scope>): <description>`, with scopes `agent`, `engine`, `region`, `eval`, `api`, `ui`, `workflow`.
- CI (`.github/workflows/ci.yml`, Python 3.10): `ruff check src/ tests/` → `pytest tests/unit/ -v`. It must be green, and the PR needs one reviewer approval.

## Definition of done

- [ ] `ruff check src/ tests/` clean and `ruff format` applied
- [ ] `pytest tests/unit/ -v` passes. New behaviour has unit tests that run offline (no API key, no network)
- [ ] Type annotations on new functions; Vietnamese docstrings; English logs and exceptions
- [ ] No `src/extensions/` imports and no generative pixel manipulation
- [ ] Masks are float32 [0, 1] and feathered; operations use the `apply_<op>` + `apply_region_op` pattern
- [ ] Toolbox changes are synced across planner, executor, VLM prompt and `docs/interfaces.md`
