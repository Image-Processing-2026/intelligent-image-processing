# Bilingual documentation delivery plan

[English](bilingual-documentation.md) | [Tiếng Việt](bilingual-documentation.vi.md)

Status: planned. Date: 2026-10-30. This file defines the work; it does not certify that the documentation or release is complete.

## Outcome and scope

Deliver a coherent English-first documentation set for AI Image Doctor, with equivalent Vietnamese editions. Readers must be able to run the application, understand the implemented pipeline and classical algorithms, reproduce available evaluation evidence, and integrate with the API.

Finish implementation by committing, pushing, and merging the documentation into `main` through the repository PR workflow. The user has requested this delivery step; it is part of completion, not an optional follow-up. This planning task itself does not start the full documentation implementation or release.

## Language and file conventions

- English is the default at existing canonical paths: `README.md`, `docs/architecture.md`, etc. Vietnamese uses the same basename with `.vi.md`: `README.vi.md`, `docs/architecture.vi.md`.
- Put reciprocal English / Tiếng Việt links at the top of every paired page. Keep internal navigation in the selected language wherever a translation exists.
- Both editions must contain the same technical facts, scope, examples, limitations, commands, and evidence. Write natural Vietnamese rather than literal translations.
- Keep code identifiers, endpoint paths, environment variables, commands, and units unchanged. Translate explanations, tables, captions, and diagram labels.
- Update both editions in the same PR whenever behavior changes. Record the verified source commit and evidence date where claims depend on implementation or measurements.
- Preserve existing paths and inbound links. Do not rename historical reports merely to fit the convention.
- Historical plans, audits, experiment logs, and internal agent instructions are outside translation scope. Identify them as historical/reference material in the index. Runtime prompts and knowledge-base cards are also outside scope; translation must not change program behavior.

## Deliverables

Every new or updated reader-facing page below has an English canonical file and a `.vi.md` companion.

| Files (English paths) | Required contents | Primary evidence to inspect |
|---|---|---|
| `README.md`, `docs/README.md` | Project purpose, classical-pixel-processing constraint, quickstart, navigation by audience, current versus historical documents | Existing README, dependency files, application entry points |
| `docs/architecture.md` | System components, four module boundaries, API/UI/data flow, actual graph states and stop conditions, fallback, persistence, clarification/refinement lifecycle | `src/agent/graph.py`, state/session modules, `src/api/`, `web/` |
| `docs/algorithms.md` | Measurement and diagnosis boundary; denoise, gamma, CLAHE, sharpen, color correction; masks and blending; formulas, parameter meaning, ordering, limitations, before/after examples | Processing and measurement implementations, region engine, planner bounds, tests |
| `docs/evaluation.md` | Software tests versus image-quality assessment; datasets and provenance; reference/no-reference metrics actually implemented; baselines, reproducible commands, available results, failure cases and limitations | `data/`, evaluation/benchmark scripts, evaluator code, existing reports |
| `docs/user-guide.md` | Upload, analysis, clarification where available, processing, comparison, refinement and export; current screenshots, explanation of scores, a short demo script | Current React application and workspace API behavior |
| `docs/api.md` | Authentication, session lifecycle, working request/response examples, errors, upload constraints, export, storage, current versus legacy routes | FastAPI route registration, schemas, auth/configuration, API tests |
| `docs/interfaces.md` | Verified image/mask contracts, public cross-module schemas and invariants; link to source for exhaustive field definitions instead of duplicating all code | Current schemas, executor, module boundaries and contract tests |
| `web/README.md`, `frontend/README.md` | Web setup/configuration and legacy Gradio setup respectively; clearly distinguish the supported entry points | Build scripts, environment example, server/static serving and frontend entry points |

Keep `CONTRIBUTING.md`, `docs/decisions.md`, module READMEs, and existing detailed benchmark reports as linked references in this delivery. Mark untranslated references explicitly; translation of every historical or specialist document is not a completion requirement.

## Execution order

### 1. Establish the documentation baseline

- Inspect repository instructions, working-tree state, upstream branches and existing PRs before editing. Preserve unrelated user changes.
- Inventory current documentation as authoritative, partially stale, historical, or reference-only. Create a coverage table in the documentation index.
- Verify behavior against source and tests. Resolve differences between old diagrams and current implementation before describing the pipeline.
- Define a shared EN/VI terminology table in the index, including diagnosis, treatment plan, mask, refinement, fallback and evaluation.

Acceptance: every deliverable has an identified source, reader, and verification method; no unsupported feature is presented as implemented.

### 2. Write the architecture and algorithm foundation

- Draft architecture and algorithms in English, then produce their Vietnamese companions before moving on.
- Add Mermaid component/sequence/state diagrams where useful. Match nodes and transitions to the implementation.
- Trace a single backlit-portrait example across measurement, diagnosis, mask selection, validated operations, blending and evaluation. Verify any numerical parameters from the actual run.
- Explain AI orchestration separately from classical pixel operations. Identify deferred extensions and unfixable defects accurately.
- Reconcile `docs/interfaces.md` with current contracts and create its Vietnamese edition.

Acceptance: diagrams, parameter direction/ranges, mask semantics and stopping/fallback behavior match source; both languages explain the same behavior.

### 3. Document reproducible evidence

- Inventory available datasets and existing benchmark outputs before deciding whether new runs are needed.
- Record dataset/split, sample count, preprocessing, command, dependency/model settings, hardware where relevant, source revision and metric interpretation for each reported experiment.
- Compare only compatible conditions. Keep synthetic reference evaluation distinct from real-image no-reference evaluation and human ratings.
- Reuse existing results only when provenance is adequate. Mark missing evidence as not measured; never invent scores or claim that passing unit tests proves image quality.
- Include representative successes and failures with original/processed images, mask overlays where useful, and captions in both languages.
- Store shareable figures under `docs/assets/`; exclude private uploads, credentials, large model files and unlicensed sample redistribution.

Acceptance: each numerical claim is traceable to an artifact or reproducible run; limitations are explicit and equivalent in both editions.

### 4. Write usage and integration guides

- Verify quickstart from dependency installation through startup, including the no-key path. Document optional semantic-model setup separately.
- Exercise the current user journey and capture screenshots from that version. Avoid documenting planned UI as existing behavior.
- Verify API examples against routes and schemas; prefer linking to generated OpenAPI for exhaustive endpoint definitions after confirming its location.
- Cover authentication and current workspace endpoints, legacy-route limitations, data persistence, and likely setup errors.
- Update root and frontend READMEs and finish the bilingual documentation index last so all navigation targets exist.

Acceptance: a new reader can follow the quickstart, complete the demo flow, and understand API example prerequisites in either language.

### 5. Verify the complete set

- Check local links, fragment anchors, images, reciprocal language links and the complete file-pair inventory.
- Render or preview Markdown, Mermaid, equations, tables and screenshots. Confirm readability and matching information in EN/VI.
- Check all commands and examples against the current repo. Explicitly report anything not executed and why; do not label it verified.
- Review terminology consistency and technical parity page by page, including configuration defaults and known limitations.
- Run repository-required checks: `ruff check src/ tests/` and `pytest tests/unit/ -v`. If frontend code/configuration changes become necessary, also run its typecheck, tests and build. Do not add unrelated code changes just to expand this documentation task.
- Run relevant demo/API/benchmark checks for claims made in the guides. Avoid paid API calls unless already authorized; the offline path and existing evidence can verify their own documented scope.

Acceptance: no broken local references, missing paired pages, unsupported claims, accidental secrets or unrelated files; required checks pass and verification limits are recorded.

### 6. Commit, push and merge into main

1. Fetch upstream state and create a documentation branch following `CONTRIBUTING.md`, for example `docs/bilingual-documentation`, from the verified integration base. Do not commit directly to `main`.
2. Review the diff and explicitly stage only the documentation and necessary shareable assets. Commit cohesive EN/VI pairs using Conventional Commits, for example `docs: add bilingual architecture and algorithm guides`.
3. Push the branch and open a PR into `develop`, with scope, paired-page inventory and actual verification results. Attach the created PR to the chat.
4. Wait for required CI checks and the repository's required reviewer approval. Address findings, updating both editions together, then merge using the repository-supported strategy.
5. Inspect the complete `develop` versus `main` diff before opening a release PR. If it includes unrelated changes not authorized for release, do not release them incidentally; prepare a documentation-only branch based on `main` using the reviewed documentation commits and open a scoped PR instead.
6. Open and attach the release/documentation PR into `main`, satisfy its checks and review requirements, and merge. Never force-push or bypass protected-branch requirements.
7. Fetch again and verify the delivered documentation is present in remote `main`. Report commit IDs, PR URLs, checks and any remaining limitations.

User authorization covers the requested documentation commit/push/merge. Do not ask for redundant permission. If credentials, branch protection, a required review, or an approval rejection prevents completion, preserve the pushed branch/PR where possible and report the exact blocker. Do not claim delivery to `main` until confirmed remotely.

## Completion checklist

- [ ] All deliverables and EN/VI pairs exist; EN is the default entry point.
- [ ] Existing authoritative references remain reachable; historical material is labeled.
- [ ] Architecture, algorithms and contracts match the verified implementation.
- [ ] Evaluation claims have reproducible evidence and honest limitations.
- [ ] Quickstart, user journey and API examples have recorded verification results.
- [ ] Links, previews, language parity and required repository checks pass.
- [ ] Documentation is committed, pushed, reviewed and merged into remote `main`.
- [ ] Final handoff includes PR links, commit IDs and verification summary.
