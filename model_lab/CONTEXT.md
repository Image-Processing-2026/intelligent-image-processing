# Local comparison UI

## Current protocol · 2026-10-23 · supersedes the earlier two-stage design

confirmed (latest owner request): compare image understanding without project tools;
input consists only of the image and a shared editable system instruction.
implemented: one image request per model, three responses per run, no separate plan
request, no hidden user text/context, no project prompt/tool contract. Free-form
recommendations may appear in the same response. Copy/export/retry retained.
Legacy results are visibly labelled and read-only; their prompts never repopulate
the new editor. New protocol_version is vision-free-v1, with stages=[analysis].
Default instruction removes algorithm/parameter/ordering restrictions, output count
and word count requirements. API output-token limits remain technical settings.
Seven offline tests passed; browser and real-API evidence in VERIFICATION.md.
Historical entries below describe earlier implementations, not the active protocol.

- Assignment: new local feature, real integration, Standard depth, Phase 1→2→4→5.
- confirmed: owner wants upload → instructed visual analysis → recommendations,
  with two models side by side; owner will supply defective images (chat).
- evidence-backed: existing pilot adapters use gemma-4-31b-it and qwen/qwen3.8-27b;
  keys and dependencies exist in sibling project; toolbox bounds from its planner.py.
- assumed: Vietnamese, two automatic stages, light working interface, no new dependencies.
- Scope: real upload, same normalized image/context, editable system prompts,
  distinct analysis/plan, per-provider failure/retry, local saved JSON. No pixel execution.
- User: project owner evaluating model usefulness, confirmed by request. No user research.
- Journey: choose image → context/settings → submit → partial analysis → quota wait → plan
  → inspect/export. Error: keep other results, retry affected provider/stage after completion.
- States: empty, validation error, upload, pending/running/waiting, partial, success,
  API error, cancellation, network reconnect, restart/interrupted, long content.
- Feasibility: provider request paths real from pilot. Native system instructions and
  full two-stage journey require new smoke check. Upload/persistence backend new.
- UI: no route navigation; single form + image + two result columns. Native controls,
  visible focus, status live region, responsive stacked columns. Plain CSS/JS selected
  to avoid dependency installation. No animation needed. Owner review: not reviewed.
- Acceptance: manual upload produces two real independently visible analyses/plans;
  settings/prompts are saved; no browser key; errors preserve successes; reload resumes.
- Release: local review only. No publishing authorized or needed.

## Delivery update

implemented and technically verified: real four-call browser journey, five offline
tests, responsive/failure recovery checks; details in VERIFICATION.md.
visually accepted: not reached (owner review pending). Released publicly: not in scope.
Decision: 1,000 tokens default after real Qwen replies exceeded 768; same cap for both.
New pattern: synchronized result filters avoid misaligned sections in long answers.

## Third candidate · 2026-10-23

confirmed (owner request): add gemini-3.5-flash-lite alongside existing candidates.
implemented: third column, same Google key, native Google API path, identical image
and prompts, independent analysis→plan and copy/retry; dynamic 6-step progress.
Legacy saved runs retain their original two candidates without new API calls.
technically verified: six offline tests; two live Gemini responses; browser layout
checks at 1440/1200/900/390 px. Owner visual acceptance: not reviewed.
