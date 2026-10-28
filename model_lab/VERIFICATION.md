# Verification · 2026-10-22

## Current mode · unrestricted image understanding · 2026-10-23

The earlier two-stage records below are historical. Active protocol is
`vision-free-v1`: one image-only user message plus an exact shared system prompt,
one response per model. No project tool contract, hidden user text, plan request,
forced issue/step counts, word count or mandatory response schema.

- Seven offline tests passed after the change. They check exactly 3 adapter calls,
  identical image bytes/instruction, text=None, no plan in new results, no extra text
  part or tools in native provider payloads, isolated retry and legacy read-only mode.
- Browser confirmed exactly one textarea, no tool whitelist in the default prompt,
  legacy results labelled without restoring their prompts into the new editor, then
  a new real run showing 0/3 models and exactly 3 visible response sections.
- Real run: `e1f04475-620f-4707-9ec8-f763cfdae328`, public astronaut darkened fixture.
  Gemma and Gemini returned image-only responses (52.67 s and 9.72 s respectively).
  Both reached the configured 1,000-token budget; UI warns that output is truncated.
  Groq initially returned 503 over-capacity; manually retried only Qwen once.
- Qwen retry succeeded (3.39 s), giving 3/3 actual responses. Qwen also reached
  its 1,000-token cap. All three truncations are explicitly indicated in the UI;
  no full-response/quality ranking is claimed from this smoke run.
- Copy preserves response text and truncation warning without inventing a separate
  plan section. Reload restores the exact free instruction; no plan sections visible.
  No browser console errors. Screenshot: `evidence-free-vision.png`.
- No document overflow at 1440, 900 or 390 px. Existing successful results remain
  visible while another provider fails or retries. No third-party judge or tools run.
- Remaining limits are provider policy/quota, image preparation, and the common
  user-selected output-token cap; removing toolbox restrictions does not remove them.

Status: implemented; technically verified for local manual comparison.
Owner visual acceptance: not reviewed. Public deployment: out of scope.
Integration: real Google and Groq APIs. No mock model replies in the delivered app.

## Real browser journey

Test image: existing `benchmarks/vlm_compare/pilot_v2/images/case_02.png`, a public
astronaut sample darkened for the earlier benchmark. No personal test image uploaded.
Saved run: `3cd79284-7c43-4379-9917-3e41ef7e98cd`, `runs/<id>/result.json` and image.
Both providers received the same 512×512 PNG and shared instructions; two calls each.

| Model | Analysis API time | Plan API time | Output tokens (analysis / plan) | Completion |
|---|---:|---:|---:|---|
| Gemma 4 31B | 69.64 s | 73.67 s | 506 / 759 | STOP / STOP |
| Qwen 3.8 27B | 6.02 s | 2.42 s | 783 / 831 | stop / stop |

No response was truncated. Groq cooldown displayed between stages; analysis stayed
readable while waiting. Time values exclude queue/cooldown, sample n=1, not a speed
benchmark. Native system instructions worked for both APIs. Thinking not exposed.
The default output cap was increased to 1,000 after observing the 783/831-token replies.

Manual observation: Qwen described a four-panel collage although this input is one
portrait. Original response preserved, no correction/judging inserted by the UI.

## Browser checks (Playwright)

- Upload through actual file input; submit; poll partial results; all 4 stages finished.
- Page reload restored image, context, prompts and results without extra model calls.
- Zoom dialog opens; Escape closes. Missing-file validation focuses file input.
- Filter "Chỉ đề xuất" shows exactly two plan sections, one per model.
- 1440 px and 390 px: no document horizontal overflow; columns stack on small screen.
- Normal and reduced-motion preferences inspected; no animations in either mode.
- Browser recovery check used request interception **only during testing** to simulate
  a Google 503: error/skipped shown, one applicable retry button, Qwen result retained.
  Interception removed and the real 4/4 results restored afterwards.
- No console errors; config/static/real upload requests returned HTTP 200.
- Screenshot of populated desktop view: `evidence-desktop.png` (before filter addition).
- Automated accessibility audit and representative user testing: not run. Native
  labels, keyboard focus, dialog Escape and status feedback checked as above.

## Offline checks

Five pytest cases cover shared image bytes and independent analysis→plan context;
failure isolation + retry including saved-run restore; upload/config validation and
secret boundaries; native provider payloads and hidden-thought exclusion; alpha
composition and no upscaling. Five passed. Ruff E/F/I checks and formatting passed.

Tests use mocked provider transport and do not spend API quota. TestClient needed
unsandboxed loopback on Windows; the initial sandbox run could not create asyncio's
socket pair. That process was stopped and tests rerun successfully.

## Known limits

- Local-only single active experiment; no auth or public deployment.
- 1,000-token cap may still truncate verbose responses; warning shown on provider flag.
- Stop cancels unsent stages, cannot retract an API request already sent.
- Model uncertainty and proposed parameter ranges are uncalibrated recommendations.
- Plans are text; no validator, mask generation or actual image filtering in this UI.

## Copy buttons

Added one "Copy kết quả" button per model. Copies model identity and the original
analysis + plan as Markdown, including sections hidden by result filters. Enabled
when any text is available; missing stages are labelled. Browser checks confirmed
both clipboard payloads exactly match saved model responses after Windows newline
normalization, success feedback appears, and 390 px layout has no overflow.

## Third candidate · 2026-10-23

- Added exact model `gemini-3.5-flash-lite`, shared Google credential, native
  systemInstruction and MINIMAL thinking. No replacement or fallback model.
- Two real smoke calls on the existing public darkened astronaut fixture succeeded:
  analysis 9.38 s / 561 output tokens; plan 4.52 s / 717 output tokens; both STOP,
  neither truncated. Results saved in `gemini_smoke.json`. These were standalone
  adapter calls, not a new six-call benchmark; existing candidates were not called.
- Six offline tests passed: updated shared-image/own-analysis test now checks all
  six calls; provider payload checks confirm Gemini uses its own Google model URL;
  legacy retry for a model absent from saved results returns 404. Ruff E/F/I passed.
- Browser: all three names and key-ready states; legacy run still 4/4, Gemini
  explicitly marked absent. Result filter shows 3 plan sections. No page overflow at
  1440, 1200, 900 or 390 px; three columns above 1100 px, stacked below.
- Browser-only intercepted fixture combined existing two-model responses with the
  two real Gemini smoke responses to check 6/6 rendering and exact Gemini clipboard
  text. No six-call run was claimed or saved by that check. Interception removed
  and the actual legacy saved run restored afterwards.
