# Audit-fix round: release note (R-01 to R-08, R-10)

Branch `fix/audit-backlit`. Contracts: `docs/interfaces.md` section 4.

## What changed
- **Honest outcome (R-01, R-08):** the result has `outcome` (`ACCEPTED`, `UNVERIFIED`, `NOT_MET`,
  `KEPT_ORIGINAL`, `NO_CHANGE`) separate from the loop `decision`. Real images are `ACCEPTED` only
  after a Gemini review accepts the returned image; no key or a failed call gives `UNVERIFIED`.
- **Face masks (R-02 to R-04):** exposure ops on a face use an oval by default; a failed mask QA
  falls back to the detection box at reduced strength. Every action reports what really ran
  (`ExecutedAction`: requested vs actual mode, backend, feather, `mask_hash`, area, skip reason).
- **Plan and measurement (R-05 to R-07):** `face_when_person` targets the face when Perceive found
  one; `defect_ref` keeps face actions tied to the parent defect; a person defect that looks
  resolved but whose face did not brighten is `unverified`; new `black_lift` side effect
  (severity capped at 1, never rolls back); side-effect check is two-way and reports
  `unavailable` on a measurement error.
- **API/UI:** outcome and review in the process result and export; executed-mask overlay stored
  with the run; `region_runtime` in `/system` and in the export; outcome badge in web and Gradio.

## Temporary thresholds (chưa hiệu chỉnh, BL-10)
All values below are placeholders, not fitted on data.

| Constant | Value | File |
|---|---|---|
| `FALLBACK_STRENGTH_SCALE` | 0.5 | executor |
| `FACE_QA_BOX_MARGIN` / `FACE_QA_MIN_INSIDE` | 0.05 / 0.9 | executor |
| `FACE_QA_MIN_COVERAGE` / `FACE_QA_MAX_COVERAGE` | 0.1 / 1.3 | executor |
| `FACE_QA_MAX_COMPONENTS_PER_FACE` / `FACE_QA_MIN_COMPONENT_SHARE` | 2 / 0.05 | executor |
| `FACE_MIN_GAIN_L` / `FACE_GAIN_RATIO` | 3.0 / 0.5 | fusion |
| `BLACK_LIFT_*` (`MIN_DELTA_L` 0.5, `DARK_L` 30, `MIN_FRACTION` 0.003, `THRESHOLDS` 4/8/14, `MAX_SEVERITY` 1) | see left | diff |
| `BACKLIT_LIGHT_PERSON_ENABLED` / `_SCALE` | False / 0.4 | knowledge/engine |
| `REVIEW_TIMEOUT_MS` / `REVIEW_MAX_CALLS` / `CROP_SIZE` / `MAX_CHANGE_CROPS` | 60000 / 2 / 384 / 2 | review |
| `MIN_CHANGE_LEVEL` / `SUSPECT_DARK_L` / `SUSPECT_MIN_LIFT` | 2.0 / 64 / 12 | review |

## Not done / known gaps
- Real `mobile_sam` inference was not run (needs the optional model download).
- `ExecutedAction.detection_box`, `expand_requested`, `expand_actual` are `None` until the
  executor fills them.
- The web UI does not show the executed-mask overlay yet (only the API returns it).
- Vitest and the web build were not run on the merged branch.
- R-09 (comparing recipe configurations on real images) is pending; defaults above are untuned.
