# Face-detection evaluation

Mock fixture cases: 7 passed, 0 failed.

`mode=mock`; `assets_ready=false`, `inference_executed=false`, and
`quality_passed=null`. This artifact does not claim real-model readiness.

Mock mode validates bbox-to-mask geometry only; it does not validate MediaPipe inference. The default command also requires prepared real assets and exits non-zero when they are missing.

Recreate with `python scripts/generate_face_mask_baselines.py` followed by `python scripts/evaluate_face_detection.py --mock-only`.
