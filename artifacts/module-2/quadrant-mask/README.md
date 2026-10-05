# Quadrant-mask evaluation

Compared 312 soft outputs from 52 cases against independent references.

Soft pass/fail: 312/0; hard pixel mismatches: 0; minimum hard IoU: 1.

Worst soft case: `odd_5x7_right` / `right` at r=24, MaxAE=1.21053e-07, MAE=5.3635e-08, RMSE=7.16135e-08.

Recreate with `python scripts/generate_quadrant_mask_baselines.py` followed by `python scripts/evaluate_quadrant_mask.py --output-dir artifacts/module-2/quadrant-mask`.
