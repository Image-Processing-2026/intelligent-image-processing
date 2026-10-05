# Bbox-mask evaluation

Compared 819 soft-mask outputs from 117 cases against independent SciPy float64 references.

Soft pass/fail: 819/0; hard pixel mismatches: 0; minimum hard IoU: 1.

Worst soft case: `random_056` at r=25, MaxAE=1.76168e-07, MAE=3.37402e-08, RMSE=4.43685e-08.

Recreate with `python scripts/generate_bbox_mask_baselines.py` followed by `python scripts/evaluate_bbox_mask.py --output-dir artifacts/module-2/bbox-mask`.
