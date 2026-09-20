# Module 2 blend_regions evaluation

Evaluated 10 fixture cases from `tests\fixtures\blend_regions` without network access.

Exact cases: 9/10. Total differing channels: 3; one-off boundary differences: 3.
All differences satisfy the documented near-integer boundary rule: `True`.

The hard/soft plot uses a deterministic 128×192 RGB gradient and compares a binary rectangle with a feathered mask. PNGs were written with Matplotlib Agg and reopened by the image writer during generation.

See `metrics.csv` for E_max, MAE, RMSE, exact rates, and boundary classification.
