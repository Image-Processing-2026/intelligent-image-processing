# Module 3 benchmark — color

Seed 2026, ảnh astronaut, coffee, chelsea, rocket (cạnh dài 384px). Trung bình trên các ảnh.

### color

| case | method | best param | PSNR | SSIM | ΔE vs GT | ΔE invariance | s/img |
|---|---|---|---|---|---|---|---|
| cast=warm | input |  | 22.98 | 0.9745 | 8.00 | — | 0.000 |
| cast=warm | wb_gray_world |  | 19.45 | 0.9111 | 15.82 | 2.95 | 0.000 |
| cast=warm | wb_white_patch |  | 16.37 | 0.8901 | 12.80 | 9.61 | 0.000 |
| cast=warm | temperature_shift | -1.0 | 34.07 | 0.9909 | 2.58 | — | 0.000 |
| cast=cool | input |  | 24.53 | 0.9858 | 7.48 | — | 0.000 |
| cast=cool | wb_gray_world |  | 19.70 | 0.9363 | 15.31 | 1.31 | 0.000 |
| cast=cool | wb_white_patch |  | 24.85 | 0.9843 | 6.48 | 3.31 | 0.000 |
| cast=cool | temperature_shift | 1.0 | 42.68 | 0.9982 | 0.92 | — | 0.000 |
| cast=green | input |  | 25.99 | 0.9886 | 12.86 | — | 0.000 |
| cast=green | wb_gray_world |  | 19.77 | 0.9345 | 15.25 | 1.32 | 0.000 |
| cast=green | wb_white_patch |  | 23.79 | 0.9761 | 6.90 | 2.42 | 0.000 |
| cast=green | temperature_shift | 0.2 | 26.22 | 0.9880 | 12.49 | — | 0.000 |
