# Module 3 benchmark — exposure

Seed 2026, ảnh astronaut, coffee, chelsea, rocket (cạnh dài 384px). Trung bình trên các ảnh.

### gamma

| case | method | best param | PSNR | SSIM | ΔE vs GT | ΔE invariance | s/img |
|---|---|---|---|---|---|---|---|
| underexposed_exp=1.5 | input |  | 18.90 | 0.8351 | 9.93 | — | 0.000 |
| underexposed_exp=1.5 | gamma | 1.5 | 49.99 | 0.9948 | 0.44 | — | 0.000 |
| underexposed_exp=2.0 | input |  | 14.61 | 0.6192 | 16.22 | — | 0.000 |
| underexposed_exp=2.0 | gamma | 2.0 | 44.69 | 0.9767 | 0.63 | — | 0.000 |
| underexposed_exp=2.5 | input |  | 12.56 | 0.4670 | 20.52 | — | 0.000 |
| underexposed_exp=2.5 | gamma | 2.5 | 37.30 | 0.9416 | 1.13 | — | 0.000 |

### clahe

| case | method | best param | PSNR | SSIM | ΔE vs GT | ΔE invariance | s/img |
|---|---|---|---|---|---|---|---|
| low_contrast | input |  | 15.42 | 0.6597 | 15.80 | — | 0.000 |
| low_contrast | clahe | 2.0 | 16.32 | 0.7983 | 14.59 | — | 0.000 |
