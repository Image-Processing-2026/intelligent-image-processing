# Module 3 benchmark — sharpen

Seed 2026, ảnh astronaut, coffee, chelsea, rocket (cạnh dài 384px). Trung bình trên các ảnh.

### sharpen

| case | method | best param | PSNR | SSIM | ΔE vs GT | ΔE invariance | s/img |
|---|---|---|---|---|---|---|---|
| defocus_r=2 | input |  | 29.59 | 0.8747 | 2.00 | — | 0.000 |
| defocus_r=2 | unsharp_mask | 0.8 | 30.50 | 0.8959 | 1.91 | — | 0.000 |
| defocus_r=2 | laplacian | 0.5 | 30.02 | 0.8811 | 2.06 | — | 0.000 |
