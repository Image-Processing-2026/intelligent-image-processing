# Module 3 benchmark — calibrate

Seed 2026, ảnh astronaut, coffee, chelsea, rocket (cạnh dài 384px). Trung bình trên các ảnh.

### calibrate

| case | method | best param | PSNR | SSIM | ΔE vs GT | ΔE invariance | s/img |
|---|---|---|---|---|---|---|---|
| GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA | gaussian | 0.06 | 30.64 | 0.7947 | 3.84 | — | 0.000 |
| MEDIAN_SIGMA_PER_KERNEL_STEP | median | 17.5 | 29.56 | 0.7783 | 4.07 | — | 0.001 |
| BILATERAL_SIGMA_PER_NOISE_SIGMA | bilateral | 4.0 | 32.99 | 0.8450 | 2.96 | — | 0.006 |
| NLM_H_PER_NOISE_SIGMA | nlm | 0.7 | 30.43 | 0.8113 | 3.22 | — | 0.619 |

Toàn bộ ứng viên (trung bình ảnh × σ ∈ (5, 10, 15, 20, 25)); chọn = PSNR cao nhất, hòa trong 0.15 dB thì xét SSIM:

| method | hằng số | hệ số | PSNR | SSIM | s/img | chọn |
|---|---|---|---|---|---|---|
| gaussian | GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA | 0.03 | 29.013 | 0.6633 | 0.000 |  |
| gaussian | GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA | 0.04 | 30.131 | 0.7306 | 0.000 |  |
| gaussian | GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA | 0.05 | 30.623 | 0.7730 | 0.000 |  |
| gaussian | GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA | 0.06 | 30.643 | 0.7947 | 0.000 | ✅ |
| gaussian | GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA | 0.07 | 30.476 | 0.8053 | 0.000 |  |
| gaussian | GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA | 0.08 | 30.280 | 0.8102 | 0.000 |  |
| median | MEDIAN_SIGMA_PER_KERNEL_STEP | 10.0 | 28.709 | 0.7917 | 0.004 |  |
| median | MEDIAN_SIGMA_PER_KERNEL_STEP | 12.5 | 29.270 | 0.7915 | 0.002 |  |
| median | MEDIAN_SIGMA_PER_KERNEL_STEP | 15.0 | 29.384 | 0.7867 | 0.001 |  |
| median | MEDIAN_SIGMA_PER_KERNEL_STEP | 17.5 | 29.555 | 0.7783 | 0.001 | ✅ |
| median | MEDIAN_SIGMA_PER_KERNEL_STEP | 20.0 | 29.555 | 0.7783 | 0.001 |  |
| median | MEDIAN_SIGMA_PER_KERNEL_STEP | 30.0 | 29.410 | 0.7408 | 0.000 |  |
| bilateral | BILATERAL_SIGMA_PER_NOISE_SIGMA | 2.0 | 30.688 | 0.7313 | 0.007 |  |
| bilateral | BILATERAL_SIGMA_PER_NOISE_SIGMA | 3.0 | 32.588 | 0.8179 | 0.007 |  |
| bilateral | BILATERAL_SIGMA_PER_NOISE_SIGMA | 4.0 | 32.985 | 0.8450 | 0.006 | ✅ |
| bilateral | BILATERAL_SIGMA_PER_NOISE_SIGMA | 5.0 | 32.790 | 0.8489 | 0.006 |  |
| bilateral | BILATERAL_SIGMA_PER_NOISE_SIGMA | 6.0 | 32.430 | 0.8453 | 0.006 |  |
| nlm | NLM_H_PER_NOISE_SIGMA | 0.5 | 29.699 | 0.7645 | 0.635 |  |
| nlm | NLM_H_PER_NOISE_SIGMA | 0.6 | 30.274 | 0.7972 | 0.618 |  |
| nlm | NLM_H_PER_NOISE_SIGMA | 0.7 | 30.426 | 0.8113 | 0.619 |  |
| nlm | NLM_H_PER_NOISE_SIGMA | 0.8 | 30.295 | 0.8145 | 0.623 | ✅ |
| nlm | NLM_H_PER_NOISE_SIGMA | 0.9 | 30.018 | 0.8124 | 0.627 |  |
