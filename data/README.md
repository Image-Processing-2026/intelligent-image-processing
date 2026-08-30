# Test Datasets & Evaluation Data Organization

This directory manages datasets for benchmarking the **intelligent-image-processing** system.

---

## 1. Directory Structure

```text
data/
├── synthetic/               # For Reference-Based Evaluation (PSNR, SSIM, MSE)
│   ├── clean/               # Pristine ground-truth original images
│   └── degraded/            # Artificially corrupted images (controlled noise, blur, underexposure)
│
└── real/                    # For No-Reference Evaluation (BRISQUE, NIQE)
    └──                      # Real degraded photos, low-light snapshots, historical images
```

---

## 2. Usage Guidelines
1. **Synthetic Dataset (`data/synthetic/`):**
   - Paired test set where each image in `degraded/` has an exact 1:1 ground-truth match in `clean/`.
   - Used for unit testing algorithms and reporting objective reconstruction fidelity (PSNR/SSIM).
2. **Real Dataset (`data/real/`):**
   - Unpaired real-world photos with unknown ground truth.
   - Evaluated solely via `src.analyzer_evaluator.no_reference_eval`.
