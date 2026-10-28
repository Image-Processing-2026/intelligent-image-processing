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

Curated basic-restoration starter set (30 real photos, 2026-10-27): see
[`BASIC_TEST_SET.md`](BASIC_TEST_SET.md). The manifest is `data/basic_test_manifest.json`;
download/verify with `python data/download_basic_test_set.py`. Local gallery:
`data/real/basic_v1/index.html`. Labels are agent-reviewed proposals for manual testing,
not human-verified ground truth or an independent hold-out.
1. **Synthetic Dataset (`data/synthetic/`):**
   - Paired test set where each image in `degraded/` has an exact 1:1 ground-truth match in `clean/`.
   - Used for unit testing algorithms and reporting objective reconstruction fidelity (PSNR/SSIM).
2. **Real Dataset (`data/real/`):**
   - Unpaired real-world photos with unknown ground truth.
   - Evaluated solely via `src.analyzer_evaluator.no_reference_eval`.
3. **Real-photo benchmark for the brain module (`data/real/benchmark/`):**
   - 26 Wikimedia Commons photos in 8 groups: `backlit`, `low_light`, `color_cast`, `noisy`,
     `multi_defect`, `intentional_style`, `already_good`, `group_small_faces`.
   - `data/real_benchmark_manifest.json` is tracked in git: source page, author, licence
     (CC0 / CC BY / CC BY-SA / public domain) and the expected answer per image
     (`scene_type`, `defects` with `min_severity`, `preserve`, `empty_plan`, `ambiguous`).
     The expected answers are a draft and still need team review.
   - The images themselves are gitignored. Fetch them with
     `python data/download_real_benchmark.py` (long side capped at 1600 px, about 16 MB).
   - When showing these images in a report or slides, credit the author and licence from the manifest.
