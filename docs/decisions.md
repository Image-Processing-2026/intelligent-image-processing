# Architectural Decision Records (ADR)

This document tracks major architectural and design decisions for the **intelligent-image-processing** project.

---

## ADR-001: Separation of AI Orchestration from Pixel Processing (No Direct Generative Models)
- **Status:** Accepted
- **Date:** 2026-08-30
- **Context:**
  The project is built for an academic Image Processing (Xử Lý Ảnh) course. Direct end-to-end generative models (e.g. Stable Diffusion, ControlNet, DALL-E) can produce high-quality aesthetics but act as black boxes that bypass the mathematical and algorithmic foundations of classical image processing.
- **Decision:**
  Use AI (Gemini Multimodal VLM + LangGraph) solely for perception, diagnosis, domain reasoning, tool selection, and loop evaluation. All pixel manipulations are performed by explicit classical image processing algorithms (OpenCV, scikit-image) written or configured by the team.
- **Consequences:**
  - The pipeline is fully explainable, transparent, and meets course requirements.
  - Operations must be strictly modularized and parameterized.

---

## ADR-002: Dual-Path Evaluation (Reference vs. No-Reference Metric Split)
- **Status:** Accepted
- **Date:** 2026-08-30
- **Context:**
  Standard metrics like PSNR and SSIM require an immaculate ground-truth reference image. However, user-uploaded damaged, noisy, or underexposed real photos have no ground truth. Applying PSNR against an arbitrary input produces invalid measurements.
- **Decision:**
  Explicitly partition the evaluation engine into two separate pipelines:
  1. **Synthetic Benchmark Pipeline (`reference_eval.py`):** Uses clean ground-truth images with controlled artificial degradation to compute exact PSNR, SSIM, and MSE for algorithm benchmarking.
  2. **Real Image Pipeline (`no_reference_eval.py`):** Uses no-reference quality metrics (BRISQUE, NIQE via `pyiqa` or OpenCV) combined with iteration-over-iteration technical metric improvement and VLM visual plausibility checks.
- **Consequences:**
  - Prevents erroneous metric calculations on real photos.
  - Requires maintaining two separate test datasets (`data/synthetic/` and `data/real/`).

---

## ADR-003: Continuous Soft-Masks & Edge Feathering for Region Blending
- **Status:** Accepted
- **Date:** 2026-08-30
- **Context:**
  Applying spatial filters (Gaussian blur, CLAHE, unsharp masking) through hard binary masks ($M \in \{0, 1\}$) produces severe visible seams and boundary artifacts where the processed patch meets the unedited image.
- **Decision:**
  All region masks are converted to continuous soft masks ($M \in [0.0, 1.0]$ with `dtype=np.float32`), smoothed with Gaussian/morphological feathering along borders. Processed results are composited using alpha blending:
  $$I_{\text{out}} = M \odot I_{\text{processed}} + (1 - M) \odot I_{\text{original}}$$
- **Consequences:**
  - Eliminates edge seams in local enhancements (e.g. sky CLAHE or face exposure correction).
  - Requires standardizing the mask contract to `float32 [0.0, 1.0]` across all modules.

---

## ADR-004: CPU-First Optimization & AMD Radeon iGPU Compatibility
- **Status:** Accepted
- **Date:** 2026-08-30
- **Context:**
  The primary development machine uses an AMD Radeon iGPU without NVIDIA CUDA support. Full-scale Segment Anything (SAM-ViT-H) and large PyTorch models require heavy CUDA memory.
- **Decision:**
  Adopt lightweight, CPU-efficient segmentation and vision models:
  - Text-prompted segmentation: MobileSAM / TinySAM combined with GroundingDINO.
  - Face detection: MediaPipe (optimized for CPU/iGPU).
  - As single-image processing (not real-time video) is the goal, 1-3 seconds CPU execution time is acceptable.
- **Consequences:**
  - Code runs universally across team laptops without CUDA setup friction.
  - Models load quickly with minimal memory footprint.
