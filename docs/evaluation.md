[English](evaluation.md) | [Tiếng Việt](evaluation.vi.md)

# Evaluation and reproducibility

Documentation baseline: `02dd6f9`, 2026-10-30. Existing reports describe their own experiment snapshots. Their results must not be presented as fresh validation of this revision.

## Three kinds of evidence

| Evidence | Question answered | Limit |
|---|---|---|
| Unit/integration tests | Do contracts, guards, routes and execution behave as asserted? | Passing tests do not establish perceptual quality |
| Synthetic reference benchmarks | How close is restoration to an available clean image? | Artificial degradation may not represent real photographs |
| Real-image assessment | Does output meet the requested edit without side effects? | No clean target; taste and VLM variability affect judgment |

Reference evaluation uses PSNR/SSIM against an aligned clean target. Higher generally means closer, not necessarily more pleasing. No-reference evaluation optionally uses pyiqa BRISQUE/NIQE when models are available, otherwise internal fallback scoring. Report the actual metric backend; scores from different backends are not interchangeable. Never use the original degraded image or a fixed-filter baseline as a clean ground truth.

## Available experiments

The [Module 3 benchmark](module3/benchmark/README.md) uses four scikit-image images (astronaut, coffee, chelsea, rocket), long side 384 pixels, seed 2026 with case-specific CRC32 seeds, known synthetic degradations, PSNR/SSIM and mean CIEDE2000. Run from the repo root:

```powershell
python scripts/benchmark_module3.py --out-dir scratch/doc-benchmark
```

The [stored exposure report](module3/benchmark/exposure.md) reports mean PSNR 14.61 → 44.69 dB and SSIM 0.6192 → 0.9767 for the synthetic exponent-2 darkening case corrected with gamma 2.0. This is a historical parameter-sweep result with known degradation, not an agent success rate. It was not rerun for documentation. The report does not supply a complete hardware/source-revision provenance record, so timing and cross-revision claims remain unverified.

The [real-image prompt A/B report](reports/2026-10-29-prompt-ab-basic.md) describes 17 images, two prompt versions and one run per image/version. It includes controls and limit cases. It does not establish repeat-call stability. Its fixed-filter references are not ground truth; prompts, metadata and snapshot hashes belong to that protocol. Read the report before interpreting ratings or latency.

## New experiment protocol

Record source commit, environment versions, model/backend, prompt revision, dataset manifest and licenses, split, sample count, seed, preprocessing, command, hardware and raw output location. Keep calibration/training cases separate from holdout evaluation. Compare identical input sizes and constraints; distinguish cold start, model loading and request latency.

Evaluate target achievement and side effects separately. Include already-good images, intentional low-key/style images, noise plus dark exposure, local backlighting, and clipped highlights. Report failures and unchanged outputs. For portraits, inspect mask leakage, skin texture, halos, color shifts and background preservation at matched zoom. Do not infer a population accuracy percentage from a small selected gallery.

## Documentation figures

```powershell
python scripts/generate_doc_figures.py
```

This offline, deterministic example creates `docs/assets/gamma-mask-example.png` from a generated gradient. There is no downloaded image or private upload. It demonstrates gamma and blending only; it does not test diagnosis, semantics or overall restoration quality.

See the bilingual verification record in [documentation verification](documentation-verification.md) for executed checks and unresolved limits.
