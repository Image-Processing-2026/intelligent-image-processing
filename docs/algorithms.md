[English](algorithms.md) | [Tiếng Việt](algorithms.vi.md)

# Classical algorithms

Baseline: `02dd6f9`, inspected 2026-10-30. The five planner operations are `denoise`, `gamma_correct`, `clahe`, `sharpen`, `color_correct`. Tool names map to Python functions in the executor. Optional extensions are outside the MVP pipeline.

## Measurements and decisions

Measurement estimates exposure, noise, sharpness, color and other defects; calibration maps measurements to severity. VLM evidence and user intent inform diagnosis. A measured abnormality can also be intentional style. Diagnosis does not itself modify pixels. See [measurement code](../src/analyzer_evaluator/measurement) and [planner bounds](../src/agent/planner.py).

## Operations

| Operation | Method and parameter meaning | Main limitation |
|---|---|---|
| Denoise | Gaussian, median, bilateral or NLM; target noise sigma = `12.5 * strength` on the 0–255 scale | Strong filtering removes texture; NLM costs more CPU |
| Gamma | `out = 255 * (in / 255) ** (1 / gamma)` using a 256-entry LUT; gamma > 1 brightens, < 1 darkens | Cannot reconstruct clipped details; lifting shadows exposes noise |
| CLAHE | Contrast-limited adaptive histogram equalization on LAB L (directly on grayscale); `clip_limit` controls limiting, `tile_grid_size` sets the grid | Local contrast can amplify noise and look unnatural |
| Sharpen | Unsharp mask adds `amount * (image - blurred_image)`; Laplacian is also supported | Halos/noise; cannot restore arbitrary motion blur |
| Color | HSV saturation scaling and R/B temperature gains; positive temperature shift warms | Global gains cannot reliably correct mixed lighting |

Denoise planner strength is bounded to 0.1–2.0, corresponding to target sigma 1.25–25. This is a parameter convention, not a guarantee of a correct noise estimate. Unsharp uses Gaussian sigma 2 pixels. Temperature gain coefficient is `20/128`: R gain `1 + k*t`, B gain `1 - k*t`. Consult source for clipping, validation and grayscale handling.

The planner sorts denoise → gamma → CLAHE → sharpen → color. This reduces noise amplification before contrast/sharpness operations. Parameter bounds and preserve/contraindication guards live in the planner, rather than being inferred from prose.

## Region processing

An image is RGB `uint8`; a soft mask is `float32` in [0, 1] with matching spatial dimensions. Feathering reduces visible seams. With mask M, input I and processed image P:

$$O = M P + (1-M)I$$

A zero mask weight keeps the input pixel; one selects the processed pixel. The shared `apply_region_op` wrapper handles blending and crop halos. Neighborhood operations need context beyond the mask boundary. Region resolution may use spatial masks, face detection/landmarks, or installed semantic backends; inspect the actual execution record.

## Examples and limits

For a backlit face, local gamma can lift the subject while a soft mask protects the background. Check mask placement, boundary transitions, skin texture and highlight clipping after processing. This is a worked rationale, not measured pipeline success.

For reproducible before/after figures, run the [figure generator](../scripts/generate_doc_figures.py). Its controlled gradient example uses gamma 2.0 and a feathered circular mask; it illustrates the exact transfer function and blending, not semantic detection. The same figure is used in both editions.

![Synthetic gamma and mask example](assets/gamma-mask-example.png)

See [processing source](../src/processing_engine), [region source](../src/region_engine), and [existing parameter benchmarks](module3/benchmark/README.md) (Vietnamese reference).
