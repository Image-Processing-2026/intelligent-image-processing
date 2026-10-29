[English](interfaces.md) | [Tiếng Việt](interfaces.vi.md)

# Cross-module contracts

Baseline `02dd6f9`. The [previous schema document](interfaces-reference.md) is retained as a reference snapshot; use runtime schemas for exhaustive current fields.

| Boundary | Invariant | Source |
|---|---|---|
| Image | Default RGB uint8 `(H, W, 3)`; supported grayscale paths retain valid shape | [processing base](../src/processing_engine/base.py) |
| Soft mask | float32 [0, 1], matching spatial size; `(H, W)` or supported singleton channel | [mask utilities](../src/region_engine/mask_utils.py) |
| Operation | `apply_*(image, mask=None, **params)` returns an array; shared wrapper blends | [engine](../src/processing_engine) |
| Diagnosis/plan/history | Pydantic validation at stage boundaries | [state](../src/agent/state.py) |
| Measurement | Structured measurements with calibration | [schemas](../src/analyzer_evaluator/measurement/schemas.py) |
| Workspace | Revision-aware request/response models | [API schemas](../src/api/workspace_schemas.py) |

Convert OpenCV BGR back to RGB at module boundaries. Blend once: `M*processed + (1-M)*input`. Validate finite values, range and dimensions. Read executor failure handling: request completion alone does not prove all actions ran.

The planner toolbox is fixed to denoise, gamma_correct, clahe, sharpen, color_correct. Changes require coordinated planner bounds, dispatch, prompt/schema, knowledge-card and documentation updates. Preserve guards and contraindications constrain plans. See [architecture](architecture.md).
