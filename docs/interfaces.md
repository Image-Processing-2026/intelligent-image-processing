# Cross-Module Interface Contracts & Schemas

This document defines the strict contracts and schemas shared across the 4 modules in **intelligent-image-processing**. All modules must conform to these interfaces to guarantee seamless integration.

---

## 1. Image & Mask Data Formats

### 1.1 Image Format
- **Color Space:** Standard **RGB** (or BGR when calling OpenCV internals, but converted back to RGB for inter-module transfer).
- **Type:** `np.ndarray`
- **Dtype:** `np.uint8` (values `[0, 255]`) or `np.float32` (values `[0.0, 1.0]`). Default inter-module representation is `uint8` RGB of shape `(H, W, 3)`.
- **Grayscale:** Shape `(H, W)` or `(H, W, 1)`.

### 1.2 Mask Format
- **Type:** `np.ndarray`
- **Dtype:** `np.float32`
- **Value Range:** Continuous values in `[0.0, 1.0]` (where `0.0` = unedited background, `1.0` = full operation intensity).
- **Shape:** `(H, W)` or `(H, W, 1)`, strictly matching the spatial dimensions of the target image.
- **Blending Rule:**
  $$I_{\text{result}} = \text{mask} \cdot I_{\text{processed}} + (1.0 - \text{mask}) \cdot I_{\text{input}}$$

---

## 2. Pydantic Schemas

```python
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Module 1: Image Analyzer Output Schema
# ---------------------------------------------------------
class TechnicalMetrics(BaseModel):
    brightness_mean: float = Field(..., description="Mean luminance [0, 255]")
    brightness_level: Literal["underexposed", "normal", "overexposed"]
    contrast_std: float = Field(..., description="Standard deviation of intensity")
    contrast_level: Literal["low", "normal", "high"]
    noise_variance: float = Field(..., description="Estimated noise variance")
    noise_level: Literal["clean", "low", "medium", "severe"]
    sharpness_laplacian_var: float = Field(..., description="Laplacian variance")
    blur_level: Literal["sharp", "mild_blur", "severe_blur"]
    color_cast: Optional[str] = Field(None, description="e.g., 'warm', 'cool', 'greenish', 'none'")
    histogram_stats: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------------------------------------
# Module 4: Agent Plan Schema (Emitted by VLM / Orchestrator)
# ---------------------------------------------------------
class RegionOperationPlan(BaseModel):
    region_id: str = Field(..., description="Unique ID or target descriptor, e.g. 'sky', 'face', 'full_image'")
    target_prompt: str = Field(..., description="Text prompt for segmentation or bounding box")
    region_type: Literal["semantic", "face", "spatial", "full", "bbox", "binary_mask"] = "semantic"
    bbox: Optional[tuple[int, int, int, int]] = None
    quadrant: Optional[str] = None
    feather_radius: int = Field(default=15, ge=0)
    expand_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    merge_policy: Literal["max"] = "max"
    binary_mask: Optional[Any] = None
    detected_issue: str = Field(..., description="e.g. 'underexposed', 'high_noise', 'low_contrast'")
    operation: Literal[
        "denoise", 
        "gamma_correct", 
        "clahe", 
        "sharpen", 
        "color_correct"
    ]
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters specific to operation (e.g. {'clip_limit': 2.0, 'gamma': 1.2})"
    )
    order: int = Field(default=0, description="Execution priority order (e.g., denoise before sharpen)")

class TreatmentPlan(BaseModel):
    iteration: int = 1
    reasoning: str = Field(..., description="VLM clinical reasoning for proposed treatments")
    actions: List[RegionOperationPlan] = Field(..., description="Ordered list of operations")

# ---------------------------------------------------------
# Module 1 & 6: Evaluation Output Schema
# ---------------------------------------------------------
class EvaluationResult(BaseModel):
    iteration: int
    is_reference_eval: bool = False
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    brisque_score: Optional[float] = None
    niqe_score: Optional[float] = None
    technical_metrics: TechnicalMetrics
    visual_plausibility_passed: bool = True
    vlm_feedback: str = ""
    decision: Literal["SHIP", "RE_PROCESS", "STOP_BEST_EFFORT"]
```

---

## 3. Module Function Signatures

### 3.1 Module 1: Analyzer & Evaluator (`src/analyzer_evaluator/`)
```python
def analyze_image(image: np.ndarray) -> TechnicalMetrics:
    """Tính toán toàn bộ các chỉ số kỹ thuật của ảnh (độ sáng, tương phản, nhiễu, độ mờ)."""
    ...

def evaluate_reference(
    current_image: np.ndarray, 
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """Đánh giá chất lượng với ảnh gốc mẫu (PSNR, SSIM, MSE)."""
    ...

def evaluate_no_reference(
    current_image: np.ndarray, 
    previous_image: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Đánh giá chất lượng ảnh thực không có ground-truth (BRISQUE, NIQE, chênh lệch chỉ số)."""
    ...
```

### 3.2 Module 2: Region Engine (`src/region_engine/`)
```python
@dataclass(frozen=True)
class RegionRequest:
    kind: Literal["full", "bbox", "spatial", "face", "semantic", "binary_mask"]
    bbox: Optional[tuple[int, int, int, int]] = None
    quadrant: Optional[str] = None
    prompt: Optional[str] = None
    binary_mask: Optional[np.ndarray] = None
    feather_radius: int = 15
    expand_ratio: float = 0.15
    merge_policy: Literal["max"] = "max"


@dataclass(frozen=True)
class RegionResult:
    status: Literal["ok", "empty"]
    mask: np.ndarray  # float32, shape (H, W), [0, 1]; never None
    instance_masks: tuple[np.ndarray, ...]
    metadata: Dict[str, Any]


def resolve_region(
    image: np.ndarray,
    request: RegionRequest | Mapping[str, Any],
) -> RegionResult:
    """Resolve one region request without blending the image."""
    ...


def capabilities() -> Dict[str, Any]:
    """Report geometric readiness, AI assets/dependencies, and verification."""
    ...


def segment_by_prompt(
    image: np.ndarray, text_prompt: str, feather_radius: int = 15
) -> np.ndarray:
    """Phân đoạn vùng đối tượng dựa trên mô tả văn bản (GroundingDINO + MobileSAM).
    Trả về soft mask dtype=float32 trong khoảng [0.0, 1.0]."""
    ...

def detect_faces(image: np.ndarray) -> List[np.ndarray]:
    """Phát hiện và tạo mặt nạ khuôn mặt bằng MediaPipe."""
    ...

def create_soft_mask(binary_mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    """Làm mịn biên mặt nạ (Gaussian feathering) để tránh tạo viền khi ghép ảnh."""
    ...
```

### 3.3 Module 3: Image Processing Engine (`src/processing_engine/`)
All operations implement the standard wrapped signature:
```python
def apply_denoise(
    image: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    method: Literal["gaussian", "median", "bilateral", "nlm"] = "bilateral",
    strength: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Khử nhiễu cục bộ hoặc toàn cục theo mặt nạ mềm."""
    ...

def apply_gamma(
    image: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    gamma: float = 1.0
) -> np.ndarray:
    """Hiệu chỉnh độ sáng phi tuyến tính bằng hàm Gamma."""
    ...

def apply_clahe(
    image: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    clip_limit: float = 2.0, 
    tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Cân bằng lược đồ độ sáng cục bộ thích ứng độ tương phản (CLAHE)."""
    ...

def apply_sharpen(
    image: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    method: Literal["unsharp_mask", "laplacian"] = "unsharp_mask",
    amount: float = 1.0
) -> np.ndarray:
    """Tăng cường độ sắc nét cục bộ."""
    ...

def apply_color_balance(
    image: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    saturation_scale: float = 1.0,
    temperature_shift: float = 0.0
) -> np.ndarray:
    """Cân bằng trắng và điều chỉnh độ bão hòa màu sắc."""
    ...
```

### 3.4 Module 4: Agent & LangGraph (`src/agent/`)
```python
from langgraph.graph import StateGraph

def run_doctor_pipeline(
    image_bytes: bytes, 
    is_synthetic: bool = False, 
    ground_truth_bytes: Optional[bytes] = None,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """Khởi chạy toàn bộ vòng lặp khép kín: Analyze -> Diagnose -> Plan -> Process -> Eval -> Decision."""
    ...
```
