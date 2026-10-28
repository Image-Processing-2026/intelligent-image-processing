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
    # Analysis v2 (D4): every *_level above and defect_severity come from the calibrated
    # measurement (measurement.calibration, calibration.yaml): one threshold source.
    defect_severity: Dict[str, int] = Field(default_factory=dict)  # LEVEL_DEFECTS -> 0..3
    calibration_version: str = ""      # calibration.yaml version used


# ---------------------------------------------------------
# Module 4: Diagnosis Schema (Perceive stage, src/agent/state.py)
# ---------------------------------------------------------
class Evidence(BaseModel):             # Analysis v2: one structured piece of evidence
    source: Literal["measured", "vlm", "exif", "user"]
    metric: str = ""                   # e.g. "noise.sigma_luma"
    value: Optional[float] = None
    text: str = ""


class Defect(BaseModel):
    type: Literal[DEFECT_TYPES]        # underexposed, overexposed, backlit_subject, low_contrast,
                                       # noise, blur, color_cast_{warm,cool,green},
                                       # oversaturated, undersaturated; not fixable by the toolbox
                                       # (D2, fixable=False): motion_blur, jpeg_artifacts, haze,
                                       # highlight_clipping
    region: str = "full"               # "full" or a subject target_prompt
    severity: int                      # 0 negligible … 3 severe
    evidence: str = ""
    origin: Literal["vlm", "rule", "measured", "user"] = "vlm"
    # Analysis v2 (all defaulted, backward compatible)
    confidence: float = 1.0            # 0..1; VLM: self-consistency agreement
    evidence_items: List[Evidence] = []
    fixable: bool = True               # False -> no action is planned for it
    contraindications: List[str] = []  # e.g. "no_sharpen", "no_brighten"


class PreserveItem(BaseModel):
    aspect: Literal[PRESERVE_ASPECTS]  # warm_tone, cool_tone, low_key, high_key, silhouette,
                                       # film_grain, soft_focus, muted_colors, vivid_colors
    region: str = "full"
    reason: str = ""


class RegionMetrics(BaseModel):        # soft-mask weighted luma stats of one region
    region: str
    backend: str                       # Module 2 resolver that produced the mask
    area_ratio: float
    brightness_mean: float
    brightness_std: float
    brightness_level: Literal["underexposed", "normal", "overexposed"]
    highlight_clip_ratio: float
    shadow_clip_ratio: float
    brightness_vs_rest: Optional[float]  # region mean minus rest-of-image mean


class DiagnosisReport(BaseModel):
    iteration: int = 1
    scene_type: Literal[SCENE_TYPES] = "other"
    lighting: str = ""
    subjects: List[str]                # regions measured individually (max 4)
    defects: List[Defect]
    preserve: List[PreserveItem]       # intentional style; enforced by planner.apply_preserve_guard
    summary: str = ""
    region_metrics: Dict[str, RegionMetrics]
    source: Literal["vlm", "rule_based", "vlm_fallback"] = "vlm"
    # Analysis v2 (all defaulted)
    rejected: List[Defect] = []        # hypotheses dropped (e.g. below self-consistency quorum)
    ask_user: List[Defect] = []        # needs a user decision (clarify)
    measurement: Optional[MeasurementReport] = None  # whole-image measurement summary
    region_stats: Dict[str, RegionStats] = {}  # per-region measurement; 'full', 'background'
    subject_boxes: Dict[str, List[int]] = {}   # Gemini box_2d (ymin, xmin, ymax, xmax) 0..1000,
                                       # reused by the executor; never sent to the Plan prompt
    measurement_maps: Optional[Any] = None     # tile maps, in-process only (excluded from JSON)
    fusion_notes: List[str] = []       # one line per fusion decision (P6), not sent to Plan
    diff: Optional[Dict[str, Any]] = None  # iteration >= 2: measurement.diff.DiffReport as dict

    @property
    def actionable_defects(self) -> List[Defect]:  # severity >= 1 and fixable
        ...


# ---------------------------------------------------------
# Module 4: Agent Plan Schema (Emitted by VLM / Orchestrator)
# ---------------------------------------------------------
class RegionOperationPlan(BaseModel):
    region_id: str = Field(
        ..., description="Unique ID or target descriptor, e.g. 'sky', 'face', 'full_image'"
    )
    target_prompt: str = Field(..., description="Text prompt for segmentation or bounding box")
    # None → the executor infers the kind from target_prompt (face / full / quadrant / semantic)
    region_type: Optional[
        Literal["semantic", "face", "spatial", "full", "bbox", "binary_mask"]
    ] = None
    # bbox / binary_mask are in-process only: the planner resets VLM-supplied values to None
    bbox: Optional[tuple[int, int, int, int]] = None
    quadrant: Optional[str] = None
    # No ge/le bounds on region fields: planner.clamp_region_fields() clamps them
    # (REGION_FIELD_BOUNDS, e.g. feather_radius in [5, 50]) so one bad VLM value
    # cannot invalidate the whole plan. Face/semantic options (face_mode, num_faces,
    # instance_selection, instance_index, *_threshold) follow the same rule.
    feather_radius: int = 15
    expand_ratio: float = 0.15
    merge_policy: Literal["max"] = "max"
    binary_mask: Optional[Any] = None
    detected_issue: str = Field(
        ..., description="e.g. 'underexposed', 'high_noise', 'low_contrast'"
    )
    operation: Literal["denoise", "gamma_correct", "clahe", "sharpen", "color_correct"]
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters specific to operation (e.g. {'clip_limit': 2.0, 'gamma': 1.2})",
    )
    order: int = Field(
        default=0, description="Execution priority order (e.g., denoise before sharpen)"
    )


class TreatmentPlan(BaseModel):
    iteration: int = 1
    reasoning: str = Field(..., description="VLM clinical reasoning for proposed treatments")
    actions: List[RegionOperationPlan] = Field(..., description="Ordered list of operations")
    # "vlm": planned by Gemini; "rule_based": no GEMINI_API_KEY;
    # "vlm_fallback": the Gemini call or parse failed, so the rule-based plan was used.
    # An empty plan therefore only means "image already good" when source == "vlm".
    source: Literal["vlm", "rule_based", "vlm_fallback"] = "vlm"
    knowledge: List[str] = []          # playbook card / principle ids behind the plan


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


# src/analyzer_evaluator/measurement/ — local measurement (Analysis v2, P1)
def measure_image(image: np.ndarray, *, tile_size: int = 64) -> MeasurementResult:
    """Đo theo tile: phơi sáng (L*, clip), nhiễu (σ luma, Liu 2013), độ nét (defocus σ, độ rộng
    cạnh, hướng mờ chuyển động), nguồn sáng (pixel trung tính), blockiness, dark channel.
    MeasurementResult = report (MeasurementReport, JSON được) + maps (lưới tile, NaN = không đo)."""
    ...


def aggregate(result: MeasurementResult, mask: np.ndarray, region: str = "full") -> RegionStats:
    """Gộp bản đồ tile theo soft mask (H, W) thành số đo của một vùng (MAP_KEYS)."""
    ...


# src/analyzer_evaluator/measurement/calibration.py + calibration.yaml (Analysis v2, P2)
def severity_from_measurement(
    defect_type: str, stats: RegionStats | MeasurementReport, *, long_side: Optional[int] = None
) -> Tuple[int, float]:
    """(severity 0..3, confidence 0..1) theo ngưỡng hiệu chỉnh trên bộ synthetic; lỗi không đo
    được → (0, 0.0). long_side: cạnh dài ảnh gốc để quy σ mờ của RegionStats về 1024 px."""
    ...


def calibration_version() -> str: ...


# src/analyzer_evaluator/measurement/diff.py — iteration >= 2 (Analysis v2, P7)
class SideEffect(BaseModel):
    kind: Literal["halo", "dark_noise", "banding", "channel_clipping", "oversaturation"]
    region: str = "full"
    severity: int                      # 0..3
    value: float
    before: Optional[float] = None
    after: Optional[float] = None
    evidence: str = ""


def compare(original: np.ndarray, current: np.ndarray,
            regions: Optional[Dict[str, np.ndarray]] = None, *,
            defects: Sequence[Tuple[str, str]] = (), original_measurement=None,
            current_measurement=None, assess=None) -> DiffReport:
    """Trạng thái từng lỗi (resolved/improved/unchanged/worse) + tác dụng phụ so với ảnh gốc."""
    ...


def side_effects(original: np.ndarray, current: np.ndarray) -> List[SideEffect]:
    """Tác dụng phụ mức ≥ 1 do xử lý, đo trên ảnh ≤ 3000 px."""
    ...


# src/analyzer_evaluator/metadata.py — EXIF priors (Analysis v2, P3)
def read_metadata(file_bytes: bytes) -> ImageMetadata:  # never raises; failures -> None fields
    ...


def metadata_priors(meta: ImageMetadata) -> List[Prior]:  # noise, motion_blur, bokeh,
    ...                                                    # no_sharpen, edited (prompt hints)


def evaluate_reference(current_image: np.ndarray, ground_truth: np.ndarray) -> EvaluationResult:
    """Đánh giá chất lượng với ảnh gốc mẫu (PSNR, SSIM, MSE)."""
    ...


def evaluate_no_reference(
    current_image: np.ndarray, previous_image: Optional[np.ndarray] = None
) -> EvaluationResult:
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
    face_mode: Literal["bbox", "oval", "sam_refined"] = "bbox"
    num_faces: int = 4
    instance_selection: Literal["all", "largest", "index"] = "all"
    instance_index: Optional[int] = None
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.8


@dataclass(frozen=True)
class RegionResult:
    status: Literal["ok", "empty"]
    mask: np.ndarray  # float32, shape (H, W), [0, 1]; never None
    instance_masks: tuple[np.ndarray, ...]
    contours: tuple[np.ndarray, ...]  # pixel-space (N, 2), used by face_mode="oval"
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
    image: np.ndarray, text_prompt: str, feather_radius: int = 15,
    *, config: PromptSegmentationConfig | None = None,
) -> np.ndarray:
    """Phân đoạn vùng đối tượng dựa trên mô tả văn bản (GroundingDINO + MobileSAM).
    Trả về soft mask dtype=float32 trong khoảng [0.0, 1.0]."""
    ...


def detect_faces(image: np.ndarray) -> List[np.ndarray]:
    """Phát hiện và tạo mặt nạ khuôn mặt bằng MediaPipe."""
    ...

def detect_face_ovals(image: np.ndarray, *, num_faces: int = 4) -> List[FaceOval]:
    """Trả hard mask và contour face-oval suy từ MediaPipe Face Landmarker."""
    ...

def create_soft_mask(binary_mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    """Làm mịn biên mặt nạ (Gaussian feathering) để tránh tạo viền khi ghép ảnh."""
    ...


# Subject masks shared by Perceive (measurement) and the executor (processing), Analysis v2 P4a
@dataclass(frozen=True)
class SubjectMask:
    mask: np.ndarray          # float32 [0, 1], (H, W), feathered
    source: Literal["face_mediapipe", "dino_sam", "gemini_sam", "gemini_grabcut", "quadrant", "full"]
    confidence: float
    box: Optional[Tuple[float, float, float, float]]  # final pixel box (xmin, ymin, xmax, ymax)


def resolve_subject_mask(
    image: np.ndarray, name: str, box_2d: Optional[Tuple[int, int, int, int]] = None, *,
    use_semantic: Optional[bool] = None, cache_key: Optional[str] = None,
) -> SubjectMask:
    """face → MediaPipe; có model → GroundingDINO theo `name`, khớp bbox Gemini (IoU ≥ 0.3) →
    MobileSAM, không khớp → SAM trên bbox Gemini; không model → GrabCut trên bbox Gemini.
    Không bbox: DINO điểm cao nhất (confidence ≤ 0.5) → quadrant theo từ khóa → full."""
    ...


def subject_capabilities() -> Dict[str, bool]:  # {"face": ..., "semantic_models": ...}, no loading
    ...
```
`region_engine.capabilities()` (controller readiness, nested dict) is unchanged.

### 3.3 Module 3: Image Processing Engine (`src/processing_engine/`)
All operations implement the standard wrapped signature:
```python
def apply_denoise(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: Literal["gaussian", "median", "bilateral", "nlm"] = "bilateral",
    strength: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Khử nhiễu cục bộ hoặc toàn cục theo mặt nạ mềm."""
    ...


def apply_gamma(
    image: np.ndarray, mask: Optional[np.ndarray] = None, gamma: float = 1.0
) -> np.ndarray:
    """Hiệu chỉnh độ sáng phi tuyến tính bằng hàm Gamma."""
    ...


def apply_clahe(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Cân bằng lược đồ độ sáng cục bộ thích ứng độ tương phản (CLAHE)."""
    ...


def apply_sharpen(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: Literal["unsharp_mask", "laplacian"] = "unsharp_mask",
    amount: float = 1.0,
) -> np.ndarray:
    """Tăng cường độ sắc nét cục bộ."""
    ...


def apply_color_balance(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    saturation_scale: float = 1.0,
    temperature_shift: float = 0.0,
) -> np.ndarray:
    """Cân bằng trắng và điều chỉnh độ bão hòa màu sắc."""
    ...
```

### 3.4 Module 4: Agent & LangGraph (`src/agent/`)
```python
from langgraph.graph import StateGraph


# src/agent/graph.py
def run_pipeline(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    is_synthetic: bool = False,
    max_iterations: int = 3,
    intent: Optional[IntentProfile] = None,
    user_id: str = DEFAULT_USER,
    image_metadata: Optional[ImageMetadata] = None,  # Analysis v2: EXIF priors for Perceive
) -> DoctorState:
    """Khởi chạy toàn bộ vòng lặp khép kín: Analyze → Perceive → Plan → Process → Eval → Decide."""
    ...


# src/agent/vlm_diagnostician.py
def diagnose_and_plan(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,  # sent alongside `image` from iteration 2
    diagnosis: Optional[DiagnosisReport] = None,
) -> TreatmentPlan:
    """Stage 2 (Plan): Gemini (google-genai, structured JSON output) or rule-based fallback."""
    ...


# src/agent/perception.py — stage 1 (Perceive v2)
def perceive(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,
    metadata: Optional[ImageMetadata] = None,   # EXIF priors, added to the prompt as hints
    cache_key: Optional[str] = None,            # subject-mask cache key (original image)
) -> DiagnosisReport:
    """measure_image → Gemini with the full frame, ≤ 6 evidence crops at 100 %, overlays and
    the measurement summary (or the rule-based diagnosis) → defects conflicting with `preserve`
    dropped → subject masks (resolve_subject_mask with the Gemini box_2d) → region_stats and
    region_metrics → fusion.fuse (with verify). From iteration 2: re-measure, compare with
    the original (measurement.diff) and call the VLM only for open semantic defects or side
    effects to judge.
    PERCEIVE_SAMPLES Gemini samples (default 1, env PERCEIVE_SAMPLES) are merged by
    self-consistency; hypotheses below quorum go to `rejected`."""
    ...


# src/agent/fusion.py — cross-check and fusion (Analysis v2, P6)
def fuse(
    hypotheses: DiagnosisReport,                 # VLM (merged samples) or rule-based diagnosis
    measurement: Optional[MeasurementReport],
    region_stats: Dict[str, RegionStats],
    priors: Sequence[Prior] = (),
    intent: Optional[IntentProfile] = None,
    *,
    verifier: Optional[Verifier] = None,         # closed-question Gemini verify; None = none
    long_side: Optional[int] = None,
    crops_for: Optional[Callable[[str, str], List[str]]] = None,
    use_weak: Optional[bool] = None,             # rejected hypotheses as weak evidence (D3)
) -> DiagnosisReport:
    """Bảng quyết định tất định: defects = confirmed (confidence, evidence_items, fixable,
    contraindications), rejected, ask_user, preserve (+ aspect verify xác nhận chủ ý)."""
    ...


def plan_contraindications(report: Optional[DiagnosisReport]) -> List[str]: ...


# src/agent/planner.py
def apply_contraindications(plan: TreatmentPlan, contraindications: Sequence[str]) -> TreatmentPlan:
    """no_sharpen → bỏ mọi sharpen; no_brighten → bỏ gamma_correct > 1 toàn ảnh."""
    ...


# src/agent/evidence.py — evidence for the VLM (Analysis v2, P4b)
def select_evidence(image: np.ndarray, measurement: MeasurementResult,
                    region_masks: Optional[Dict[str, np.ndarray]] = None) -> List[EvidenceCrop]:
    """≤ MAX_CROPS (6) crops of 512 px at 100 %, labelled A, B, …: face → largest clip →
    flattest patch → noisiest patch → strongest edge on the subject → on the background.
    No crops when the long side ≤ 768 px."""
    ...


def build_overlays(image: np.ndarray, measurement: MeasurementResult) -> List[EvidenceOverlay]:
    """≤ 2 downscaled overlays (clipping map, …) sent with the crops."""
    ...


# src/agent/executor.py
def resolve_region_mask(
    image: np.ndarray, target_prompt: str, feather_radius: int = 15,
    box_2d: Optional[Sequence[int]] = None, cache_key: Optional[str] = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
    """Semantic region with a box_2d → resolve_subject_mask (the mask Perceive measured);
    otherwise face/full/quadrant/semantic through the controller."""
    ...


# src/agent/graph.py — used by the graph node and POST /api/v1/diagnose
def plan_treatment(
    image: np.ndarray,
    metrics: Dict[str, Any],
    diagnosis: Optional[DiagnosisReport],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,
) -> TreatmentPlan:
    """No actionable defect (severity >= 1 and fixable) → empty plan without a second VLM call.
    Otherwise diagnose_and_plan() → validate_and_sort_plan() → apply_preserve_guard() →
    apply_contraindications(plan, fusion.plan_contraindications(diagnosis))."""
    ...


# src/agent/knowledge — Knowledge Base (playbook cards + principles), see its README.md
def retrieve(diagnosis, metrics=None, kb=None, max_cards=4, max_principles=3) -> KnowledgeContext
def format_context(context: KnowledgeContext) -> str      # 'TRI THỨC CHUYÊN MÔN' prompt block
def plan_actions_from_knowledge(diagnosis, metrics, kb=None) -> (actions, card_ids)  # offline
def diagnose_from_metrics(metrics, iteration=1, source="rule_based") -> DiagnosisReport
```

The VLM plan schema has one `anyOf` variant per operation, each requiring exactly that
operation's parameters. Parameters that belong to another operation are dropped.
The rule-based plan (no API key, or Gemini failed) runs the recipes of the `auto_apply`
playbook cards that match the diagnosis plus the defects implied by the Module 1 levels.

Graph: `analyze → perceive → clarify → diagnose_and_plan → process → evaluate → decide`, then
`RE_PROCESS` loops back to `analyze` and anything else ends. Each run yields one result.
`POST /api/v1/diagnose` returns `{technical_metrics, diagnosis, treatment_plan}`.

**Treatment (`src/agent/treatment.py`):** `treatment_recipe(history)` returns the actions of every
iteration that was not rolled back, in order. Multi-style variants (balanced/natural/vivid) were
removed; old sessions keep their `variants` rows read-only in the workspace DB.
`planner.sanitize_actions(actions)` drops unknown operations and clamps parameters and region
fields **without reordering**, so a re-render at full resolution matches the original result.

**Intent conversation (Phase 4, `src/agent/intent.py`, `session.py`, `refine.py`):**
```python
class IntentProfile(BaseModel):        # src/agent/state.py
    style: Optional[Literal["natural", "balanced", "vivid"]] = None  # deprecated, always None
    edits: List[DiagnosisEdit]         # user edits to the diagnosis (chat "bổ sung phân tích")
    keep: List[PreserveItem]           # user-declared intentional style (overrides the VLM)
    fix: List[DEFECT_TYPES]            # defects the user wants fixed even if "intentional"
    notes: str = ""                    # free text, also sent to the Plan prompt
    answers: Dict[str, str]            # raw question id -> chosen value

def start_session(image, max_iterations=3) -> SessionResult                    # pauses
def answer_session(session_id, answers, notes="", edits=None) -> SessionResult # resumes
def refine(original, actions, feedback) -> RefineResult                        # stateless

class DiagnosisEdit(BaseModel):        # src/agent/state.py
    op: Literal["add_defect", "remove_defect", "keep"]
    type: Optional[DEFECT_TYPES]       # add_defect / remove_defect
    aspect: Optional[PRESERVE_ASPECTS] # keep
    region: str = "full"               # canonical_region
    severity: int = 2                  # 1..3, add_defect only
    source_text: str = ""

# src/agent/user_analysis.py
def parse_edits(text, diagnosis=None) -> Tuple[List[DiagnosisEdit], str]  # source: rule|vlm|vlm_fallback|none
def apply_edits(diagnosis, edits) -> DiagnosisReport                       # deterministic copy
```
- `parse_edits`: Vietnamese rules first, then Gemini (text only, no image) when the rules find
  nothing and a key is set. A user-added defect has `origin="user"`, is never dropped by fusion
  and is raised, not duplicated, when the diagnosis already has `backlit_subject` on the subject.
- Graph: `perceive → clarify → diagnose_and_plan`. `clarify` only acts on iteration 1 of an
  interactive run with no intent yet: it calls `interrupt()` with deterministic questions
  built from the diagnosis (at most 3 ambiguities, chosen in priority order: fusion `ask_user`,
  dark mood, warm tone, grain, cool tone). The list may be empty and the graph still pauses, so
  the user can add analysis notes. There is no style question any more. On resume
  (`Command(resume={"answers", "notes", "edits"})`) it builds the intent. `edits` are
  `DiagnosisEdit` dicts already parsed at the API layer (the node never parses text); a broken
  edit is skipped with a warning (`user_analysis.validate_edits`). Free-text notes are mapped by
  rules; explicit answers win.
- The intent overrides every iteration's diagnosis (`apply_intent`): `fix` removes conflicting
  preserve items and ensures the defect at severity ≥ 2 (`origin="user"`), and `keep` adds
  preserve items and drops conflicting defects. The Plan prompt gets an
  "Ý ĐỊNH NGƯỜI DÙNG" block (including a "Người dùng bổ sung phân tích" line listing the edits). `intent.edits` are applied last (`user_analysis.apply_edits`), so
  they beat the multiple-choice answers on every iteration. `run_pipeline(..., intent=...)` accepts a preset intent without asking.
- Sessions use an in-process `MemorySaver` (lost on restart; at most `MAX_SESSIONS`, oldest
  evicted).
- Feedback (`refine`): Vietnamese rules first, then Gemini when the rules find nothing and a
  key is set. The kinds are brighter/darker, more/less contrast, warmer/cooler, more/less
  saturated, sharper/softer and less noise, each with a region (face/sky/full) and a strength
  (1–2). Adjustments edit the treatment: denoise is inserted first, "less" kinds scale existing
  operations, and the rest are appended. The image is re-rendered from the original. Feedback
  is an explicit user request, so it bypasses the preserve guard.

**Case memory (Phase 5, `src/agent/memory.py`):**
- `configure_case_memory(path)` enables a SQLite store; `None` disables it. It is disabled by
  default. `src/api/main.py` and `frontend/app.py` (`__main__`) enable it at
  `CASE_MEMORY_PATH` (default `data/memory/cases.sqlite`, gitignored; an empty value disables
  it). The DB opens lazily on first use, and the test suite disables memory for every test.
- A `CaseRecord` is stored per finished run: user id, scene, a normalised 6-value metric
  signature of the original image, defects, preserve, intent style, treatment, best score,
  then **feedback** (texts and parsed adjustments) as the user acts. **No images are stored.**
- `similar(diagnosis, metrics, user_id)` returns up to 3 cases with distance ≤ 0.6. The
  distance is the RMS signature distance, plus 0.5 × (1 − Jaccard of defect types), minus
  0.2 for the same scene. On iteration 1 they go into the Plan prompt as
  "KINH NGHIỆM TỪ CA TƯƠNG TỰ", and VLM plans list them in `knowledge` as `case:<id8>`.

**API:** `POST /api/v1/process` returns `ProcessResponse`: `treatment` (the result's actions),
`intent` and `case_id` plus the usual result fields. `POST /api/v1/render` (multipart `file` +
form `actions` as a JSON list of `RegionOperation`) returns
`{image_base64, width, height, applied_actions}`. `/process` and `/sessions` accept `user_id`
(default `"default"`). `/refine` accepts an optional `case_id` to record understood feedback.
`GET /api/v1/memory?user_id=` returns `{enabled, cases, feedback}`.
- `POST /api/v1/sessions` (file, `max_iterations`) →
  `{session_id, status: "needs_input", questions, diagnosis}`.
- `POST /api/v1/sessions/{id}/answers` (JSON `{answers, notes}`) →
  `{status: "done", result: ProcessResponse}`. Returns 404 for an unknown session and 409 when
  the session is not waiting for input. `DELETE /api/v1/sessions/{id}` → 204.
- **Workspace (web app, `/api/v1/workspace`) — analysis chat before processing:**
  - `POST /sessions/{id}/analysis-notes` (JSON `{message}`, ≤ 1000 chars) parses one Vietnamese
    message with `user_analysis.parse_edits` against the draft diagnosis, appends the edits to
    `ProcessingSession.draft_edits` (max 20, else 422) and stores two messages: `analysis_note`
    (user) and `analysis_update` (`{text, edits, source, draft_diagnosis, unparsed}`). Only valid
    while the session is `awaiting_answers` (else 409 `invalid_state`). A message that yields no
    edit has `unparsed=true` and is appended to `notes` when processing starts.
  - `DELETE /sessions/{id}/analysis-edits/{index}` removes one edit (0-based; 404 if out of range).
  - `POST /sessions/{id}/answers` passes `draft_edits` to `answer_session(..., edits=...)` (or
    `intent_from_answers(..., edits=...)` for ground-truth sessions); `POST /sessions/{id}/clarify`
    clears them.
  - `SessionDetailOut` adds `draft_edits`, `draft_diagnosis` (diagnosis + edits, only while
    awaiting answers), `region_overlay_url` and `regions: [{name, source, confidence}]`. The
    overlay (`region_overlay.png`, artifact kind `region_overlay`) is built with
    `region_overlay.diagnosis_region_overlay` after the diagnosis and rebuilt when an edit changes
    the set of regions; a failure only logs a warning and leaves the URL `null`.
  - `processing_sessions.draft_edits` is added to an existing SQLite file at startup
    (`Database._add_missing_columns`); there is no Alembic.
- `POST /api/v1/refine` (file, `actions` JSON, `feedback`) →
  `{image_base64, width, height, actions, adjustments, notes, source}`, where `source` is
  `rules`, `vlm` or `none`. Send the returned `actions` back with the next feedback.

**Decision rules (`graph.decide_node`), in order:**
1. Degraded iteration → `STOP_BEST_EFFORT` and roll back to the previous image.
   A non-empty plan that changed no pixels (every action skipped) → `STOP_BEST_EFFORT`
   without rollback.
2. Empty plan → `SHIP`.
3. Synthetic: PSNR ≥ 28 dB and SSIM ≥ 0.88 → `SHIP`.
   Real: `estimated_quality_score` ≥ `REAL_TARGET_SCORE` (85) → `SHIP`;
   gain over the previous iteration < `REAL_MIN_GAIN` (1.0) → `STOP_BEST_EFFORT` (plateau).
4. Real images only: a score-based `SHIP` becomes `RE_PROCESS` when this iteration's plan acted
   on a defect with severity ≥ `VERIFY_SEVERITY` (2), meaning an action on its region or on the
   full image, and iterations remain. The next Perceive then confirms the fix.
5. Otherwise `RE_PROCESS`, or `STOP_BEST_EFFORT` when `max_iterations` is reached.

---

## 4. Audit-fix contracts (`src/agent/state.py`)

Added by the audit-fix round (`fix/audit-backlit`). Step 0 only introduces the types and their
defaults; behaviour is unchanged until the lane that owns each field fills it in.

| Type / field | Filled by | Meaning |
|---|---|---|
| `RegionOperation.defect_ref: Optional[str]` | planner (S2) | Defect the action targets, `type@region` (e.g. `backlit_subject@person`). Lets a sub-region action (face) be tracked as the parent region's defect. `None` falls back to matching the region name. |
| `ExecutedAction` | executor (S1) | Per-action report: `region_id, operation, parameters, requested_mode, actual_mode, backend, fallback_reason, feather_requested, feather_actual, strength_scale (1.0), mask_hash, mask_area, skipped, skip_reason`. |
| `SideEffectCheck` | `graph._side_effects_check` (S2 body, S3 caller) | `status` is `ok`, `unavailable` (measurement failed, never an empty list) or `not_run`; `previous` and `original` are `SideEffect` dicts for previous→candidate and original→candidate. |
| `ReviewFinding`, `ReviewResult` | `review.py` (S3) | `status` `completed|unavailable|not_run`; `verdict` `ACCEPT|REVISE|REJECT|UNCERTAIN`; plus `goal, preserve, findings, confidence, reason, candidate_sha256, model, prompt_version, latency_ms, error`. |
| `Outcome` | end of graph (S3) | `status` `ACCEPTED|UNVERIFIED|NOT_MET|KEPT_ORIGINAL|NO_CHANGE`; `accepted_by` `review|reference`; `candidate_sha256`; `reason`. Only `ACCEPTED` corresponds to `final_decision == SHIP`. |
| `HistoryItem.executed / side_effect_check / review` | `decide_node` | Copied from `executed_actions`, `_side_effects_check(state)` and the review node. `HistoryItem.side_effects` stays and mirrors `side_effect_check.previous`. |
| `DoctorState.executed_actions / outcome / review` | `process_node`, S3 | Report of the latest iteration; final outcome; final review. |

`ExecutedAction` value sets (filled by S1): `actual_mode` is `bbox`, `oval` or `sam_refined`;
`backend` includes `face_sam_roi` when SAM refined the face inside its ROI. `detection_box` is
the face detector box `[x0, y0, x1, y1]`, kept apart from the edit mask that is blended;
`expand_requested` / `expand_actual` mirror `feather_*` for the expand ratio.

`executor.execute_plan(..., report: Optional[List[ExecutedAction]] = None)` appends one
`ExecutedAction` per action when a list is passed. `EvaluationResult` is unchanged; agent and API
code must stop reading `visual_plausibility_passed` and `vlm_feedback` in favour of `Outcome`.

### 4.1 Mask tracing (R-02 to R-04)

- `region_engine.subject.resolve_subject_mask(image, name, box_2d, ..., feather_radius: Optional[int] = None)`:
  `None` uses `DEFAULT_FEATHER_RADIUS` (15). The feather is part of the cache key, so a caller
  that passes 15 and one that passes nothing share one cache entry; any other value does not.
- `executor`: `mask_fingerprint(mask)` returns `(sha256, area_ratio)` of the float32 mask rounded
  to 3 digits; `executed_mask(mask_hash)` returns the mask an action used (in-process LRU of 32
  entries, lost on restart); `face_mask_qa(mask, boxes, image_shape)` returns `None` when the mask
  passes or a failure reason.
- `agent.region_overlay.executed_region_overlay(image, executed)` draws the masks that were really
  executed, looked up by `mask_hash`.
- `GET /system` returns an extra `region_runtime = {interpreter, mobile_sam_installed, face,
  semantic_models}`.
- Face exposure operations (`gamma_correct`, `clahe`) default to `face_mode="oval"` when the plan
  did not set `face_mode` (checked with `model_fields_set`); an explicit `bbox` is respected.
  A fallback to `bbox` runs at reduced strength (`FALLBACK_STRENGTH_SCALE`, 0.5), and a failed
  mask QA counts as a failure. Thresholds in `executor.py` (`FALLBACK_STRENGTH_SCALE`,
  `FACE_QA_*`) are temporary and uncalibrated (BL-10).
