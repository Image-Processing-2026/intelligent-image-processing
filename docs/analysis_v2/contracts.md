# Analysis v2 — Hợp đồng interface giữa các lane

Tài liệu này **đóng băng** chữ ký hàm và schema mà các lane dùng chung, để các agent làm song song
mà không chờ nhau. Kế hoạch tổng: `docs/plan_analysis_v2.md`. Cách tổ chức lane:
`docs/analysis_v2/orchestration.md`.

Quy tắc:
- Lane **sở hữu** một mục thì cài đúng chữ ký ở đây. Lane **dùng** một mục thì code theo chữ ký ở
  đây, test bằng stub/fake, không chờ lane kia merge.
- Muốn đổi hợp đồng: không tự sửa. Ghi đề xuất vào `docs/analysis_v2/contract-changes.md`
  (lý do + diff chữ ký) và báo lead. Lead quyết định, sửa file này, báo các lane liên quan.
- Thêm trường **có default** vào model Pydantic là được phép, không cần đề xuất.
- Quy ước chung của repo vẫn áp dụng: ảnh RGB `uint8 (H, W, 3)` (grayscale cũng phải chạy),
  mask `float32 [0, 1]` đã feather (ADR-003), Pydantic v2, type annotation đầy đủ.

---

## C1. Đo cục bộ — sở hữu: lane `measure` (P1)

Package `src/analyzer_evaluator/measurement/`. Export từ `__init__.py`:

```python
from src.analyzer_evaluator.measurement import (
    Estimate, MeasurementReport, MeasurementResult, RegionStats,
    measure_image, aggregate,
)
```

```python
# schemas.py
class Estimate(BaseModel):
    value: float                 # NaN không được phép; không ước lượng được thì không có key
    confidence: float            # 0..1
    n_samples: int = 0

class MeasurementReport(BaseModel):
    """Tóm tắt toàn ảnh, JSON được (đi vào DiagnosisReport/API)."""
    width: int
    height: int
    tile_size: int
    exposure: Dict[str, Estimate] = {}
    noise: Dict[str, Estimate] = {}
    sharpness: Dict[str, Estimate] = {}
    color: Dict[str, Estimate] = {}
    compression: Dict[str, Estimate] = {}
    haze: Dict[str, Estimate] = {}
    notes: List[str] = []        # cảnh báo đo (ví dụ "few flat patches")

class RegionStats(BaseModel):
    region: str
    area_fraction: float                       # trung bình mask
    values: Dict[str, Estimate] = {}           # cùng key với MAP_KEYS, gộp có trọng số mask
    mask_source: str = "unknown"               # xem C2 MaskSource
    mask_confidence: float = 1.0

@dataclass
class MeasurementResult:
    """Không serialize ra API/HistoryItem."""
    report: MeasurementReport
    maps: Dict[str, np.ndarray]   # lưới tile float32, shape (ceil(H/T), ceil(W/T)); NaN = không đo được
    tile_size: int
```

```python
def measure_image(image: np.ndarray, *, tile_size: int = 64) -> MeasurementResult: ...
def aggregate(result: MeasurementResult, mask: np.ndarray, region: str = "full") -> RegionStats: ...
```

**Key bắt buộc** (tên cố định, lane khác truy cập theo key). Đơn vị ghi trong ngoặc.

| Nhóm | Key trong report | Ý nghĩa |
|---|---|---|
| exposure | `mean_l` (L\* 0–100), `p5_l`, `p95_l`, `clip_high_any`, `clip_high_all`, `clip_low_any`, `clip_low_all` (tỉ lệ 0–1), `largest_clip_blob` (tỉ lệ diện tích), `key` (log-mean luminance) | |
| noise | `sigma_luma`, `sigma_cb`, `sigma_cr` (đơn vị mức 0–255), `sigma_dark_ratio` (σ bin tối / σ bin sáng), `immerkaer` | Liu 2013 |
| sharpness | `edge_width_px`, `edge_width_norm` (chuẩn hóa cạnh dài 1000 px), `defocus_sigma`, `motion_coherence` (0–1), `motion_angle_deg`, `laplacian_var` | |
| color | `illuminant_angle_deg` (lệch so trung tính), `illuminant_r`, `illuminant_g`, `illuminant_b` (chromaticity, tổng = 1), `algorithm_spread_deg`, `chroma_p50`, `chroma_p95` (C\*), `neutral_fraction` | |
| compression | `blockiness` | |
| haze | `dark_channel_median` | |

**`MAP_KEYS`** (hằng số export, key của `maps` và `RegionStats.values`):
`exposure.mean_l`, `exposure.clip_high`, `exposure.clip_low`, `noise.sigma_luma`,
`sharpness.edge_width`, `sharpness.defocus_sigma`, `color.chroma`.

`aggregate`: mask (H, W) float32 được thu về lưới tile bằng trung bình; giá trị vùng = trung bình
có trọng số bỏ qua NaN; `confidence` giảm khi tổng trọng số hợp lệ nhỏ; `n_samples` = số tile
đóng góp.

Hiệu năng (không phải tiêu chí chính, chỉ để biết): ảnh 12 MP nên đo < 30 s trên CPU.

---

## C2. Phân vùng — sở hữu: lane `region` (P4a)

Export thêm từ `src/region_engine/__init__.py`:

```python
BoxXYXY = Tuple[float, float, float, float]   # pixel, (xmin, ymin, xmax, ymax), như spatial.create_bbox_mask
Box2D = Tuple[int, int, int, int]             # chuẩn Gemini: (ymin, xmin, ymax, xmax) trong 0..1000

MaskSource = Literal["face_mediapipe", "dino_sam", "gemini_sam", "gemini_grabcut", "quadrant", "full"]

@dataclass(frozen=True)
class SubjectMask:
    mask: np.ndarray            # float32 [0,1], (H, W), đã feather
    source: MaskSource
    confidence: float           # 0..1
    box: Optional[BoxXYXY]      # bbox cuối cùng đã dùng

# Export từ package dưới tên `subject_capabilities` (cài ở subject.capabilities).
# `region_engine.capabilities` là API cũ (controller, dict lồng nhau), giữ nguyên.
# Quyết định 2026-10-25, contract-changes.md mục 1 (A).
def subject_capabilities() -> Dict[str, bool]:
    """{"face": ..., "semantic_models": ...} — không nạp model nặng, chỉ kiểm tra file/gói."""

def box2d_to_xyxy(box_2d: Box2D, image_shape: Tuple[int, int]) -> BoxXYXY: ...

def segment_box(image: np.ndarray, box: BoxXYXY, *, feather_radius: int = 15) -> np.ndarray:
    """MobileSAM trong bbox. Raise SegmentationUnavailableError nếu không có model."""

def refine_bbox_mask(image: np.ndarray, box: BoxXYXY, *, iterations: int = 5,
                     max_side: int = 1024, feather_radius: int = 15) -> np.ndarray:
    """GrabCut khởi tạo từ bbox trên ảnh thu nhỏ, upscale mask, feather. Không cần model."""

def resolve_subject_mask(image: np.ndarray, name: str, box_2d: Optional[Box2D] = None, *,
                         use_semantic: Optional[bool] = None,
                         cache_key: Optional[str] = None) -> SubjectMask:
    """Thứ tự (plan P4): face → MediaPipe; có model → DINO theo `name`, giữ detection IoU ≥ 0.3
    với bbox Gemini, loại bbox phủ ≥ 90 % khung → SAM; không khớp → SAM trên bbox Gemini;
    không model → GrabCut trên bbox Gemini.
    Không bbox (`box_2d=None`): có model → DINO theo `name`, loại bbox phủ ≥ 90 % khung, lấy
    detection điểm cao nhất → SAM, `source="dino_sam"`, confidence ≤ UNANCHORED_DINO_CONFIDENCE
    (0.5); không detection hợp lệ hoặc không model → quadrant theo từ khóa → full
    (contract-changes.md mục 2).
    `use_semantic=None` = tự theo subject_capabilities(). Cache trong process theo
    (cache_key, name, box_2d) khi cache_key khác None."""

def clear_subject_mask_cache() -> None: ...
```

Ngưỡng (`DINO_MATCH_IOU = 0.3`, `DINO_MAX_BOX_FRACTION = 0.9`) là hằng số module, test được.
Không thêm thông báo lỗi khi thiếu model: chỉ `logger.info` một lần.

---

## C3. Metadata — sở hữu: lane `meta` (P3)

`src/analyzer_evaluator/metadata.py`:

```python
class ImageMetadata(BaseModel):
    iso: Optional[int] = None
    exposure_time_s: Optional[float] = None
    f_number: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    flash_fired: Optional[bool] = None
    camera: Optional[str] = None
    software: Optional[str] = None
    jpeg_quality_estimate: Optional[int] = None    # 1..100, từ bảng lượng tử
    icc_profile_name: Optional[str] = None

PriorKind = Literal["noise", "motion_blur", "bokeh", "no_sharpen", "edited"]

class Prior(BaseModel):
    kind: PriorKind
    strength: float      # 0..1
    reason: str          # tiếng Việt, đưa thẳng vào prompt

def read_metadata(file_bytes: bytes) -> ImageMetadata: ...      # không bao giờ raise; lỗi → các trường None
def metadata_priors(meta: ImageMetadata) -> List[Prior]: ...
```

Plumbing: `run_pipeline(..., image_metadata: Optional[ImageMetadata] = None)`;
`DoctorState["image_metadata"]` (Optional). Lane `meta` làm phần plumbing này.

---

## C4. Hiệu chỉnh severity — sở hữu: lane `calib` (P2)

`src/analyzer_evaluator/measurement/calibration.py` + `calibration.yaml`:

```python
def severity_from_measurement(defect_type: str, stats: Union[RegionStats, MeasurementReport],
                              *, long_side: Optional[int] = None) -> Tuple[int, float]:
    """(severity 0..3, confidence 0..1). defect_type nằm trong DEFECT_TYPES (kể cả defect mới
    của P6); defect không đo được → (0, 0.0).
    `long_side`: cạnh dài ảnh gốc (px) để quy σ mờ về 1024 px. MeasurementReport tự lấy
    max(width, height); RegionStats cần truyền vào (không truyền → không quy đổi).
    Người dùng duyệt 2026-10-25, contract-changes.md mục 5."""

def calibration_version() -> str: ...
```

Trước khi lane `calib` merge, lane khác dùng stub: `(0, 0.0)`.

---

## C5. Benchmark — sở hữu: lane `bench` (P0)

`scripts/benchmark_analysis.py`:

```
python scripts/benchmark_analysis.py --set real|synthetic --config baseline|measure_only|vlm_only|fused [--limit N] [--out DIR]
```

- Mỗi config là một hàm đăng ký trong `CONFIGS: Dict[str, Callable[[np.ndarray, CaseInfo], DiagnosisReport]]`.
  Lane `bench` cài `baseline` (luồng `perception.perceive` hiện tại) và `vlm_only`;
  `measure_only` và `fused` để `NotImplementedError` — lane `agent`/`fusion` điền sau.
- Kết quả: `results.json` (từng ca: expected, predicted, khớp) + `report.md` vào
  `data/real/benchmark_results/<timestamp>-<config>/` (thư mục này gitignored).
- Phần chấm điểm nằm trong `scripts/benchmark_scoring.py` (hàm thuần, có unit test offline):
  precision/recall theo defect, FP rate trên `already_good` + `intentional_style`, MAE severity,
  độ đúng vùng, preserve recall; synthetic: sai số tương đối σ nhiễu, σ mờ, góc nguồn sáng.
- Bộ synthetic: `scripts/build_calibration_set.py --out data/synthetic/calibration/` sinh ảnh +
  `params.json` cạnh mỗi ảnh:

```json
{"source": "already_good_01.jpg", "seed": 7,
 "degradations": [{"type": "gaussian_noise", "sigma": 10.0},
                  {"type": "gamma", "gamma": 1.5}],
 "expected": {"defects": [{"type": "noise", "region": "full", "severity": 2}]}}
```

Tên `type` degradation: `gaussian_noise`, `poisson_gaussian_noise`, `chroma_noise`, `gaussian_blur`,
`motion_blur`, `gamma`, `contrast_compress`, `illuminant_shift`, `saturation_scale`, `jpeg`, `haze`.

---

## C6. Chẩn đoán v2 — sở hữu: lane `agent` (P4b, P5) rồi `fusion` (P6, P7)

`src/agent/state.py` (bổ sung, mọi trường có default, tương thích ngược):

```python
class Evidence(BaseModel):
    source: Literal["measured", "vlm", "exif", "user"]
    metric: str = ""
    value: Optional[float] = None
    text: str = ""

class Defect(BaseModel):          # thêm
    confidence: float = 1.0
    evidence_items: List[Evidence] = []
    fixable: bool = True
    contraindications: List[str] = []

class DiagnosisReport(BaseModel):  # thêm
    rejected: List[Defect] = []
    ask_user: List[Defect] = []
    measurement: Optional[MeasurementReport] = None
    region_stats: Dict[str, RegionStats] = {}
```

Perceive schema: `subjects: [{"name": str, "box_2d": [ymin, xmin, ymax, xmax]}]` (0..1000).
Lane `agent` sở hữu `state.py`, `perception.py`, `vlm_diagnostician.py` trong wave 2; lane `fusion`
tiếp quản trong wave 3. Không lane nào khác sửa ba file này (trừ phần plumbing `image_metadata`
của lane `meta` ở wave 1).
