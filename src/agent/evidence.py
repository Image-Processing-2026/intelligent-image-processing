"""
Bằng chứng cho VLM (Analysis v2, Phase 5 / S3).

Chọn crop 100 % độ phân giải từ bản đồ đo cục bộ (Module 1) để Gemini soi chi tiết mà ảnh toàn
khung bị lấy mẫu lại làm mất: patch phẳng nhất và nhiễu nhất (soi nhiễu), cạnh mạnh nhất trên chủ
thể và trên nền (soi độ nét), khuôn mặt, vùng cháy/bệt lớn nhất. Mỗi crop có nhãn (A, B, …) và mô
tả số đo đi kèm. Thêm hai overlay (vùng clip, heatmap σ nhiễu).

Crop và overlay CHỈ để VLM nhìn, không bao giờ là pixel đầu ra (ADR-001: không sinh/sửa pixel
bằng AI; ở đây chỉ cắt, thu nhỏ và tô màu bằng OpenCV).
"""

import math
import string
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.analyzer_evaluator.measurement import (
    Estimate,
    MeasurementReport,
    MeasurementResult,
    aggregate,
)
from src.analyzer_evaluator.measurement.tiles import tile_mean

# Cạnh crop (pixel, 100 % độ phân giải gốc)
CROP_SIZE = 512
MAX_CROPS = 6
# Ảnh có cạnh dài không quá mức này: VLM đã thấy gần 100 % ở ảnh toàn khung → không cắt crop
MIN_LONG_SIDE_FOR_CROPS = 768
# Cạnh dài của overlay (ảnh thu nhỏ)
OVERLAY_MAX_SIDE = 768
# Hai crop chồng nhau quá mức này (IoU) → coi là trùng, lấy ứng viên kế tiếp
MAX_CROP_IOU = 0.3
# Cửa sổ thuộc một vùng khi trung bình mask trong cửa sổ đạt mức này
MIN_REGION_WEIGHT = 0.6
# Patch "phẳng" để soi nhiễu không được cháy/bệt và phải có độ sáng trung bình (L*)
FLAT_MAX_CLIP = 0.05
FLAT_L_RANGE = (12.0, 92.0)
# Crop "nhiễu nhất" chỉ khi σ nhiễu trong cửa sổ đạt mức này (mức xám 0–255)
NOISY_MIN_SIGMA = 1.5
# Crop vùng clip chỉ khi tỉ lệ pixel clip trong cửa sổ đạt mức này
CLIP_MIN_FRACTION = 0.05
# Overlay clip chỉ khi toàn ảnh có ít nhất chừng này pixel clip
OVERLAY_MIN_CLIP = 0.001
# Làm mịn trước khi tính gradient để nhiễu không bị tính là cạnh
GRADIENT_SMOOTH_SIGMA = 1.5
# Ngưỡng mask khuôn mặt để lấy bbox
FACE_MASK_THRESHOLD = 0.5
# Khung trung tâm dùng làm "chủ thể" khi chưa biết chủ thể (tỉ lệ cạnh)
CENTER_PRIOR_FRACTION = 0.5

EvidenceKind = Literal["face", "flat", "noisy", "edge_subject", "edge_background", "clip"]
BoxXYXY = Tuple[int, int, int, int]


@dataclass(frozen=True)
class EvidenceCrop:
    """Một crop 100 % gửi VLM kèm nhãn và mô tả số đo."""

    label: str  # "A", "B", …
    kind: EvidenceKind
    box: BoxXYXY  # pixel (x0, y0, x1, y1), nửa mở, trên ảnh gốc
    image: np.ndarray  # RGB uint8, không thu phóng
    description: str  # tiếng Việt, kèm số đo
    region: str  # vùng chứa crop: 'face', 'subject', 'background', 'full'


@dataclass(frozen=True)
class EvidenceOverlay:
    """Ảnh thu nhỏ có tô màu theo số đo, chỉ để VLM xem."""

    label: str  # "clip" | "noise"
    image: np.ndarray  # RGB uint8
    description: str


# ---------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------
def _as_rgb(image: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3); ảnh xám được lặp kênh."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _mask2d(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)


def _gradient_tiles(rgb: np.ndarray, tile_size: int) -> np.ndarray:
    """Độ lớn gradient trung bình theo tile, sau khi làm mịn nhẹ để nhiễu không thành cạnh."""
    luma = cv2.GaussianBlur(_luma(rgb), (0, 0), GRADIENT_SMOOTH_SIGMA)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    return tile_mean(np.sqrt(gx * gx + gy * gy), tile_size)


def _window_tiles(image_side: int, tile_size: int, grid_side: int) -> int:
    """Số tile của một cạnh cửa sổ crop (không vượt lưới)."""
    return max(1, min(grid_side, math.ceil(min(CROP_SIZE, image_side) / tile_size)))


def _window_stats(grid: np.ndarray, kh: int, kw: int) -> Tuple[np.ndarray, np.ndarray]:
    """(trung bình bỏ NaN, tỉ lệ tile hữu hạn) của mọi cửa sổ kh × kw trên lưới tile."""
    finite = np.isfinite(grid)
    values = np.where(finite, grid, 0.0).astype(np.float64)
    sums = sliding_window_view(values, (kh, kw)).sum(axis=(2, 3))
    counts = sliding_window_view(finite.astype(np.float64), (kh, kw)).sum(axis=(2, 3))
    means = np.full(sums.shape, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts / float(kh * kw)


def _window_box(ty: int, tx: int, tile_size: int, shape: Tuple[int, int]) -> BoxXYXY:
    """Bbox pixel của cửa sổ bắt đầu ở tile (ty, tx), cạnh CROP_SIZE, kẹp trong ảnh."""
    height, width = shape
    crop_h, crop_w = min(CROP_SIZE, height), min(CROP_SIZE, width)
    y0 = min(ty * tile_size, height - crop_h)
    x0 = min(tx * tile_size, width - crop_w)
    return x0, y0, x0 + crop_w, y0 + crop_h


def _centered_box(cx: float, cy: float, shape: Tuple[int, int]) -> BoxXYXY:
    """Bbox CROP_SIZE quanh tâm (cx, cy), kẹp trong ảnh."""
    height, width = shape
    crop_h, crop_w = min(CROP_SIZE, height), min(CROP_SIZE, width)
    x0 = int(round(min(max(cx - crop_w / 2, 0), width - crop_w)))
    y0 = int(round(min(max(cy - crop_h / 2, 0), height - crop_h)))
    return x0, y0, x0 + crop_w, y0 + crop_h


def _box_iou(first: BoxXYXY, second: BoxXYXY) -> float:
    ix = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    iy = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    inter = ix * iy
    area = (first[2] - first[0]) * (first[3] - first[1])
    other = (second[2] - second[0]) * (second[3] - second[1])
    union = area + other - inter
    return inter / union if union > 0 else 0.0


def _fmt(estimate: Optional[Estimate], digits: int = 1, scale: float = 1.0, unit: str = "") -> str:
    """Định dạng một ước lượng: 'giá trị đơn vị (tin cậy c)'; thiếu → 'không đo được'."""
    if estimate is None:
        return "không đo được"
    return f"{estimate.value * scale:.{digits}f}{unit} (tin cậy {estimate.confidence:.2f})"


def _grid_value(grid: Optional[np.ndarray], ty: int, tx: int, kh: int, kw: int) -> Optional[float]:
    """Trung bình bỏ NaN của lưới trong cửa sổ; None nếu không có tile hữu hạn."""
    if grid is None:
        return None
    window = grid[ty : ty + kh, tx : tx + kw]
    finite = window[np.isfinite(window)]
    return float(finite.mean()) if finite.size else None


# ---------------------------------------------------------
# Chọn crop
# ---------------------------------------------------------
class _Selector:
    """Trạng thái chung khi chọn crop: lưới tile, kích thước cửa sổ, crop đã chọn."""

    def __init__(self, rgb: np.ndarray, measurement: MeasurementResult) -> None:
        self.rgb = rgb
        self.shape = rgb.shape[:2]
        self.measurement = measurement
        self.tile = measurement.tile_size
        self.maps = measurement.maps
        self.gradient = _gradient_tiles(rgb, self.tile)
        gh, gw = self.gradient.shape
        self.kh = _window_tiles(self.shape[0], self.tile, gh)
        self.kw = _window_tiles(self.shape[1], self.tile, gw)
        self.boxes: List[BoxXYXY] = []

    def windows(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _window_stats(grid, self.kh, self.kw)

    def region_weight(self, mask: np.ndarray) -> np.ndarray:
        """Trung bình mask trong mỗi cửa sổ."""
        means, _ = self.windows(tile_mean(mask, self.tile))
        return np.nan_to_num(means, nan=0.0)

    def pick(self, score: np.ndarray, allowed: np.ndarray, maximize: bool) -> Optional[BoxXYXY]:
        """Cửa sổ tốt nhất (theo score) trong vùng cho phép, không trùng crop đã chọn."""
        valid = allowed & np.isfinite(score)
        if not valid.any():
            return None
        order = np.argsort(np.where(valid, -score if maximize else score, np.inf), axis=None)
        for flat_index in order[: int(valid.sum())]:
            ty, tx = np.unravel_index(int(flat_index), score.shape)
            box = _window_box(int(ty), int(tx), self.tile, self.shape)
            if all(_box_iou(box, other) < MAX_CROP_IOU for other in self.boxes):
                self.boxes.append(box)
                return box
        return None

    def tile_index(self, box: BoxXYXY) -> Tuple[int, int]:
        return box[1] // self.tile, box[0] // self.tile

    def window_value(self, key: str, box: BoxXYXY) -> Optional[float]:
        ty, tx = self.tile_index(box)
        return _grid_value(self.maps.get(key), ty, tx, self.kh, self.kw)

    def crop(self, box: BoxXYXY) -> np.ndarray:
        x0, y0, x1, y1 = box
        return np.ascontiguousarray(self.rgb[y0:y1, x0:x1])


def _value_text(selector: _Selector, box: BoxXYXY) -> str:
    """Số đo cục bộ của một crop (độ sáng, σ nhiễu, độ rộng cạnh) cho mô tả."""
    parts = []
    l_star = selector.window_value("exposure.mean_l", box)
    if l_star is not None:
        parts.append(f"L* {l_star:.0f}")
    sigma = selector.window_value("noise.sigma_luma", box)
    if sigma is not None:
        parts.append(f"σ nhiễu {sigma:.1f}")
    width = selector.window_value("sharpness.edge_width", box)
    if width is not None:
        parts.append(f"độ rộng cạnh {width:.1f} px")
    return ", ".join(parts) if parts else "không có số đo cục bộ"


def _subject_union(region_masks: Dict[str, np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Hợp các mask chủ thể (mọi vùng trừ 'full'/'background'); rỗng nếu không có."""
    union = np.zeros(shape, dtype=np.float32)
    for name, mask in region_masks.items():
        if name in ("full", "background"):
            continue
        arr = _mask2d(mask)
        if arr.shape == shape:
            union = np.maximum(union, arr)
    return union


def _center_prior(shape: Tuple[int, int]) -> np.ndarray:
    """Khung giữa ảnh (CENTER_PRIOR_FRACTION mỗi cạnh) làm chủ thể giả định."""
    height, width = shape
    mask = np.zeros(shape, dtype=np.float32)
    margin_y = int(height * (1 - CENTER_PRIOR_FRACTION) / 2)
    margin_x = int(width * (1 - CENTER_PRIOR_FRACTION) / 2)
    mask[margin_y : height - margin_y, margin_x : width - margin_x] = 1.0
    return mask


def _face_crop(
    selector: _Selector, region_masks: Dict[str, np.ndarray]
) -> Optional[Tuple[BoxXYXY, str]]:
    """Crop quanh khuôn mặt lớn nhất trong mask 'face' (nếu có)."""
    mask = region_masks.get("face")
    if mask is None:
        return None
    face = _mask2d(mask)
    if face.shape != selector.shape or face.max() < FACE_MASK_THRESHOLD:
        return None
    ys, xs = np.nonzero(face >= FACE_MASK_THRESHOLD)
    box = _centered_box((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, selector.shape)
    selector.boxes.append(box)
    stats = aggregate(selector.measurement, face, "face").values
    description = (
        f"Khuôn mặt (100 %): L* {_fmt(stats.get('exposure.mean_l'), 0)}, "
        f"σ nhiễu {_fmt(stats.get('noise.sigma_luma'))}, "
        f"cháy sáng {_fmt(stats.get('exposure.clip_high'), 1, 100, ' %')}."
    )
    return box, description


def select_evidence(
    image: np.ndarray,
    measurement: MeasurementResult,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
) -> List[EvidenceCrop]:
    """
    Chọn tối đa MAX_CROPS crop CROP_SIZE px (100 %) từ bản đồ đo, gắn nhãn A, B, … theo thứ tự:
    khuôn mặt → vùng clip lớn nhất → patch phẳng nhất → patch nhiễu nhất → cạnh mạnh nhất trên
    chủ thể → trên nền. Crop sau không được chồng crop trước quá MAX_CROP_IOU. region_masks: mask float32 (H, W) theo tên vùng ('face', subject…);
    không có chủ thể → khung giữa ảnh làm chủ thể giả định. Ảnh nhỏ (cạnh dài ≤
    MIN_LONG_SIDE_FOR_CROPS) → không cắt crop.
    """
    rgb = _as_rgb(image)
    shape = rgb.shape[:2]
    if max(shape) <= MIN_LONG_SIDE_FOR_CROPS:
        return []
    if (measurement.report.height, measurement.report.width) != shape:
        raise ValueError(
            f"measurement size {(measurement.report.height, measurement.report.width)} "
            f"does not match image size {shape}"
        )
    region_masks = region_masks or {}
    selector = _Selector(rgb, measurement)
    chosen: List[Tuple[EvidenceKind, BoxXYXY, str, str]] = []

    face = _face_crop(selector, region_masks)
    if face is not None:
        chosen.append(("face", face[0], face[1], "face"))

    clip_high = selector.maps.get("exposure.clip_high")
    clip_low = selector.maps.get("exposure.clip_low")
    gradient, _ = selector.windows(selector.gradient)
    everywhere = np.ones(gradient.shape, dtype=bool)

    # Vùng clip lớn nhất (cháy sáng hoặc bệt đen)
    best_clip: Optional[Tuple[float, str, np.ndarray]] = None
    for grid, name in ((clip_high, "cháy sáng"), (clip_low, "bệt đen")):
        if grid is None:
            continue
        fraction, _ = selector.windows(grid)
        peak = float(np.nanmax(fraction)) if np.isfinite(fraction).any() else 0.0
        if peak >= CLIP_MIN_FRACTION and (best_clip is None or peak > best_clip[0]):
            best_clip = (peak, name, fraction)
    if best_clip is not None:
        _, name, fraction = best_clip
        box = selector.pick(fraction, fraction >= CLIP_MIN_FRACTION, True)
        if box is not None:
            ty, tx = selector.tile_index(box)
            grid = clip_high if name == "cháy sáng" else clip_low
            local = _grid_value(grid, ty, tx, selector.kh, selector.kw) or 0.0
            chosen.append(
                (
                    "clip",
                    box,
                    f"Vùng {name} lớn nhất (100 %): {local:.0%} pixel bị {name} trong crop.",
                    "full",
                )
            )

    # Patch phẳng nhất: gradient nhỏ nhất, không cháy/bệt, độ sáng trung bình
    flat_allowed = everywhere.copy()
    for grid in (clip_high, clip_low):
        if grid is not None:
            flat_allowed &= np.nan_to_num(selector.windows(grid)[0], nan=1.0) < FLAT_MAX_CLIP
    mean_l = selector.maps.get("exposure.mean_l")
    if mean_l is not None:
        light = np.nan_to_num(selector.windows(mean_l)[0], nan=-1.0)
        flat_allowed &= (light >= FLAT_L_RANGE[0]) & (light <= FLAT_L_RANGE[1])
    box = selector.pick(gradient, flat_allowed if flat_allowed.any() else everywhere, False)
    if box is not None:
        chosen.append(
            (
                "flat",
                box,
                f"Vùng phẳng nhất (soi nhiễu, 100 %): {_value_text(selector, box)}.",
                "full",
            )
        )

    # Patch nhiễu nhất: σ nhiễu theo tile cao nhất, cần đủ tile đo được
    sigma_map = selector.maps.get("noise.sigma_luma")
    if sigma_map is not None and np.isfinite(sigma_map).any():
        sigma, coverage = selector.windows(sigma_map)
        # σ quá thấp thì crop không cho thấy gì (ảnh sạch): bỏ
        box = selector.pick(sigma, (coverage >= 0.25) & (sigma >= NOISY_MIN_SIGMA), True)
        if box is not None:
            chosen.append(
                (
                    "noisy",
                    box,
                    f"Vùng σ nhiễu cao nhất (100 %): {_value_text(selector, box)}.",
                    "full",
                )
            )

    # Cạnh mạnh nhất trên chủ thể và trên nền (soi độ nét)
    subject = _subject_union(region_masks, shape)
    known_subject = bool(subject.max() >= FACE_MASK_THRESHOLD)
    if not known_subject:
        subject = _center_prior(shape)
    subject_weight = selector.region_weight(subject)
    box = selector.pick(gradient, subject_weight >= MIN_REGION_WEIGHT, True)
    if box is not None:
        where = "chủ thể" if known_subject else "giữa khung (chưa rõ chủ thể)"
        chosen.append(
            (
                "edge_subject",
                box,
                f"Cạnh mạnh nhất trên {where} (soi độ nét, 100 %): {_value_text(selector, box)}.",
                "subject",
            )
        )
    background_weight = selector.region_weight(1.0 - subject)
    box = selector.pick(gradient, background_weight >= MIN_REGION_WEIGHT, True)
    if box is not None:
        chosen.append(
            (
                "edge_background",
                box,
                f"Cạnh mạnh nhất trên nền (soi độ nét, 100 %): {_value_text(selector, box)}.",
                "background",
            )
        )

    crops = []
    for label, (kind, box, description, region) in zip(string.ascii_uppercase, chosen[:MAX_CROPS]):
        crops.append(
            EvidenceCrop(
                label=label,
                kind=kind,
                box=box,
                image=selector.crop(box),
                description=description,
                region=region,
            )
        )
    return crops


# ---------------------------------------------------------
# Overlay
# ---------------------------------------------------------
def _thumbnail_size(shape: Tuple[int, int]) -> Tuple[int, int]:
    """(width, height) của ảnh thu nhỏ có cạnh dài ≤ OVERLAY_MAX_SIDE."""
    height, width = shape
    scale = min(1.0, OVERLAY_MAX_SIDE / max(height, width))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _dimmed_gray(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nền xám mờ (float32, 3 kênh) của ảnh thu nhỏ để màu overlay nổi bật."""
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) * 0.6
    return np.repeat(gray[..., None], 3, axis=2)


def build_overlays(image: np.ndarray, measurement: MeasurementResult) -> List[EvidenceOverlay]:
    """
    Tối đa hai overlay trên ảnh thu nhỏ: (1) tô đỏ pixel cháy sáng, xanh dương pixel bệt đen
    (khi có clip đáng kể); (2) heatmap σ nhiễu theo tile (khi đo được ở ít nhất một tile).
    """
    rgb = _as_rgb(image)
    size = _thumbnail_size(rgb.shape[:2])
    report = measurement.report
    overlays: List[EvidenceOverlay] = []

    high_any = report.exposure.get("clip_high_any")
    low_any = report.exposure.get("clip_low_any")
    high_fraction = high_any.value if high_any else 0.0
    low_fraction = low_any.value if low_any else 0.0
    if max(high_fraction, low_fraction) >= OVERLAY_MIN_CLIP:
        base = _dimmed_gray(rgb, size)
        high = cv2.resize(
            (rgb >= 250).any(axis=2).astype(np.float32), size, interpolation=cv2.INTER_AREA
        )
        low = cv2.resize(
            (rgb <= 5).any(axis=2).astype(np.float32), size, interpolation=cv2.INTER_AREA
        )
        base[high > 0.0] = (255.0, 0.0, 0.0)
        base[(low > 0.0) & (high == 0.0)] = (0.0, 80.0, 255.0)
        overlays.append(
            EvidenceOverlay(
                label="clip",
                image=base.astype(np.uint8),
                description=(
                    "Overlay clip: đỏ = cháy sáng (kênh ≥ 250), xanh dương = bệt đen (kênh ≤ 5); "
                    f"toàn ảnh {high_fraction:.1%} cháy, {low_fraction:.1%} bệt."
                ),
            )
        )

    sigma_map = measurement.maps.get("noise.sigma_luma")
    if sigma_map is not None and np.isfinite(sigma_map).any():
        finite = sigma_map[np.isfinite(sigma_map)]
        vmax = max(10.0, float(np.percentile(finite, 95)))
        normalized = np.clip(np.nan_to_num(sigma_map, nan=0.0) / vmax, 0.0, 1.0)
        heat_small = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(
            cv2.resize(heat_small, size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB
        ).astype(np.float32)
        measured = cv2.resize(
            np.isfinite(sigma_map).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        base = _dimmed_gray(rgb, size)
        base[measured] = 0.35 * base[measured] + 0.65 * heat[measured]
        overlays.append(
            EvidenceOverlay(
                label="noise",
                image=base.astype(np.uint8),
                description=(
                    "Heatmap σ nhiễu theo tile: tím đậm = σ thấp, vàng = σ cao (thang 0–"
                    f"{vmax:.0f} mức xám); vùng xám không tô = không đo được (không phẳng)."
                ),
            )
        )
    return overlays


# ---------------------------------------------------------
# Tóm tắt bằng chứng cho prompt
# ---------------------------------------------------------
def summarize_measurement(report: MeasurementReport) -> str:
    """Tóm tắt có cấu trúc của số đo toàn ảnh (giá trị + độ tin cậy), thay cho dump JSON."""
    exposure, noise, sharpness = report.exposure, report.noise, report.sharpness
    color, compression, haze = report.color, report.compression, report.haze
    lines = [
        f"SỐ ĐO CLASSICAL TOÀN ẢNH ({report.width}×{report.height} px, tile {report.tile_size}):",
        (
            f"- Phơi sáng: L* trung bình {_fmt(exposure.get('mean_l'))}; "
            f"p5 {_fmt(exposure.get('p5_l'))}; p95 {_fmt(exposure.get('p95_l'))}; "
            f"cháy sáng (≥1 kênh) {_fmt(exposure.get('clip_high_any'), 2, 100, ' %')}; "
            f"bệt đen (≥1 kênh) {_fmt(exposure.get('clip_low_any'), 2, 100, ' %')}; "
            f"mảng cháy lớn nhất {_fmt(exposure.get('largest_clip_blob'), 2, 100, ' %')} diện tích."
        ),
        (
            f"- Nhiễu (chỉ trên patch phẳng): σ luma {_fmt(noise.get('sigma_luma'))} mức xám; "
            f"σ Cb {_fmt(noise.get('sigma_cb'))}; σ Cr {_fmt(noise.get('sigma_cr'))}; "
            f"σ vùng tối / vùng sáng {_fmt(noise.get('sigma_dark_ratio'), 2)}."
        ),
        (
            f"- Độ nét (chỉ trên cạnh): độ rộng cạnh {_fmt(sharpness.get('edge_width_px'))} px, "
            f"chuẩn hóa cạnh dài 1000 px {_fmt(sharpness.get('edge_width_norm'))} px; "
            f"defocus σ {_fmt(sharpness.get('defocus_sigma'))}; "
            f"độ đồng hướng mờ {_fmt(sharpness.get('motion_coherence'), 2)} "
            f"(góc {_fmt(sharpness.get('motion_angle_deg'), 0, 1, '°')})."
        ),
        (
            f"- Màu (trên pixel gần trung tính): lệch nguồn sáng "
            f"{_fmt(color.get('illuminant_angle_deg'), 1, 1, '°')}, chromaticity r/g/b "
            f"{_fmt(color.get('illuminant_r'), 3)} / {_fmt(color.get('illuminant_g'), 3)} / "
            f"{_fmt(color.get('illuminant_b'), 3)}; độ bất đồng giữa thuật toán "
            f"{_fmt(color.get('algorithm_spread_deg'), 1, 1, '°')}; "
            f"chroma C* p50 {_fmt(color.get('chroma_p50'))}, p95 {_fmt(color.get('chroma_p95'))}; "
            f"tỉ lệ pixel trung tính {_fmt(color.get('neutral_fraction'), 1, 100, ' %')}."
        ),
        (
            f"- Nén JPEG: blockiness {_fmt(compression.get('blockiness'), 2)}; "
            f"sương mù: dark channel median {_fmt(haze.get('dark_channel_median'))}."
        ),
    ]
    if report.notes:
        lines.append(f"- Ghi chú đo: {'; '.join(report.notes)}.")
    return "\n".join(lines)


def describe_crops(crops: Sequence[EvidenceCrop]) -> str:
    """Danh sách crop (nhãn, vị trí, mô tả) cho prompt."""
    if not crops:
        return ""
    lines = ["CROP BẰNG CHỨNG (100 % độ phân giải, gửi kèm sau ảnh toàn khung):"]
    for crop in crops:
        x0, y0, x1, y1 = crop.box
        lines.append(f"- Crop {crop.label} [x {x0}–{x1}, y {y0}–{y1}]: {crop.description}")
    return "\n".join(lines)
