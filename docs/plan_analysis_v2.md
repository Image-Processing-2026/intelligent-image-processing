# Plan nâng cấp luồng phân tích ảnh (Analysis v2)

Ngày: 2026-10-23. Trạng thái: **đề xuất**, chưa triển khai (trừ mục 0.1 đã làm).

Mục tiêu duy nhất: **độ chính xác của chẩn đoán**. Chưa tối ưu hiệu năng: chấp nhận nhiều lần gọi
Gemini hơn, đo ở độ phân giải gốc, thuật toán đo nặng hơn. Vẫn giữ ADR-001 (AI chỉ chẩn đoán, mọi
phép đo và xử lý pixel là classical) và ADR-004 (CPU-only).

---

## 1. Hiện trạng và vấn đề

### 1.1 Luồng hiện tại

```
Ảnh ─► analyze_image (toàn ảnh) ─► TechnicalMetrics (level theo ngưỡng tay)
     ─► perceive: Gemini (metrics JSON + ảnh toàn khung)  | không key/lỗi → luật từ level
     ─► DiagnosisReport (scene, subjects, defects 0-3, preserve)
     ─► bỏ defect mâu thuẫn preserve
     ─► measure_regions (ảnh ≤1024px): face=MediaPipe, sky/top...=quadrant, semantic=GroundingDINO
     ─► RegionMetrics (chỉ độ sáng) ─► thêm measured defect (backlit/under/over)
     ─► apply_intent ─► plan
```

### 1.2 Vấn đề (đã xác nhận khi đọc code)

| # | Vấn đề | Vị trí |
|---|---|---|
| V1 | Mọi chỉ số tính trên toàn ảnh, không có bản đồ cục bộ | `analyzer.analyze_image` |
| V2 | Nhiễu Immerkær đếm cạnh và texture thành nhiễu → ảnh nhiều chi tiết bị báo nhiễu | `_estimate_noise_immerkaer` |
| V3 | Var Laplacian bị nhiễu làm tăng → ảnh vừa mờ vừa nhiễu bị coi là nét; Tenengrad tính rồi bỏ | `analyze_image` bước 5 |
| V4 | Ngưỡng tuyệt đối (3/8/15, 100/300, 70/185) chọn tay, phụ thuộc độ phân giải và nội dung | `analyzer.py` đầu file |
| V5 | Ám màu = mean Lab toàn ảnh → cỏ, hoàng hôn, tường màu bị báo lệch màu | `_detect_color_cast` |
| V6 | Không đo độ bão hòa, dù `DEFECT_TYPES` có `oversaturated`/`undersaturated` | `state.DEFECT_TYPES` |
| V7 | Không có EXIF, blockiness JPEG, haze, hướng mờ | — |
| V8 | RegionMetrics chỉ có độ sáng → measured defect chỉ bổ sung lỗi phơi sáng | `perception.compute_region_metrics` |
| V9 | Không có GroundingDINO trên máy dev → mọi vùng ngữ nghĩa (`interior`, `pillar`…) bị bỏ qua | `executor.resolve_region_mask` |
| V10 | VLM chỉ nhận ảnh toàn khung + JSON, không có crop 100% hay bản đồ | `perception.perceive` |
| V11 | Đối chiếu một chiều: số đo bổ sung lỗi VLM bỏ sót, nhưng không kiểm tra lỗi VLM tự khai | `perception._measured_defects` |
| V12 | Severity do VLM đoán hoặc map cứng từ level | `knowledge/engine._metric_defects` |
| V13 | Vòng ≥2 chẩn đoán lại từ đầu; không phát hiện tác dụng phụ do xử lý (halo, nhiễu vùng tối tăng) | `perceive_node` |

---

## 2. Nguyên tắc thiết kế

1. **Tách Đo → Diễn giải → Phán quyết.** Phép đo cho bằng chứng định lượng; VLM cho ngữ cảnh
   (lỗi hay chủ ý, vùng nào quan trọng); một bước hợp nhất **tất định** ra kết luận.
2. **Cục bộ trước, toàn cục sau.** Đo thành bản đồ (tile), rồi gộp theo mask vùng. Toàn ảnh là
   một trường hợp gộp với mask = 1.
3. **Đo có nhận biết nội dung.** Nhiễu chỉ trên patch phẳng, độ nét chỉ trên cạnh, ám màu chỉ trên
   pixel gần trung tính.
4. **Mọi con số có độ tin cậy; mọi kết luận có bằng chứng.**
5. **Severity từ số đo đã hiệu chỉnh** trên dữ liệu có nhãn; VLM chỉ chỉnh ±1 kèm lý do.
6. **Luồng phân tích phải đo được độ chính xác** (Phase 0 là điều kiện tiên quyết).

---

## 3. Kiến trúc đích

```
                         ┌───────────────────────────┐
                         │ Ảnh + bytes file gốc      │
                         └─────────────┬─────────────┘
 ═══ S0. METADATA ═════════════════════▼═══════════════════════════════════════════
   EXIF (ISO, màn trập, khẩu độ, tiêu cự, flash), bảng lượng tử JPEG, ICC
   → priors: kỳ vọng nhiễu, nguy cơ mờ rung, khả năng bokeh, chất lượng nén
                                       │
 ═══ S1. BẢN ĐỒ ĐO CỤC BỘ (classical, độ phân giải gốc) ═▼═════════════════════════
   Phơi sáng │ Nhiễu        │ Độ nét/mờ      │ Màu            │ Nén / khác
   clip theo │ PCA trên     │ độ rộng cạnh,  │ nguồn sáng trên│ blockiness 8×8
   kênh, blob│ patch phẳng, │ defocus map,   │ pixel trung    │ haze (dark
   clip      │ σ(I), luma/  │ hướng mờ       │ tính, chroma C*│ channel)
             │ chroma       │                │                │
                                       ▼
                         MeasurementReport + MeasurementMaps
 ═══ S2. PHÂN VÙNG ════════════════════▼═══════════════════════════════════════════
   face: MediaPipe │ vùng do VLM liệt kê: bbox Gemini → DINO xác nhận → MobileSAM
   (không model: GrabCut; không API: quadrant) │ vùng vô danh: SLIC  → mask mềm (ADR-003) → RegionStats đầy đủ (gộp mọi bản đồ S1)
 ═══ S3. SINH GIẢ THUYẾT ══════════════▼═══════════════════════════════════════════
   H_measure: luật hiệu chỉnh trên RegionStats + prior S0
   H_vlm:     Gemini quan sát mở (ảnh toàn khung + crop 100% do S1 chọn + overlay),
              chạy N lần (self-consistency)
 ═══ S4. KIỂM CHỨNG CHÉO ══════════════▼═══════════════════════════════════════════
   đo được → đo lại có mục tiêu trên đúng vùng; mơ hồ → câu hỏi ĐÓNG kèm crop
 ═══ S5. HỢP NHẤT (tất định) ══════════▼═══════════════════════════════════════════
   bảng quyết định: số đo × VLM × preserve × intent
   → confirmed / rejected / preserved / ask_user, severity, confidence, fixable,
     contraindications  → DiagnosisReport v2 ─► clarify ─► plan
 ═══ S6. VÒNG ≥ 2: PHÂN TÍCH KHÁC BIỆT ═════════════════════════════════════════════
   so bản đồ ảnh hiện tại với ảnh gốc: lỗi đã hết chưa; tác dụng phụ do xử lý
```

---

## 4. Lộ trình

Mỗi phase là một PR riêng từ `develop` (`feat/analysis-v2-<phase>`). Phase 0 phải xong trước mọi
phase khác. Phase 1→2 tuần tự; 3 và 4 song song được sau Phase 1; 5→6→7 tuần tự.

```
P0 Benchmark ─► P1 Đo cục bộ ─► P2 Hiệu chỉnh ─┬─► P5 Bằng chứng VLM ─► P6 Kiểm chứng+Hợp nhất ─► P7 Vòng≥2
                     │                          │
                     ├─► P3 Metadata ───────────┤
                     └─► P4 Phân vùng ──────────┘                                 P8 Docs (cuối mỗi phase)
```

### Phase 0 — Benchmark độ chính xác chẩn đoán

**Mục đích:** có số baseline trước khi sửa bất cứ thứ gì, và có thước đo cho mọi phase sau.

**0.1 (ĐÃ LÀM, 2026-10-23):** ảnh gửi VLM giữ nguyên độ phân giải (PNG, hoặc JPEG q95 4:4:4 khi
vượt `VLM_INLINE_IMAGE_BUDGET`), `media_resolution=HIGH`. Đo thực tế: ảnh 12 MP ×2 mất ~45 s/lần gọi
so với ~10 s cho ảnh 1600 px. Gemini vẫn tự lấy mẫu lại theo ngân sách token, nên crop 100% (P5)
mới là cách để VLM thấy chi tiết thật.

**0.2 Bộ hiệu chỉnh synthetic** — `scripts/build_calibration_set.py` (mới)
- Ảnh sạch: `data/real/benchmark/already_good/` + ảnh sạch tự chọn (đủ đa dạng: chân dung,
  phong cảnh, texture dày như cỏ/vải, cảnh tối có chủ ý).
- Mỗi ảnh sạch sinh các biến thể với **tham số biết trước**, lưu `params.json` cạnh ảnh:
  - Nhiễu Gaussian σ ∈ {2, 5, 10, 15, 25}; nhiễu Poisson-Gaussian (phụ thuộc độ sáng); nhiễu chroma.
  - Mờ Gaussian σ ∈ {0.8, 1.5, 3}; mờ chuyển động dài {5, 15} px, góc ngẫu nhiên.
  - Gamma ∈ {0.5, 0.7, 1.5, 2.0}; giảm tương phản (co dải về [64, 192]).
  - Dịch nguồn sáng: nhân kênh theo CCT 3000K/4500K/8000K và tint xanh lá.
  - Bão hòa ×{0.5, 1.6}; JPEG q ∈ {20, 40, 70}; haze tổng hợp theo mô hình khí quyển `I = J·t + A(1−t)`.
  - Tổ hợp hai lỗi (nhiễu + tối, mờ + nhiễu) để kiểm tra nhầm lẫn chéo.
- Dùng để **hiệu chỉnh** (P2) và kiểm tra **sai số ước lượng** của từng phép đo (P1).

**0.3 Mở rộng manifest ảnh thật** — `data/real_benchmark_manifest.json` (schema_version +1)
- Thêm cho mỗi defect: `severity` (0–3) và `region`. Thêm `measurable_truth` khi biết (ví dụ ảnh
  nhiễu ISO cao có EXIF).
- Thêm nhóm `intentional_style` đủ lớn (đêm, low-key, silhouette, bokeh, hoàng hôn, film grain,
  muted) — nơi đo **báo lỗi sai**. Hiện có 26 ảnh: quá nhỏ để hiệu chỉnh, chỉ dùng để **kiểm định**.
- Lưu ý: manifest hiện tải ảnh về `max_side=1600`, việc thu nhỏ đã làm giảm nhiễu. Với nhóm nhiễu/mờ
  cần thêm bản gốc độ phân giải đầy đủ.

**0.4 Script chấm chẩn đoán** — `scripts/benchmark_analysis.py` (mới; tái sử dụng phần chấm của
`scripts/benchmark_brain.py`)
- Chạy 3 cấu hình trên cùng bộ ảnh: `measure_only` (S1+S2+luật), `vlm_only` (perceive hiện tại,
  không measured defect), `fused` (luồng mới). Baseline = luồng hiện tại.
- Chỉ số:
  - Precision / recall theo từng loại defect.
  - **False positive rate** trên `already_good` + `intentional_style`.
  - MAE severity; độ chính xác vùng (region đúng).
  - Preserve recall.
  - Trên bộ synthetic: sai số ước lượng σ nhiễu, σ mờ, góc lệch nguồn sáng.
- Xuất `report.md` + `results.json` vào `data/real/benchmark_results/<timestamp>/`.
- Hàm chấm điểm có unit test offline (`tests/unit/test_benchmark_scoring.py`).

**Tiêu chí xong:** có baseline của luồng hiện tại được commit trong `docs/` (bảng số), script chạy
lại được.

**Mục tiêu khởi điểm** (chỉnh lại sau khi có baseline):
- FP rate trên ảnh tốt và ảnh chủ ý ≤ 10%.
- Recall defect ≥ 85%; MAE severity ≤ 0.6.
- Sai số σ nhiễu ≤ 15% (σ ∈ [2, 30]) kể cả trên ảnh texture dày; ảnh texture sạch không bị báo nhiễu.
- Sai số góc nguồn sáng ≤ 3° trên ảnh có pixel trung tính.

---

### Phase 1 — Lõi đo cục bộ (Module 1)

**Vị trí:** package mới `src/analyzer_evaluator/measurement/`. Không xóa `analyze_image` /
`TechnicalMetrics`: nhánh đánh giá (`no_reference_eval`, delta metrics, `decide_node`) vẫn dùng.
Sau P2, các trường `*_level` của `TechnicalMetrics` được suy ra từ đo mới để hai nơi không mâu thuẫn.

```
measurement/
  __init__.py      measure_image(image) -> MeasurementResult  (report + maps)
  tiles.py         lưới tile, gộp bản đồ theo mask mềm: aggregate(maps, mask) -> RegionStats
  exposure.py
  noise.py
  sharpness.py
  color.py
  compression.py
  haze.py
  schemas.py       Estimate, MeasurementReport, RegionStats
```

**Kiểu dữ liệu chung**

```python
class Estimate(BaseModel):
    value: float
    confidence: float  # 0..1, từ số mẫu và độ đồng thuận
    n_samples: int

class MeasurementResult:          # dataclass, KHÔNG serialize ra API
    report: MeasurementReport      # Pydantic, tóm tắt toàn ảnh, JSON được
    maps: Dict[str, np.ndarray]    # bản đồ theo tile (float32), giữ trong bộ nhớ
    tile_size: int
```

Bản đồ không đưa vào JSON của API hay `HistoryItem`; chỉ `MeasurementReport` và `RegionStats`.

**1.1 Phơi sáng** — `exposure.py`
- Luma Rec.709 tuyến tính hóa (sRGB → linear) cho thống kê, giữ thêm L\* (CIELAB) cho ngưỡng cảm nhận.
- Clip **theo từng kênh** (≥ 250 hoặc ≤ 5); tỉ lệ pixel clip ít nhất một kênh và cả ba kênh.
- **Connected components** của mask clip sáng: blob ≥ 0.5% diện tích là vùng cháy, nhỏ hơn là
  phản xạ gương (không phải lỗi). Báo diện tích blob lớn nhất và vị trí.
- Bản đồ tile: mean L\*, p5, p95.
- Key (trung bình log luminance) và zone histogram (11 zone kiểu Ansel Adams) cho VLM và luật.

**1.2 Nhiễu** — `noise.py`
- Ước lượng chính: **Liu, Tanaka, Okutomi (2013)**, "Single-image noise level estimation for blind
  denoising": patch 7×7 (stride 3) trên luma; chọn lặp các patch yếu texture bằng ngưỡng trace ma
  trận hiệp phương sai gradient; σ² = trị riêng nhỏ nhất của hiệp phương sai các patch đã chọn.
  Có thể cài bằng NumPy/scikit-image, không cần dependency mới.
- **Hàm nhiễu theo độ sáng σ(I):** chia patch phẳng theo mean thành 8 bin, ước lượng σ mỗi bin.
  Phân biệt nhiễu Gaussian (phẳng) và nhiễu cảm biến (tăng ở vùng tối).
- **Luma và chroma riêng:** chạy lại trên Cb, Cr. Nhiễu chroma cao là dấu hiệu nhiễu rất đặc trưng
  (texture thật hiếm có dao động màu tần số cao).
- Bản đồ tile 64×64: σ cục bộ (NaN khi tile không đủ patch phẳng).
- Confidence theo số patch phẳng: < 30 patch → confidence thấp.
- Giữ Immerkær làm chỉ số phụ để so sánh trong benchmark.

**1.3 Độ nét / mờ** — `sharpness.py`
- Ước lượng nhiễu trước (1.2), khử nhiễu nhẹ có hướng dẫn bằng σ đó (chỉ để đo, không trả pixel ra
  ngoài), rồi mới đo độ nét. Đây là cách tách mờ khỏi nhiễu.
- **Độ rộng cạnh** (Marziliano 2002 / CPBD): Canny trên luma đã khử nhiễu, đo độ rộng cạnh dọc
  pháp tuyến. Báo hai giá trị: px gốc và chuẩn hóa theo cạnh dài 1000 px.
- **Defocus map** (Zhuo & Sim 2011): làm mờ lại với σ₀ = 1, tỉ số gradient R tại cạnh →
  σ = σ₀ / √(R² − 1); nội suy theo tile (median).
- **Chủ thể có nét không:** σ defocus trên mask chủ thể so với phân vị 10% của toàn ảnh. Chủ thể
  nét + nền mờ ⇒ bokeh (chủ ý); mọi nơi đều mờ ⇒ lỗi.
- **Mờ chuyển động:** structure tensor trên cạnh, độ coherence của hướng gradient; xác nhận bằng các
  vạch song song trên log-phổ FFT. Báo góc và độ dài ước lượng.
- Giữ Var Laplacian và Tenengrad làm chỉ số phụ.

**1.4 Màu** — `color.py`
- Tập pixel tin được: không clip, không quá tối (L\* > 15), chroma thấp (gần trung tính).
- Ước lượng nguồn sáng bằng 4 thuật toán trên tập đó: Gray-World, White-Patch (p99),
  Shades-of-Gray (p = 6), Gray-Edge (bậc 1, p = 6, σ = 2).
- Kết quả: màu nguồn sáng (chromaticity), **độ lệch góc so với trung tính**, hướng
  (warm/cool/green/magenta), độ tin cậy = độ đồng thuận góc giữa các thuật toán (lệch > 5° ⇒ thấp).
- Có khuôn mặt: kiểm tra hue/chroma da so với skin locus.
- **Bão hòa:** C\* = √(a\*² + b\*²) trên CIELAB float (đúng thang, không dùng mã hóa uint8 của
  OpenCV); p50, p95; tỉ lệ pixel cháy kênh do quá bão hòa.

**1.5 Nén** — `compression.py`
- Blockiness: tỉ lệ chênh lệch tuyệt đối trung bình tại biên khối 8×8 so với bên trong khối
  (kiểu Wang–Bovik–Evans 2002), theo cả hai hướng.
- Chất lượng JPEG ước lượng từ bảng lượng tử (đọc ở P3); không có file gốc thì chỉ dùng blockiness.

**1.6 Haze** — `haze.py`
- Dark channel (min RGB trên cửa sổ 15×15) median ở vùng không phải trời; kết hợp chroma thấp và
  tương phản cục bộ thấp.

**Test (offline, `tests/unit/test_measurement_*.py`):**
- Nhiễu: ảnh phẳng + texture + σ biết trước → sai số ≤ 15%; ảnh texture sạch → σ thấp.
- Độ nét: mờ Gaussian σ tăng → độ rộng cạnh tăng đơn điệu; ảnh nhiễu nhưng nét không bị coi là
  nét hơn do nhiễu.
- Màu: ảnh có patch xám bị nhân kênh biết trước → sai số góc ≤ 3°; ảnh cảnh vàng (hoàng hôn tổng
  hợp) có patch trung tính → không ám màu.
- Exposure: đốm phản xạ nhỏ không bị báo cháy; mảng trời cháy lớn thì có.
- Grayscale và (H, W, 1) vẫn chạy được (`test_grayscale_support.py`).
- Coverage `analyzer_evaluator` giữ ≥ 90%.

**Tiêu chí xong:** đạt các ngưỡng sai số synthetic ở mục 0.4.

---

### Phase 2 — Hiệu chỉnh severity

- `scripts/calibrate_severity.py`: trên bộ synthetic, khớp ánh xạ **số đo → severity 0–3** cho từng
  defect (ngưỡng theo phân vị, hoặc hồi quy thứ tự đơn giản), có xét theo cảm nhận (σ nhiễu ở vùng
  tối ít nghiêm trọng hơn ở vùng sáng).
- Kết quả lưu ở `src/analyzer_evaluator/measurement/calibration.yaml` (có version, ngày, bộ dữ liệu).
- `severity_from_measurement(defect_type, stats) -> (severity, confidence)`.
- Thay các ngưỡng tay ở `analyzer.py`: `*_level` của `TechnicalMetrics` suy ra từ calibration để
  `knowledge/engine._metric_defects` và chẩn đoán mới dùng chung một nguồn.
- Kiểm định trên ảnh thật (P0.3). **Không** hiệu chỉnh trên ảnh thật để tránh overfit 26 ảnh.

**Tiêu chí xong:** MAE severity trên synthetic ≤ 0.4; FP rate trên `already_good` không tăng.

---

### Phase 3 — Metadata (S0)

- `src/analyzer_evaluator/metadata.py`: `read_metadata(file_bytes) -> ImageMetadata` bằng Pillow
  (`getexif()`, `quantization` của JPEG). Không thêm dependency.
- `ImageMetadata`: iso, exposure_time, f_number, focal_length_35mm, flash, camera, software,
  jpeg_quality_estimate, icc_profile_name. Mọi trường Optional.
- **Priors** (`metadata_priors(meta) -> List[Prior]`):
  - ISO ≥ 1600 → tăng prior nhiễu.
  - Màn trập chậm hơn 1/(tiêu cự 35mm) → prior mờ rung.
  - f ≤ 2.8 → prior bokeh (nền mờ có chủ ý).
  - JPEG q < 60 → chống chỉ định sharpen.
  - Có `Software` (đã qua chỉnh sửa) → ghi chú cho VLM.
- **Plumbing** (Module 4 + API + UI): `run_pipeline(..., image_metadata=None)`; `DoctorState` thêm
  `image_metadata`; API `/diagnose`, `/process` đọc bytes upload; Gradio đọc từ file path. Không có
  metadata thì luồng vẫn chạy bình thường.

**Test:** JPEG tạo trong test với EXIF giả (Pillow ghi được EXIF); ảnh không EXIF; PNG.

---

### Phase 4 — Phân vùng (S2)

**Nguyên tắc (D1, chốt 2026-10-24):** VLM quyết định vùng nào **tồn tại**; DINO/SAM chỉ **định
vị**. Lý do: số đo ở §7 D1 cho thấy DINO báo nhầm "sky" khi ảnh không có trời và bắt người trên
poster, còn mask của SAM bám biên tốt hơn GrabCut rõ rệt (IoU giữa hai bên chỉ 0.25–0.73).

**Thứ tự phân giải vùng** (resolver mới `region_engine.resolve_subject_mask`, dùng chung cho
perception và executor):
1. `face` → MediaPipe (giữ nguyên).
2. Vùng do Perceive liệt kê (`{name, box_2d}`), **có model** (`region_engine.subject_capabilities()`):
   a. DINO chạy riêng từng tên vùng (**không gộp prompt**: gộp làm mất nhãn ở 4/8 ảnh, không nhanh hơn).
   b. Giữ detection DINO có IoU ≥ 0.3 với bbox Gemini; bỏ bbox DINO phủ ≥ 90 % khung (coi là trượt).
   c. Có detection khớp → bbox DINO → MobileSAM. Không khớp → bbox Gemini → MobileSAM, hạ confidence.
3. **Không có model** (CI, máy thành viên khác) → bbox Gemini → GrabCut trên ảnh thu về 1024 px,
   upscale mask → feather. Confidence thấp hơn bước 2.
4. **Không có bbox** (rule-based, hoặc VLM không trả `box_2d`): có model → DINO theo tên vùng, bỏ
   bbox phủ ≥ 90 % khung, lấy detection điểm cao nhất → MobileSAM, confidence chặn ở 0.5
   (`UNANCHORED_DINO_CONFIDENCE`); không có detection hợp lệ hoặc không model → quadrant heuristic
   theo từ khóa → full (phương án cuối). Chốt 2026-10-25, `analysis_v2/contract-changes.md` mục 2.

Mọi mask đi qua `mask_utils.create_soft_mask` (ADR-003). Mỗi `RegionStats` ghi `mask_source`
(`dino_sam` / `gemini_sam` / `gemini_grabcut` / `quadrant`) và `mask_confidence`.

**Cache mask:** khóa theo (hash ảnh gốc, tên vùng, bbox). Vòng 2–3 và executor dùng lại mask của
vòng 1 vì xử lý không đổi hình học. Không cache thì DINO tốn ~4.5 s × số vùng × 3 vòng; có cache
còn ~4.5 s/vùng + 0.6 s encoder SAM một lần. Nạp model (~13 s, RSS ~1.2 GB, đỉnh ~3.2 GB) một lần
mỗi process; API/Gradio nạp sẵn lúc khởi động khi có model.

**Việc cụ thể:**
- Module 2: tách `set_image` khỏi từng prompt (encoder SAM chạy một lần cho mọi bbox của ảnh);
  thêm `segment_box(image, box) -> np.ndarray` (MobileSAM, mask float32 feather) và
  `refine_bbox_mask(image, box, iterations=5, max_side=1024) -> np.ndarray` (GrabCut dự phòng).
  Thống nhất với Person 2; cập nhật `docs/interfaces.md` §3.2.
- Perceive schema: `subjects` thành `[{name, box_2d}]` (chuẩn hóa 0–1000). Prompt perceive cho
  phép trả bbox (khác plan prompt, vẫn cấm bbox).
- Phụ thuộc: `requirements-semantic.txt` tùy chọn (`torch` CPU, `transformers`, `mobile_sam` ghim
  `@f706ad9c`); CI không cài. Script `scripts/fetch_segmentation_models.py` tải asset ghim và kiểm
  sha256 theo `tests/fixtures/prompt_segmentation/assets.json` (điền `sha256`, revision). Thêm
  `models/segmentation/` vào `.gitignore` (hiện chỉ chặn `*.pt`, snapshot DINO sẽ lọt).
- `RegionStats` = `aggregate(maps, mask)`: sáng, clip (theo kênh), σ nhiễu, độ nét/defocus, nguồn
  sáng, chroma. Thay `RegionMetrics` (giữ các trường cũ để tương thích UI/API).
- Vùng nền = phần bù của hợp các subject; luôn đo `full` và `background`.
- (Tùy chọn) SLIC superpixel gộp theo màu/texture cho vùng không tên, để phát hiện lỗi cục bộ mà
  VLM không đặt tên (ví dụ một góc ảnh bị nhiễu nặng).

**Test:**
- Unit (offline, backend giả): thứ tự fallback đúng theo model/API có hay không; bbox DINO phủ
  toàn khung bị loại; DINO không khớp bbox Gemini → dùng bbox Gemini; cache trúng ở vòng 2;
  mask hợp lệ theo `executor._validate_mask`.
- Unit GrabCut: bbox giả trên ảnh synthetic có vật thể rõ → IoU ≥ 0.7.
- `real_model` (chạy local): các ca benchmark có người/trời đã chú thích → IoU mask người ≥ 0.6;
  `backlit_02`, `multi_defect_01` không sinh vùng "sky".

---

### Phase 5 — Bằng chứng cho VLM (S3)

- `src/agent/evidence.py`: `select_evidence(image, measurement, regions) -> List[EvidenceCrop]`:
  - Patch phẳng nhất (để soi nhiễu), patch nhiễu cao nhất.
  - Patch cạnh mạnh nhất **trên chủ thể** (soi độ nét) và trên nền.
  - Crop khuôn mặt.
  - Vùng clip lớn nhất.
  - Mỗi crop 100% độ phân giải, cạnh 512 px, có nhãn (`A`, `B`…) và mô tả số đo đi kèm.
- **Overlay** (chỉ để hiển thị cho VLM, không phải pixel đầu ra, phù hợp ADR-001): ảnh thu nhỏ có
  tô đỏ vùng clip sáng, xanh vùng clip tối; heatmap σ nhiễu theo tile.
- **Prompt perceive v2:** thay dump JSON metrics bằng **tóm tắt bằng chứng có cấu trúc** (giá trị +
  độ tin cậy + vị trí), mô tả từng crop. Yêu cầu VLM trả: scene, lighting, subjects + bbox,
  **giả thuyết** defect (kèm crop làm bằng chứng), preserve. Cho phép và khuyến khích "không có lỗi".
- **Self-consistency:** chạy perceive N = 3 lần (temperature 0.7), gộp: defect xuất hiện ≥ 2/3 lần
  thì giữ, tỉ lệ đồng thuận thành confidence; scene/preserve theo đa số.
- Ngân sách: ảnh toàn khung + ≤ 6 crop + 2 overlay mỗi lần gọi.

**Test:** chọn crop đúng vị trí trên ảnh synthetic (patch phẳng nằm ở vùng phẳng); prompt chứa đủ
nhãn crop; gộp self-consistency với 3 phản hồi mock khác nhau.

---

### Phase 6 — Kiểm chứng chéo và hợp nhất (S4 + S5)

- `src/agent/fusion.py`: `fuse(hypotheses, measurement, region_stats, priors, intent) ->
  DiagnosisReport`. Hoàn toàn tất định, test được offline.
- **Kiểm chứng** cho giả thuyết mơ hồ: một lần gọi Gemini `verify` với **câu hỏi đóng** theo schema:

  ```json
  {"answers": [{"hypothesis_id": "h2",
                "verdict": "defect | intentional | texture | not_present",
                "confidence": 0.0, "reason": "..."}]}
  ```

  Ví dụ: "Hạt trong crop B là nhiễu hay texture vải?", "Nền mờ ở crop D là xóa phông có chủ ý hay
  lệch nét?". Chỉ hỏi khi bảng dưới yêu cầu.

- **Bảng quyết định:**

  | Số đo (sau hiệu chỉnh) | VLM / verify | Kết luận |
  |---|---|---|
  | severity ≥ 1, conf cao | defect | **confirmed**, severity = số đo (VLM ±1 có lý do) |
  | severity ≥ 1 | intentional (preserve) | **preserved**; nếu severity = 3 trên mặt/chủ thể → **ask_user** |
  | severity ≥ 1, conf cao | không nhắc | **confirmed**, confidence trung bình |
  | severity ≥ 1, conf thấp | không nhắc | hỏi verify; không xác nhận → bỏ |
  | severity 0 | defect đo được (noise, blur, exposure, color, saturation) | verify với crop; vẫn `not_present` → **rejected** (ảo giác) |
  | không đo được | defect ngữ nghĩa | **confirmed**, confidence = tỉ lệ đồng thuận N lần |
  | bất kỳ | — | **intent người dùng thắng** (`apply_intent` sau cùng) |

- Prior EXIF chỉ dịch confidence (không tự tạo defect).
- **DiagnosisReport v2** (`state.py`), tương thích ngược (trường mới đều có default):

  ```python
  class Evidence(BaseModel):
      source: Literal["measured", "vlm", "exif", "user"]
      metric: str = ""          # ví dụ "noise.sigma_luma"
      value: Optional[float] = None
      text: str = ""

  class Defect(BaseModel):      # bổ sung
      confidence: float = 1.0
      evidence_items: List[Evidence] = []
      fixable: bool = True      # toolbox 5 thao tác có sửa được không
      contraindications: List[str] = []   # ví dụ "no_sharpen"

  class DiagnosisReport(BaseModel):   # bổ sung
      rejected: List[Defect] = []       # để giải thích và benchmark
      ask_user: List[Defect] = []       # clarify dùng để sinh câu hỏi
      measurement: Optional[MeasurementReport] = None
      region_stats: Dict[str, RegionStats] = {}
  ```

- `defects` chỉ chứa **confirmed** → planner, knowledge retrieval, preserve guard không phải đổi
  logic chọn.
- **Defect mới** (cần đồng bộ `DEFECT_TYPES`, schema perceive, knowledge cards, `docs/interfaces.md`):
  `motion_blur`, `jpeg_artifacts`, `haze`, `highlight_clipping`. Các lỗi toolbox không sửa được
  (`jpeg_artifacts`, `motion_blur` nặng) có `fixable=False`: báo cho người dùng, sinh
  contraindication, không lập action.
- **Planner** đọc `contraindications`: `no_sharpen` → loại sharpen; `denoise_before_sharpen` đã có
  sẵn qua thứ tự ưu tiên.
- **Clarify** dùng `ask_user` để sinh câu hỏi tất định (giữ yêu cầu tất định trước `interrupt()`).

**Test:** mỗi dòng bảng quyết định một test; VLM khai nhiễu trên ảnh sạch → rejected; ảnh hoàng hôn
+ VLM preserve warm_tone → preserved; contraindication loại sharpen trong plan.

---

### Phase 7 — Vòng ≥ 2: phân tích khác biệt (S6)

- `measurement/diff.py`: `compare(original: MeasurementResult, current: MeasurementResult,
  regions) -> DiffReport`.
  - **Trạng thái từng defect cũ** theo đúng vùng: resolved / improved / unchanged / worse.
  - **Tác dụng phụ do xử lý:**
    - Halo: overshoot quanh cạnh mạnh (so profile cạnh trước/sau).
    - Nhiễu vùng tối tăng: tỉ số σ(I) ở bin tối sau/trước (gamma, CLAHE).
    - Banding: số mức xám bị bỏ trống trong vùng gradient mịn tăng.
    - Cháy kênh màu tăng; bão hòa vượt ngưỡng.
- `perceive_node` vòng ≥ 2: đo lại (S1) + diff; chỉ gọi VLM cho các defect ngữ nghĩa còn mở hoặc
  tác dụng phụ cần phân xử.
- `decide_node`: tác dụng phụ severity ≥ 2 ⇒ coi như suy thoái ⇒ rollback (bổ sung cho
  `_is_degraded_real`).

**Test:** sharpen mạnh ảnh synthetic → phát hiện halo; gamma 2.0 trên ảnh tối nhiễu → phát hiện
nhiễu vùng tối tăng; xử lý đúng → defect resolved.

---

### Phase 8 — Tài liệu và đồng bộ (làm dần theo từng phase)

- `docs/decisions.md`: **ADR-005** "Measurement-first diagnosis with deterministic fusion".
- `docs/interfaces.md`: MeasurementReport, RegionStats, DiagnosisReport v2, `segment_box`,
  `refine_bbox_mask`, `resolve_subject_mask`, defect mới.
- `CLAUDE.md`: luồng perceive mới, quy tắc hợp nhất, cách mock nhiều lần gọi Gemini (perceive ×N,
  verify, plan) trong test.
- `SYSTEM_PROMPT`, `PERCEIVE_PROMPT`, knowledge cards (`src/agent/knowledge/cards/*.yaml`) cho
  defect mới; `test_agent_knowledge.py` phải xanh.

---

## 5. Phân công theo module

| Phase | Module | Chủ sở hữu | Ghi chú |
|---|---|---|---|
| P0 | scripts, tests, data | Person 4 | Không đụng `src/` |
| P1, P2 | `src/analyzer_evaluator/` | Person 1 | Cần thống nhất trước khi làm |
| P3 | M1 + `src/api/` + `frontend/` | Person 1 + Person 4 | |
| P4 | `src/region_engine/` + M4 | Person 2 + Person 4 | Interface mới: `segment_box`, `refine_bbox_mask`, `resolve_subject_mask` |
| P5–P7 | `src/agent/` | Person 4 | |

---

## 6. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Bộ ảnh thật nhỏ (26) → overfit | Hiệu chỉnh trên synthetic, ảnh thật chỉ để kiểm định; bổ sung ảnh `intentional_style` |
| Synthetic khác ảnh thật (nhiễu cảm biến, demosaic, nén) | Dùng nhiễu Poisson-Gaussian + nhiễu chroma + JPEG; kiểm định chéo trên ảnh thật có EXIF ISO cao |
| Độ trễ: perceive ×3 + verify + plan, ảnh full-res | Chấp nhận ở giai đoạn này; đo và ghi lại; tối ưu sau (giảm N, thu nhỏ ảnh toàn khung vì đã có crop) |
| Bbox của Gemini lệch | DINO tinh chỉnh bbox, SAM sửa biên; vùng không phân giải được thì đo toàn ảnh và hạ confidence |
| DINO báo nhầm vùng không tồn tại | Chỉ chạy DINO cho vùng VLM đã liệt kê; yêu cầu khớp bbox Gemini; loại bbox gần toàn khung |
| Máy khác/CI không có model | Fallback GrabCut/quadrant, ghi `mask_source`; test `real_model` chạy local |
| Thay đổi schema làm hỏng API/UI/memory | Mọi trường mới có default; `defects` vẫn là danh sách confirmed; test API hiện có phải xanh |
| Liu 2013 cài sai | Kiểm chứng với σ biết trước trên synthetic trước khi tích hợp; giữ Immerkær để so |
| Phối hợp nhiều module | Thống nhất interface trong `docs/interfaces.md` trước khi code P1/P4 |

---

## 7. Quyết định

**Đã chốt (D2–D4: 2026-10-23, D1: 2026-10-24):**

- **D2 — Phạm vi lỗi:** chẩn đoán đầy đủ, kể cả lỗi toolbox không sửa được (`motion_blur`,
  `jpeg_artifacts`, `haze`, `highlight_clipping` với `fixable=False`: báo người dùng + sinh
  contraindication, không lập action).
- **D3 — Self-consistency:** N = 3 lần quan sát mở, giữ defect xuất hiện ≥ 2/3.
- **D4 — Ngưỡng `TechnicalMetrics.*_level`:** thay bằng ngưỡng hiệu chỉnh ngay trong P2 (một nguồn
  ngưỡng duy nhất); P2 phải chạy lại benchmark pipeline để kiểm tra quyết định SHIP/RE_PROCESS.
- **D1 — Phân vùng (chốt 2026-10-24):** SAM là đường chính. VLM liệt kê vùng kèm bbox → DINO
  xác nhận/tinh chỉnh → MobileSAM → cache theo ảnh gốc; DINO không khớp → bbox Gemini → MobileSAM;
  không có model → bbox Gemini → GrabCut 1024 px; không có API → quadrant. Chi tiết ở Phase 4.
  M2-INT-07 đã chạy với model thật và pass. Lý do không lấy GrabCut làm chính: tách theo màu nên
  sai biên ở chủ thể ngược sáng, và đo thực tế còn chậm hơn SAM.

  **Số đo thật (2026-10-24, Ryzen 7 8845HS, torch 2.12 CPU 8 luồng, 8 ảnh benchmark ≤1600 px).**
  Asset ghim: DINO `IDEA-Research/grounding-dino-tiny@a2bb814d` (659 MB, safetensors sha256
  `1a2412ef…42f3`), MobileSAM `ChaoningZhang/MobileSAM@f706ad9c` (`mobile_sam.pt` 39 MB, sha256
  `6dbb9052…6c2f`). Test real_model `test_prompt_segmentation_model.py` pass.

  | Bước | Đo được |
  |---|---|
  | Import + nạp model | 1.8 s + 11.3 s; RSS 37 → 1232 MB |
  | DINO, 1 prompt / lần gọi | 4.3–5.2 s |
  | DINO, gộp prompt (`"person . sky ."`) | 4.3–5.2 s, nhưng **mất nhãn** ở 4/8 ảnh |
  | MobileSAM encoder | 0.50–0.61 s / ảnh |
  | MobileSAM decoder | 36–51 ms / bbox |
  | GrabCut 5 vòng, ảnh 1024 px | 0.9–16.8 s / bbox (bbox lớn chậm) |
  | GrabCut, ảnh gốc ≤1600 px | 1.0–31.7 s / bbox |
  | RSS đỉnh | 3.16 GB |
  | IoU mask SAM vs GrabCut cùng bbox | 0.25–0.73 |

  Quan sát chất lượng: mask người của SAM bám biên tốt (kể cả ca ngược sáng `backlit_02`, nhóm
  đông người). DINO **báo nhầm "sky"** khi ảnh không có trời (bbox gần toàn khung ở `backlit_02`,
  phông xanh ở `multi_defect_01`) và bắt người trên poster (`low_light_01`). Hệ quả cho P4: không
  gộp prompt; danh sách vùng phải do Perceive (VLM) quyết định có tồn tại, DINO/SAM chỉ định vị;
  bbox DINO gần toàn khung coi là trượt; cache mask theo ảnh gốc (DINO ~4.5 s/vùng × 3 vòng).
  GrabCut không rẻ hơn SAM — chỉ giữ làm dự phòng khi thiếu model.

**Còn mở:**

- Có giữ ảnh toàn khung full-res khi đã gửi crop 100% hay không (quyết định sau khi đo ở P5).
