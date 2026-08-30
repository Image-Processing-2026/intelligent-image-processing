# Intelligent Image Processing — Module Handbook & Task Roadmap

Tài liệu này dành cho **toàn bộ 4 thành viên trong nhóm**. Mỗi người đọc phần Module của mình sẽ hiểu rõ:
- Vai trò module trong hệ thống tổng thể.
- Dữ liệu đầu vào nhận từ ai, đầu ra giao cho ai.
- Thuật toán Xử lý ảnh cần cài đặt.
- Danh sách công việc cụ thể cần hoàn thành.
- Các cạm bẫy kỹ thuật cần tránh.

> **Cập nhật tiến độ:** Đánh dấu `[ ]` → `[/]` (đang làm) → `[x]` (hoàn thành) khi phát triển.

---

## 🔁 Tổng quan Chu trình Hệ thống

```
Image → [Module 1: Analyze] → [VLM: Diagnose] → [Module 2: Locate Regions] → [Module 4: Plan]
                                                                                      │
                                              ┌──────────────────────────────────────────┘
                                              ▼
                                   [Module 3: Process] → [Module 1: Evaluate] → [Module 4: Decide]
                                                                                      │
                                                                   ┌──────────────────┴──────────────────┐
                                                              Đạt chất lượng                       Chưa đạt
                                                                   │                                    │
                                                                   ▼                                    ▼
                                                                 SHIP                         Re-Plan (tối đa 3 vòng)
```

**Triết lý cốt lõi:** AI chỉ là bộ não chẩn đoán & ra quyết định. Mọi thao tác biến đổi pixel đều do thuật toán Xử lý ảnh kinh điển (OpenCV, scikit-image) thực hiện — đây là phần chấm điểm chính của môn học.

---

## 👥 Bảng Phân công & Liên kết Module

| Module | Vai trò trong hệ thống | Ẩn dụ | Assignee | Thư mục |
|---|---|---|---|---|
| **Module 1** | Đo lường chỉ số kỹ thuật & Đánh giá chất lượng | 🔬 **Máy đo / KCS** | **Person 1** | `src/analyzer_evaluator/` |
| **Module 2** | Định vị vùng ảnh & Tạo mặt nạ mềm | 👁️ **Mắt** | **Person 2** | `src/region_engine/` |
| **Module 3** | Thực thi thuật toán xử lý ảnh kinh điển | ✋ **Tay** | **Person 3** | `src/processing_engine/` |
| **Module 4** | Điều phối AI Agent, VLM & Backend | 🧠 **Não** | **Person 4** | `src/agent/`, `src/api/`, `frontend/` |

### Sơ đồ luồng dữ liệu giữa các Module

```
Person 4 (Agent) ─── gọi ──→ Person 1 (Analyzer)    → nhận TechnicalMetrics
Person 4 (Agent) ─── gọi ──→ Person 2 (Region)      → nhận Soft-Mask float32
Person 4 (Agent) ─── gọi ──→ Person 3 (Processing)  → nhận Ảnh đã xử lý
Person 4 (Agent) ─── gọi ──→ Person 1 (Evaluator)   → nhận EvaluationResult
Person 3 (Processing) ── dùng ──→ Person 2 (mask_utils.blend_regions) → hòa trộn alpha
```

---
---

## 📋 Module 1: Image Analyzer & Evaluator — 🔬 Máy đo & KCS

**Assignee:** Person 1
**Thư mục:** `src/analyzer_evaluator/`
**File chính:** `analyzer.py`, `reference_eval.py`, `no_reference_eval.py`

---

### 1.1 Vai trò trong hệ thống

Module này chạy ở **hai thời điểm** trong chu trình:
1. **Đầu vòng lặp (Analyze):** Đo đạc các chỉ số kỹ thuật ban đầu của ảnh (độ sáng, tương phản, nhiễu, mờ, màu sắc) để Agent biết ảnh bị vấn đề gì.
2. **Cuối vòng lặp (Evaluate):** Đánh giá chất lượng ảnh sau xử lý để Agent quyết định SHIP hay RE-PROCESS.

---

### 1.2 Đầu vào / Đầu ra (Input → Output)

| Hàm | Nhận từ | Input | Output | Giao cho |
|---|---|---|---|---|
| `analyze_image(image)` | Module 4 (Agent) | Ảnh RGB `np.uint8` shape `(H, W, 3)` | Object `TechnicalMetrics` | Module 4 (Agent → VLM) |
| `evaluate_reference(current, ground_truth)` | Module 4 (Agent) | 2 ảnh RGB `np.uint8` | `{"psnr": float, "ssim": float, "mse": float}` | Module 4 (Agent: decide) |
| `evaluate_no_reference(current, previous?)` | Module 4 (Agent) | 1–2 ảnh RGB `np.uint8` | `{"brisque": float, "delta_contrast": float, ...}` | Module 4 (Agent: decide) |

**Schema dữ liệu chính — `TechnicalMetrics`:**
```python
class TechnicalMetrics(BaseModel):
    brightness_mean: float          # Độ sáng trung bình [0, 255]
    brightness_level: str           # "underexposed" | "normal" | "overexposed"
    contrast_std: float             # Độ lệch chuẩn cường độ
    contrast_level: str             # "low" | "normal" | "high"
    noise_variance: float           # Phương sai nhiễu ước lượng
    noise_level: str                # "clean" | "low" | "medium" | "severe"
    sharpness_laplacian_var: float  # Phương sai toán tử Laplacian
    blur_level: str                 # "sharp" | "mild_blur" | "severe_blur"
    color_cast: str                 # "warm" | "cool" | "greenish" | "none"
    histogram_stats: dict           # Thống kê histogram
```

---

### 1.3 Thuật toán Xử lý ảnh cần cài đặt

#### A. Phân tích ảnh (`analyzer.py`)

| Chỉ số | Thuật toán / Công thức | Thư viện |
|---|---|---|
| **Độ sáng (Brightness)** | Chuyển sang ảnh xám, tính `np.mean(gray)`. Phân loại theo ngưỡng (<70 = tối, >185 = chói). | NumPy |
| **Độ tương phản (Contrast)** | Tính `np.std(gray)` — độ lệch chuẩn mức xám. Thấp < 40, Cao > 80. | NumPy |
| **Độ mờ / Sắc nét (Blur/Sharpness)** | Toán tử vi phân bậc hai **Laplacian** $\nabla^2 f$: `cv2.Laplacian(gray, CV_64F).var()`. Phương sai thấp = ảnh mờ. | OpenCV |
| **Ước lượng nhiễu (Noise)** | Chênh lệch giữa ảnh gốc và ảnh qua **Median Filter** 3×3: `cv2.absdiff(gray, cv2.medianBlur(gray, 3))`. | OpenCV |
| **Phân bố Histogram** | `cv2.calcHist()` cho 3 kênh RGB, tính min/max/skewness. | OpenCV |
| **Ám màu (Color Cast)** | So sánh giá trị trung bình kênh R, G, B. Chênh lệch lớn = ám màu (warm/cool). | NumPy |

#### B. Đánh giá có ảnh gốc mẫu (`reference_eval.py`)

| Chỉ số | Công thức | Thư viện |
|---|---|---|
| **MSE** | $\text{MSE} = \frac{1}{N}\sum(I_{\text{ref}} - I_{\text{cur}})^2$ | NumPy |
| **PSNR** | $\text{PSNR} = 20 \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right)$ dB | scikit-image hoặc NumPy |
| **SSIM** | So sánh cấu trúc, độ sáng, độ tương phản giữa 2 ảnh. | scikit-image |

#### C. Đánh giá không có ảnh gốc (`no_reference_eval.py`)

| Chỉ số | Ý nghĩa | Thư viện |
|---|---|---|
| **BRISQUE** | Điểm chất lượng cảm nhận (thấp = tốt, 0–100) | `pyiqa` hoặc OpenCV |
| **NIQE** | Đánh giá tự nhiên dựa trên mô hình thống kê (thấp = tốt) | `pyiqa` |
| **Delta metrics** | So sánh chỉ số kỹ thuật giữa vòng lặp trước và sau (cải thiện hay suy giảm) | Tự tính |

---

### 1.4 Checklist Công việc

- [ ] **Phân tích kỹ thuật (`analyzer.py`)**
  - [ ] Implement đo độ sáng trung bình & phân loại (underexposed / normal / overexposed).
  - [ ] Implement đo độ tương phản bằng độ lệch chuẩn & dynamic range.
  - [ ] Implement ước lượng nhiễu bằng chênh lệch Median Filter.
  - [ ] Implement đo độ mờ bằng phương sai Laplacian & gradient Tenengrad.
  - [ ] Implement phân tích histogram RGB/HSV & phát hiện ám màu.
- [ ] **Đánh giá có Ground-Truth (`reference_eval.py`)**
  - [ ] Implement tính PSNR, SSIM, MSE bằng `scikit-image`.
  - [ ] Xây dựng bộ tải synthetic dataset (load cặp ảnh sạch + ảnh suy giảm).
- [ ] **Đánh giá không có Ground-Truth (`no_reference_eval.py`)**
  - [ ] Implement tính điểm BRISQUE / NIQE (bằng `pyiqa` hoặc OpenCV).
  - [ ] Implement so sánh delta metrics giữa các vòng lặp (theo dõi cải thiện/suy giảm).
- [ ] **Unit Tests (`tests/unit/test_analyzer_evaluator.py`)**

---

### 1.5 Cạm bẫy & Lưu ý quan trọng

> [!CAUTION]
> **KHÔNG ĐƯỢC tính PSNR/SSIM trên ảnh thực tế từ người dùng.** PSNR/SSIM yêu cầu ảnh gốc mẫu ground-truth. Ảnh thực tế (ảnh cũ hỏng, ảnh chụp mờ) không có ground-truth → kết quả PSNR/SSIM sẽ vô nghĩa. Chỉ dùng PSNR/SSIM cho tập synthetic test (`data/synthetic/`).

> [!IMPORTANT]
> Đảm bảo tất cả hàm xử lý cả ảnh grayscale `(H, W)` lẫn ảnh màu `(H, W, 3)` đúng cách, không bị crash khi nhận ảnh 1 kênh.

---
---

## 📋 Module 2: Region Engine & Mask Synthesis — 👁️ Mắt

**Assignee:** Person 2
**Thư mục:** `src/region_engine/`
**File chính:** `detector.py`, `face_detector.py`, `spatial.py`, `mask_utils.py`

---

### 2.1 Vai trò trong hệ thống

Module này đóng vai **"con mắt"** của hệ thống — nó trả lời câu hỏi **"Ở đâu?"**:
- Agent nói: *"Cần xử lý vùng bầu trời"* → Module 2 trả về một **mặt nạ mềm (soft-mask)** chỉ ra chính xác vùng "bầu trời" ở đâu trên ảnh.
- Module 3 (Processing Engine) nhận mặt nạ này để chỉ áp dụng thuật toán lên vùng đó, không ảnh hưởng phần còn lại.

**Đây là module khó nhất về mặt kỹ thuật** vì phải xử lý biên mặt nạ mượt mà, tránh tạo vết cắt (seam artifacts).

---

### 2.2 Đầu vào / Đầu ra (Input → Output)

| Hàm | Nhận từ | Input | Output | Giao cho |
|---|---|---|---|---|
| `segment_by_prompt(image, text_prompt)` | Module 4 (Agent/Executor) | Ảnh RGB `uint8` + chuỗi VD `"sky"` | Soft-mask `float32 [0.0, 1.0]` shape `(H, W)` | Module 3 (Processing) & Module 4 |
| `detect_faces(image)` | Module 4 (Agent/Executor) | Ảnh RGB `uint8` | List các soft-mask khuôn mặt | Module 3 (Processing) & Module 4 |
| `create_soft_mask(binary_mask, feather_radius)` | Nội bộ Module 2 | Mặt nạ nhị phân `uint8` 0/255 | Soft-mask `float32 [0.0, 1.0]` | Hàm khác trong Module 2 |
| `blend_regions(original, processed, soft_mask)` | Module 3 (Processing Engine) | 2 ảnh RGB `uint8` + soft-mask | Ảnh hòa trộn RGB `uint8` | Module 4 (Agent) |

**Quy ước mặt nạ bắt buộc:**
- Kiểu dữ liệu: `np.float32`
- Khoảng giá trị: `[0.0, 1.0]` liên tục (không phải `0/255` nhị phân)
- Kích thước: `(H, W)` khớp chính xác với ảnh đầu vào

---

### 2.3 Thuật toán Xử lý ảnh cần cài đặt

#### A. Nhận diện khuôn mặt (`face_detector.py`)

| Kỹ thuật | Chi tiết | Thư viện |
|---|---|---|
| **MediaPipe Face Detection** | Mô hình nhẹ tối ưu cho CPU. Trả về bounding box tọa độ tương đối (relative). Mở rộng vùng mặt thêm ~15% để bao trán & cằm. | `mediapipe` |
| **Chuyển đổi BBox → Mask** | Tạo mặt nạ nhị phân `uint8` từ tọa độ bounding box, rồi gọi `create_soft_mask()` để làm mềm biên. | NumPy + OpenCV |

#### B. Phân đoạn ngữ nghĩa (`detector.py`)

| Kỹ thuật | Chi tiết | Thư viện |
|---|---|---|
| **GroundingDINO** | Nhận chuỗi văn bản (VD: "sky"), trả về bounding box chứa đối tượng đó. | `groundingdino` |
| **MobileSAM / SAM2** | Nhận bounding box từ GroundingDINO, tạo mặt nạ phân đoạn pixel-level chính xác. | `mobile_sam` |
| **Spatial Fallback** | Khi chưa load model SAM: dùng heuristic đơn giản (VD: "sky" → nửa trên ảnh). | NumPy |

#### C. Tạo mặt nạ hình học (`spatial.py`)

| Kỹ thuật | Chi tiết |
|---|---|
| **Bounding Box Mask** | Tạo mask hình chữ nhật từ tọa độ (xmin, ymin, xmax, ymax). |
| **Quadrant Mask** | Tạo mask theo phần tư: "top", "bottom", "left", "right", "center". |

#### D. Làm mịn biên & Hòa trộn (`mask_utils.py`) — ⭐ PHẦN QUAN TRỌNG NHẤT

| Kỹ thuật | Công thức / Chi tiết | Thư viện |
|---|---|---|
| **Gaussian Edge Feathering** | Áp dụng `cv2.GaussianBlur` lên mặt nạ nhị phân để biến đổi biên cứng 0/1 thành gradient mượt 0.0→1.0. Kernel size phải là số lẻ. | OpenCV |
| **Alpha Compositing** | $I_{\text{out}} = M \odot I_{\text{processed}} + (1 - M) \odot I_{\text{original}}$ — Phép hòa trộn điểm ảnh theo trọng số mặt nạ mềm. Thực hiện trên kiểu `float32` rồi cast về `uint8`. | NumPy |

---

### 2.4 Checklist Công việc

- [ ] **Nhận diện khuôn mặt (`face_detector.py`)**
  - [ ] Tích hợp MediaPipe Face Detection chạy trên CPU/iGPU.
  - [ ] Chuyển đổi bounding box thành mặt nạ nhị phân, mở rộng vùng ~15%.
  - [ ] Gọi `create_soft_mask()` để làm mềm biên mặt nạ khuôn mặt.
- [ ] **Phân đoạn ngữ nghĩa (`detector.py`)**
  - [ ] Tích hợp GroundingDINO + MobileSAM/SAM2 cho phân đoạn theo câu lệnh văn bản.
  - [ ] Đảm bảo chạy mượt trên CPU/iGPU (AMD Radeon, không có CUDA).
  - [ ] Xây dựng heuristic fallback khi chưa nạp model SAM.
- [ ] **Mặt nạ hình học (`spatial.py`)**
  - [ ] Implement tạo mask bounding box và mask phần tư (quadrant).
- [ ] **Làm mịn biên & Hòa trộn (`mask_utils.py`)**
  - [ ] Implement Gaussian feathering cho mặt nạ nhị phân → soft-mask `float32 [0.0, 1.0]`.
  - [ ] Implement hàm `blend_regions()` hòa trộn alpha 3 kênh màu.
- [ ] **Unit Tests (`tests/unit/test_region_engine.py`)**

---

### 2.5 Cạm bẫy & Lưu ý quan trọng

> [!CAUTION]
> **Mặt nạ trả về PHẢI là `float32` trong khoảng `[0.0, 1.0]`.** Nếu trả về `uint8 0/255` (nhị phân cứng), Module 3 sẽ tạo ra vết cắt rõ ràng ở biên vùng xử lý — đây là lỗi nghiêm trọng nhất của hệ thống.

> [!WARNING]
> **Tất cả model phải chạy được trên CPU.** Máy phát triển chính dùng AMD Radeon iGPU (không có CUDA). Dùng MobileSAM thay vì SAM-ViT-H, dùng MediaPipe thay vì dlib/MTCNN. Chấp nhận 1–3 giây latency mỗi ảnh.

> [!TIP]
> Kiểm thử mặt nạ bằng cách visualize: `plt.imshow(soft_mask, cmap='gray')` — biên phải chuyển gradient mượt từ đen sang trắng, không có đường cắt sắc nét.

---
---

## 📋 Module 3: Classical Image Processing Engine — ✋ Tay

**Assignee:** Person 3
**Thư mục:** `src/processing_engine/`
**File chính:** `base.py`, `denoise.py`, `exposure_contrast.py`, `sharpen.py`, `color.py`

---

### 3.1 Vai trò trong hệ thống

Module này là **"đôi tay"** thực thi — nơi chứa toàn bộ thuật toán Xử lý ảnh kinh điển được chấm điểm trong môn học. Module này **KHÔNG** tự quyết định xử lý gì — nó chỉ nhận lệnh từ Module 4 (Agent) và thực thi đúng thuật toán được yêu cầu lên đúng vùng ảnh được chỉ định.

---

### 3.2 Đầu vào / Đầu ra (Input → Output)

**Mọi hàm xử lý ảnh đều tuân thủ chữ ký thống nhất:**

```python
def apply_<operation>(
    image: np.ndarray,              # Ảnh RGB uint8 (H, W, 3)
    mask: np.ndarray | None = None, # Soft-mask float32 [0.0, 1.0] (H, W) — từ Module 2
    **params                        # Siêu tham số thuật toán
) -> np.ndarray:                    # Ảnh RGB uint8 đã xử lý & hòa trộn
```

| Hàm | Tham số chính | Nhận mask từ |
|---|---|---|
| `apply_denoise(image, mask, method, strength)` | `method`: "gaussian"/"median"/"bilateral"/"nlm", `strength`: 0.1–2.0 | Module 2 |
| `apply_gamma(image, mask, gamma)` | `gamma`: 0.5–2.5 (>1 sáng hơn, <1 tối hơn) | Module 2 |
| `apply_clahe(image, mask, clip_limit, tile_grid_size)` | `clip_limit`: 1.0–4.0 | Module 2 |
| `apply_sharpen(image, mask, method, amount)` | `method`: "unsharp_mask"/"laplacian", `amount`: 0.2–2.0 | Module 2 |
| `apply_color_balance(image, mask, saturation_scale, temperature_shift)` | `saturation_scale`: 0.5–1.5, `temperature_shift`: -1.0–1.0 | Module 2 |

**Cơ chế hoạt động bên trong mỗi hàm:**
1. Áp dụng thuật toán lên **toàn bộ ảnh** (vì nhiều bộ lọc không thể áp dụng trực tiếp lên vùng cắt mà không gây artifact ở biên).
2. Nếu có `mask`: gọi `blend_regions(original, processed, mask)` từ Module 2 để hòa trộn alpha.
3. Nếu `mask = None`: trả về ảnh đã xử lý toàn cục.

---

### 3.3 Thuật toán Xử lý ảnh cần cài đặt

#### A. Khử nhiễu (`denoise.py`)

| Thuật toán | Nguyên lý | Công thức / API | Ưu-Nhược |
|---|---|---|---|
| **Gaussian Blur** | Tích chập ảnh với kernel Gauss $G_\sigma$ | `cv2.GaussianBlur(img, (k,k), σ)` | Nhanh, mờ đều — mất chi tiết cạnh |
| **Median Filter** | Thay mỗi pixel bằng trung vị vùng lân cận | `cv2.medianBlur(img, k)` | Hiệu quả với nhiễu muối tiêu (salt-and-pepper) |
| **Bilateral Filter** | Lọc theo cả khoảng cách không gian lẫn cường độ — bảo toàn cạnh biên | `cv2.bilateralFilter(img, d, σ_color, σ_space)` | Chậm hơn nhưng giữ được cạnh nét |
| **Non-Local Means (NLM)** | So sánh mẫu lặp lại (patch matching) trong toàn ảnh | `cv2.fastNlMeansDenoisingColored(img, h, hColor, templateWindowSize, searchWindowSize)` | Chất lượng cao nhất nhưng chậm nhất |

#### B. Hiệu chỉnh Độ sáng & Tương phản (`exposure_contrast.py`)

| Thuật toán | Nguyên lý | Công thức | Không gian màu |
|---|---|---|---|
| **Gamma Correction** | Biến đổi phi tuyến tính: tăng/giảm sáng tổng thể | $I_{\text{out}} = 255 \cdot \left(\frac{I_{\text{in}}}{255}\right)^{1/\gamma}$ — Dùng LUT để tối ưu tốc độ | RGB |
| **CLAHE** | Cân bằng lược đồ thích ứng cục bộ có giới hạn clip (chống over-enhancement). Chia ảnh thành tile nhỏ, equalize từng tile rồi nội suy biên. | `cv2.createCLAHE(clipLimit, tileGridSize).apply(L)` | **LAB** — chỉ xử lý kênh L (Luminance), giữ nguyên A, B để không biến dạng màu |

#### C. Làm nét (`sharpen.py`)

| Thuật toán | Nguyên lý | Công thức |
|---|---|---|
| **Unsharp Masking** | Cộng thêm phần chi tiết tần số cao: lấy ảnh gốc trừ ảnh mờ, nhân hệ số, rồi cộng lại | $I_{\text{sharp}} = I + \alpha \cdot (I - G_\sigma * I)$ |
| **Laplacian Sharpening** | Dùng kernel vi phân bậc hai 3×3 để tăng cường biên | Kernel: $\begin{bmatrix} 0 & -1 & 0 \\ -1 & 4+\alpha & -1 \\ 0 & -1 & 0 \end{bmatrix}$ |

#### D. Hiệu chỉnh Màu sắc (`color.py`)

| Thuật toán | Nguyên lý | Không gian màu |
|---|---|---|
| **Gray World White Balance** | Giả định trung bình 3 kênh RGB phải bằng nhau → scale từng kênh | RGB |
| **HSV Saturation Adjustment** | Điều chỉnh kênh S (Saturation) trong không gian HSV để tăng/giảm độ bão hòa | HSV |
| **Temperature Shift** | Tăng kênh R + giảm kênh B (ấm hơn) hoặc ngược lại (lạnh hơn) | RGB |

---

### 3.4 Checklist Công việc

- [ ] **Wrapper hòa trộn (`base.py`)**
  - [ ] Implement `apply_region_op(image, mask, op_func, **kwargs)` — wrapper tự động blend.
- [ ] **Khử nhiễu (`denoise.py`)**
  - [ ] Implement Gaussian, Median, Bilateral, NLM với tham số `strength` điều khiển cường độ.
- [ ] **Độ sáng & Tương phản (`exposure_contrast.py`)**
  - [ ] Implement Gamma Correction bằng LUT (Lookup Table).
  - [ ] Implement CLAHE trong không gian màu LAB (chỉ xử lý kênh L).
- [ ] **Làm nét (`sharpen.py`)**
  - [ ] Implement Unsharp Masking với `amount` có thể điều chỉnh.
  - [ ] Implement Laplacian Sharpening.
- [ ] **Hiệu chỉnh màu (`color.py`)**
  - [ ] Implement cân bằng trắng Gray World / White Patch.
  - [ ] Implement điều chỉnh Saturation trong HSV và Temperature Shift.
- [ ] **Unit Tests (`tests/unit/test_processing_engine.py`)**

---

### 3.5 Cạm bẫy & Lưu ý quan trọng

> [!CAUTION]
> **CLAHE phải xử lý trong không gian LAB, chỉ trên kênh L.** Nếu áp dụng CLAHE trực tiếp lên RGB, màu sắc sẽ bị méo nghiêm trọng (VD: mặt người chuyển sang tông xanh lá).

> [!WARNING]
> **Khi có mask, KHÔNG được cắt ảnh theo mask rồi xử lý phần cắt.** Phải xử lý toàn bộ ảnh rồi dùng `blend_regions()` để hòa trộn. Lý do: Kernel convolution (Gaussian, Median, CLAHE) cần thông tin pixel lân cận — nếu cắt ảnh trước sẽ tạo artifact ở biên vùng cắt.

> [!TIP]
> Luôn `np.clip(result, 0, 255).astype(np.uint8)` sau mỗi phép tính toán số thực để tránh tràn giá trị.

---
---

## 📋 Module 4: AI Agent, VLM & Backend — 🧠 Não

**Assignee:** Person 4
**Thư mục:** `src/agent/`, `src/api/`, `frontend/`
**File chính:** `state.py`, `graph.py`, `vlm_diagnostician.py`, `planner.py`, `executor.py`, `src/api/main.py`, `frontend/app.py`

---

### 4.1 Vai trò trong hệ thống

Module này là **"bộ não"** điều phối toàn bộ chu trình khép kín. Trách nhiệm:
1. **Hiểu:** Nhận ảnh đầu vào, gọi Module 1 đo chỉ số, gọi VLM Gemini chẩn đoán bệnh.
2. **Lên kế hoạch:** Tạo danh sách thao tác (denoise vùng nào, gamma vùng nào, thứ tự ra sao).
3. **Thực thi:** Gọi Module 2 tạo mask, gọi Module 3 xử lý ảnh theo kế hoạch.
4. **Đánh giá:** Gọi Module 1 đo lại chất lượng sau xử lý.
5. **Quyết định:** SHIP (xuất ảnh) hay RE-PROCESS (lặp lại với kế hoạch mới).
6. **Cung cấp API & UI** cho người dùng tương tác.

---

### 4.2 Đầu vào / Đầu ra (Input → Output)

| Thành phần | Input | Output |
|---|---|---|
| **LangGraph Pipeline** | Ảnh RGB `uint8` (+ ground-truth tùy chọn) | `DoctorState` chứa ảnh kết quả, lịch sử xử lý, quyết định cuối cùng |
| **VLM Diagnostician** | Ảnh RGB + `TechnicalMetrics` (từ Module 1) | `TreatmentPlan` JSON (danh sách thao tác theo vùng) |
| **Plan Validator** | `TreatmentPlan` thô từ VLM | `TreatmentPlan` đã lọc (chỉ giữ tool hợp lệ, sắp xếp đúng thứ tự) |
| **Executor** | `TreatmentPlan` đã validate + Ảnh hiện tại | Ảnh sau khi xử lý tuần tự tất cả actions |
| **FastAPI** | HTTP request + file ảnh | JSON response + ảnh base64 |
| **Gradio UI** | Ảnh tải lên từ browser | Ảnh kết quả + log chẩn đoán hiển thị trực quan |

**Schema kế hoạch điều trị — `TreatmentPlan`:**
```json
{
  "iteration": 1,
  "reasoning": "Bầu trời bị chói sáng, nền bị nhiễu Gaussian mức trung bình, khuôn mặt tối...",
  "actions": [
    { "region_id": "background", "target_prompt": "background", "detected_issue": "noise",
      "operation": "denoise", "parameters": {"method": "bilateral", "strength": 1.0}, "order": 10 },
    { "region_id": "sky", "target_prompt": "sky", "detected_issue": "overexposed",
      "operation": "gamma_correct", "parameters": {"gamma": 0.8}, "order": 20 },
    { "region_id": "face", "target_prompt": "face", "detected_issue": "underexposed",
      "operation": "gamma_correct", "parameters": {"gamma": 1.3}, "order": 21 }
  ]
}
```

**Danh sách Toolbox cố định** (Agent chỉ được gọi các tool này, không được phát minh tool mới):
```
analyze_image()     → Module 1
detect_regions()    → Module 2
create_soft_mask()  → Module 2
denoise()           → Module 3
gamma_correct()     → Module 3
clahe()             → Module 3
sharpen()           → Module 3
color_correct()     → Module 3
evaluate_quality()  → Module 1
```

---

### 4.3 Kỹ thuật cần cài đặt

#### A. LangGraph State Machine (`state.py`, `graph.py`)

| Node | Gọi Module | Vai trò |
|---|---|---|
| `analyze_node` | Module 1: `analyze_image()` | Đo chỉ số kỹ thuật ảnh hiện tại |
| `diagnose_and_plan_node` | Gemini VLM + `validate_and_sort_plan()` | Chẩn đoán + sinh kế hoạch JSON + validate |
| `process_node` | Module 2 + Module 3 qua `execute_plan()` | Tạo mask cho từng vùng rồi xử lý tuần tự |
| `evaluate_node` | Module 1: `evaluate_reference()` hoặc `evaluate_no_reference()` | Đo chất lượng sau xử lý |
| `decide_node` | Logic nội bộ | SHIP / RE_PROCESS / STOP_BEST_EFFORT |

**Điều kiện dừng (Stop conditions):**
- `SHIP`: Chất lượng đạt ngưỡng chấp nhận.
- `RE_PROCESS`: Chưa đạt nhưng còn dư số lần lặp.
- `STOP_BEST_EFFORT`: Đã đạt `MAX_ITERATIONS = 3` hoặc không cải thiện giữa 2 vòng lặp.

#### B. VLM Prompt Engineering (`vlm_diagnostician.py`)

| Yếu tố | Chi tiết |
|---|---|
| **Model** | Gemini 1.5 Flash (multimodal, chi phí thấp cho nhiều lần gọi trong loop) |
| **Input** | System prompt (vai trò, toolbox, ràng buộc thứ tự) + Ảnh PIL + Chỉ số kỹ thuật JSON |
| **Output** | JSON thuần `TreatmentPlan` — phải parse và validate trước khi dùng |
| **Fallback** | Nếu không có API key hoặc lỗi mạng: dùng rule-based heuristic (VD: noise cao → denoise, tối → gamma) |

#### C. Plan Validation (`planner.py`)

| Quy tắc | Lý do |
|---|---|
| Loại bỏ operation không nằm trong toolbox | Ngăn VLM "phát minh" tool không tồn tại |
| Sắp xếp: **denoise (10) → gamma (20) → clahe (30) → sharpen (40) → color (50)** | Khử nhiễu TRƯỚC làm nét (sharpening khuếch đại nhiễu nếu làm trước) |

---

### 4.4 Checklist Công việc

- [ ] **LangGraph State & Graph (`state.py`, `graph.py`)**
  - [ ] Định nghĩa `DoctorState` TypedDict (original image, current image, history, iteration, metrics).
  - [ ] Xây dựng các node: analyze → diagnose_and_plan → process → evaluate → decide.
  - [ ] Enforce bounded iteration (`MAX_ITERATIONS = 3`), prevent infinite loop.
  - [ ] Lưu lịch sử đầy đủ qua mỗi vòng (HistoryItem: plan, metrics before/after, decision).
- [ ] **VLM Chẩn đoán (`vlm_diagnostician.py`)**
  - [ ] Thiết kế system prompt bằng tiếng Việt cho Gemini (role, toolbox, constraints).
  - [ ] Implement gọi Gemini Multimodal API (gửi ảnh PIL + metrics JSON).
  - [ ] Parse JSON response thành `TreatmentPlan` schema.
  - [ ] Implement fallback rule-based khi không có API key.
- [ ] **Kiểm tra & Sắp xếp kế hoạch (`planner.py`, `executor.py`)**
  - [ ] Validate operation names chỉ nằm trong toolbox hợp lệ.
  - [ ] Sắp xếp theo thứ tự ưu tiên (denoise → gamma → clahe → sharpen → color).
  - [ ] Implement `execute_plan()`: lặp qua actions, tạo mask (Module 2), xử lý (Module 3).
- [ ] **FastAPI Backend (`src/api/`)**
  - [ ] Endpoint `/api/v1/health` — kiểm tra trạng thái.
  - [ ] Endpoint `/api/v1/diagnose` — chỉ phân tích & chẩn đoán (không xử lý).
  - [ ] Endpoint `/api/v1/process` — chạy toàn bộ pipeline khép kín.
- [ ] **Demo Frontend (`frontend/app.py`)**
  - [ ] Gradio UI: Upload ảnh, slider MAX_ITERATIONS, hiển thị Before/After, log chẩn đoán JSON.
- [ ] **Integration Tests (`tests/integration/test_pipeline_loop.py`)**

---

### 4.5 Cạm bẫy & Lưu ý quan trọng

> [!CAUTION]
> **Phải chặn vòng lặp bằng `MAX_ITERATIONS`.** Tuyệt đối không viết `while not perfect: process()` — VLM có thể không bao giờ hài lòng và chạy vô tận, tốn hết API quota.

> [!WARNING]
> **VLM có thể trả về operation không tồn tại** (VD: "super_resolution", "face_beautify"). Plan Validator phải loại bỏ chúng trước khi dispatch. Nếu không sẽ crash toàn pipeline.

> [!IMPORTANT]
> **Thứ tự xử lý rất quan trọng:** Luôn khử nhiễu TRƯỚC khi làm nét. Sharpening trước sẽ khuếch đại nhiễu, làm ảnh xấu đi nghiêm trọng. Validator phải ép thứ tự này dù VLM đề xuất khác.

---
---

## 🚫 Deferred Items — Không bắt đầu cho đến khi MVP hoàn thành

Các tính năng sau nằm trong `src/extensions/` và **KHÔNG ĐƯỢC import** vào pipeline chính:

- [ ] Deblurring: Wiener filter, Richardson-Lucy (`src/extensions/deblur.py`)
- [ ] Inpainting / Scratch removal: Telea, Navier-Stokes (`src/extensions/inpaint.py`)
- [ ] Super-resolution: SRCNN, Real-ESRGAN (`src/extensions/super_res.py`)

---

## 📌 Quy ước chung toàn nhóm

| Hạng mục | Quy ước |
|---|---|
| **Log messages & exceptions** | Viết bằng **tiếng Anh** |
| **Code comments & docstrings** | Viết bằng **tiếng Việt có dấu** (giải thích toán học, thuật toán) |
| **File `.ps1` (PowerShell)** | Lưu UTF-8 with BOM |
| **Type hints** | Python 3.10+ (`float \| None`, `tuple`, `dict`) |
| **Schemas giao tiếp** | Dùng Pydantic v2 `BaseModel` |
| **File cá nhân (prompts, notes)** | Đặt trong `personal/<tên_bạn>/` (đã gitignore) |
| **Tài liệu kỹ thuật dùng chung** | `docs/interfaces.md`, `docs/decisions.md` |
