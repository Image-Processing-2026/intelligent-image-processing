# Kế hoạch triển khai create_bbox_mask — Module 2

Ngày lập: 2026-09-21. Trạng thái: **đã triển khai, kiểm thử và nghiệm thu phần Module 2 trong phạm vi dependency hiện có**.

Hàm mục tiêu: `create_bbox_mask(image_shape, bbox, feather_radius=15)`.

Mục tiêu: chuyển bbox tọa độ pixel thành mask toàn ảnh `float32`, shape `(H, W)`, hữu hạn trong `[0, 1]`; dùng được trực tiếp với `blend_regions` và Module 3. Phạm vi không bao gồm detector, segmentation hoặc thay đổi thuật toán soft-mask đã nghiệm thu.

## 1. Nguồn trình bày thuật toán

Không cần một model hoặc thuật toán phát hiện đối tượng: bbox đã là input. Bài toán gồm raster hóa hình chữ nhật, lấy giao với miền ảnh và Gaussian feathering.

| Nguồn chính thức | Vai trò |
|---|---|
| [OpenCV — cv::Rect_](https://docs.opencv.org/4.x/d2/d44/classcv_1_1Rect__.html) | Nguồn chính cho quy ước ROI: chứa biên trái/trên, không chứa biên phải/dưới; phù hợp slicing NumPy |
| [scikit-image 0.25.2 — draw.rectangle](https://scikit-image.org/docs/0.25.x/api/skimage.draw.html#skimage.draw.rectangle) | Có ví dụ input và output dạng ma trận để lấy baseline công khai |
| [OpenCV — GaussianBlur](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html) | Tham chiếu Gaussian filter, kích thước kernel, sigma và border mode |
| [SciPy — gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html) | Oracle số độc lập cho phần feathering: cấu hình cùng sigma, radius và chế độ biên |

Nguồn được tra cứu ngày 2026-09-21. Đây là các tài liệu chính thức phù hợp trực tiếp; không khẳng định có một bài báo duy nhất định nghĩa toàn bộ API riêng của dự án. Quyết định báo lỗi bbox đảo và xử lý input sai là contract của dự án.

Lưu ý khi chuyển testcase: scikit-image dùng `(row, column)` và `end` **bao gồm** điểm cuối. API dự án dùng `(xmin, ymin, xmax, ymax)` với cận cuối **không bao gồm**. Đổi `end=(r,c)` thành `xmax=c+1, ymax=r+1`. Không dùng trực tiếp hai điểm cuối của hàm vẽ rectangle khác mà bỏ qua khác biệt này.

## 2. Triển khai thuật toán

### 2.1. Hiện trạng trong repository

- `src/region_engine/spatial.py`: đã tạo mask uint8, clip bốn tọa độ, tô 255 bằng slicing và gọi `create_soft_mask`.
- `image_shape[:2]` hiện vô tình chấp nhận cả `(H,W,3)`; thiếu validation tường minh cho shape và bbox.
- Bbox đảo hiện thành vùng rỗng theo hành vi slicing, chưa phân biệt lỗi input với vùng không giao ảnh.
- `src/region_engine/__init__.py` đã export hàm; tìm kiếm hiện chưa thấy caller khác trong `src` hoặc test riêng cho bbox.
- `tests/unit/test_region_engine.py` có test quadrant nhưng chưa kiểm thử bbox.
- `create_soft_mask` đã cố định Gaussian: tham số mang tên radius nhưng thực tế là kích thước kernel; giữ quy ước đó để tương thích.

### 2.2. Contract đề xuất

| Input/tình huống | Quy định |
|---|---|
| `image_shape` | Tuple/list hoặc ndarray 1D đúng hai phần tử nguyên `(H,W)`; H,W > 0 |
| `bbox` | Tuple/list hoặc ndarray 1D đúng bốn phần tử nguyên; nhận Python int và NumPy integer |
| Bool, float, string, complex | Không nhận làm kích thước/tọa độ; `TypeError`, kể cả float có giá trị nguyên |
| Sai số chiều/độ dài | `ValueError`; không nhận `(H,W,3)`, caller truyền `image.shape[:2]` |
| Tọa độ âm hoặc vượt ảnh | Cho phép; clip sau khi kiểm tra thứ tự tọa độ |
| `xmin > xmax` hoặc `ymin > ymax` | `ValueError` trước clipping; không tự đổi hai đầu |
| Hai đầu bằng nhau | Bbox không có diện tích: trả mask toàn 0 |
| Bbox đúng thứ tự nằm ngoài ảnh | Trả mask toàn 0 |
| Bbox bao toàn ảnh | Trả mask toàn 1, kể cả feathering |
| `feather_radius` | Integer không âm, loại bool; sai kiểu → `TypeError`, âm → `ValueError` |
| Output | Mảng mới float32 `(H,W)`, finite, `[0,1]`; không sửa input |

Không làm tròn tọa độ float ngầm. Detector tương lai phải chuyển bbox sang pixel nguyên theo contract riêng trước khi gọi. Khi validate ndarray, kiểm tra từng phần tử trước mọi ép kiểu có thể làm mất dấu bool hoặc biến float thành int. Không ép tọa độ Python integer rất lớn sang int32/int64 trước clipping.

Việc từ chối shape ba chiều và bbox đảo là thay đổi hành vi có chủ đích; ghi vào README/docstring. Kiểm tra lại caller khi triển khai.

### 2.3. Định nghĩa toán và thứ tự xử lý

Sau validation, đặt:

```text
x0 = max(0, min(W, xmin)); x1 = max(0, min(W, xmax))
y0 = max(0, min(H, ymin)); y1 = max(0, min(H, ymax))
B[y,x] = 1 nếu x0 <= x < x1 và y0 <= y < y1; ngược lại 0
M = create_soft_mask(B_uint8, feather_radius)
```

Các bước production:

1. Validate container, số chiều, số phần tử, kiểu scalar và H/W dương.
2. Validate feather trước mọi nhánh trả sớm, kể cả bbox rỗng.
3. Validate thứ tự bbox trên tọa độ gốc.
4. Clip tọa độ về `[0,W]`, `[0,H]`.
5. Tạo `np.zeros((H,W), dtype=np.uint8)`; chỉ tô 255 nếu `x0<x1` và `y0<y1`.
6. Gọi `create_soft_mask` để thống nhất chuẩn hóa, feathering và dtype, kể cả mask toàn 0/1.
7. Trả kết quả; cập nhật docstring và README.

Giữ nguyên Gaussian hiện có: r=0 hoặc 1 không blur; r chẵn dương tăng lên số lẻ kế tiếp, r lẻ giữ nguyên; `k` là kích thước kernel, `sigma=k/3`, `BORDER_REFLECT_101`. Với mặc định r=15: k=15, sigma=5, bán kính hỗ trợ thực là 7 pixel. Không đổi thành `2*r+1`.

Gaussian có thể làm mask dương ngoài bbox và giảm giá trị trong bbox nhỏ. Không yêu cầu tổng soft-mask bằng diện tích bbox, đặc biệt gần mép ảnh có phản xạ. Clip bbox trước blur; không dựng vùng ngoài canvas rồi blur và crop. Hàm trả mask, không tự chỉnh ảnh.

## 3. Testcase và baseline

### 3.1. Testcase công khai đã xác minh

Nguồn: mục Examples của [scikit-image draw.rectangle 0.25.2](https://scikit-image.org/docs/0.25.x/api/skimage.draw.html#skimage.draw.rectangle). Lấy hai ví dụ có output ma trận rõ ràng, chuyển dtype sang float32 và tắt feathering. Không cần tải ảnh tự nhiên hoặc thêm scikit-image vào production.

**WEB-EXTENT:** nguồn dùng shape `(5,5)`, start `(1,1)`, extent `(3,3)`. Input dự án: `(5,5), (1,1,4,4), 0`. Expected:

```text
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

**WEB-END:** nguồn dùng shape `(5,5)`, start `(0,1)`, end `(3,3)` inclusive. Input dự án: `(5,5), (1,0,4,4), 0`. Expected:

```text
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

Hai baseline trên là ví dụ công khai chuyển thể; nguồn không cung cấp soft-mask theo sigma của dự án. Không gọi output feathered tự sinh là baseline công khai.

### 3.2. Test hình học do dự án thiết kế

Với mọi dòng sau, feather=0, shape `(4,6)` trừ khi ghi khác. Expected bằng 1 đúng tại tập chỉ số đã ghi, bằng 0 ở mọi nơi khác.

| ID | Bbox/input | Expected đầy đủ theo quy tắc tập chỉ số |
|---|---|---|
| RECT | `(1,1,5,3)` | y thuộc {1,2}, x thuộc {1,2,3,4}; tổng 8 |
| CLIP-TL | `(-2,-1,3,2)` | y thuộc {0,1}, x thuộc {0,1,2}; tổng 6 |
| CLIP-BR | `(4,2,9,8)` | y thuộc {2,3}, x thuộc {4,5}; tổng 4 |
| FULL | `(-5,-5,10,10)` | Toàn 1; tổng 24 |
| PIXEL | `(5,3,6,4)` | Chỉ `[3,5]=1` |
| OUTSIDE | `(6,0,9,3)`; `(-3,0,-1,3)`; `(0,4,3,6)`; `(0,-3,3,-1)` | Toàn 0 cho cả bốn hướng |
| EMPTY | `(2,1,2,3)`; `(1,2,4,2)`; `(1,1,1,1)` | Toàn 0 |
| REVERSED | `(4,1,2,3)`; `(1,3,4,2)`; `(9,1,8,3)` | ValueError, kể cả khi clipping sẽ xóa dấu hiệu đảo |
| TINY | shape `(1,1)`, bbox `(0,0,1,1)` | `[[1]]` |
| ROW/COL | shape `(1,5)`, bbox `(1,0,4,1)` và shape `(5,1)`, bbox `(0,1,1,4)` | `[0,1,1,1,0]` theo hàng/cột |

Bổ sung parametrized tests cho sai container; thiếu/thừa phần tử; shape 2D ndarray; H/W bằng 0 hoặc âm; bool; float; NaN/Inf; string; feather âm/float/bool. Test NumPy integers, bbox read-only, input không bị sửa, output giữa hai lần gọi không dùng chung bộ nhớ. Kiểm tra feather sai vẫn raise trên EMPTY/FULL.

### 3.3. Oracle độc lập cho soft-mask

Sinh hard baseline bằng phép so sánh trên lưới chỉ số, không gọi production và không dùng slicing giống implementation:

```python
yy, xx = np.indices((h, w))
hard = ((xx >= xmin) & (xx < xmax) &
        (yy >= ymin) & (yy < ymax)).astype(np.float64)
```

Lưới tự giới hạn miền ảnh; chỉ áp dụng cho bbox hợp lệ. Soft reference dùng SciPy float64:

```python
from scipy.ndimage import gaussian_filter

if r in (0, 1):
    expected = hard.copy()
else:
    k = r if r % 2 else r + 1
    expected = gaussian_filter(
        hard, sigma=k / 3.0, radius=k // 2,
        mode="mirror", output=np.float64,
    )
```

SciPy `mirror` tương ứng phản xạ không lặp pixel biên của OpenCV `BORDER_REFLECT_101`. Phải chỉ định radius để cùng miền hỗ trợ, không để SciPy lấy mặc định theo truncate. Tài liệu tham chiếu: [SciPy gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).

Chạy với r thuộc `{0,1,2,3,14,15,25}` trên RECT, CLIP-TL, CLIP-BR, FULL, EMPTY, PIXEL, ROW/COL và TINY. Kiểm tra r=14 giống r=15. Tạo thêm 100 bbox hợp lệ bằng `default_rng(20260921)` trên canvas `(17,23)`, gồm tọa độ ngoài ảnh; lưu input để tái lập.

Lưu golden `.npy` bằng script riêng không import `create_bbox_mask` hoặc `create_soft_mask`. Manifest ghi nguồn công khai/tự thiết kế/oracle, input, dtype, tham số Gaussian, seed, phiên bản thư viện và SHA-256. Không tự ghi đè expected khi chạy đánh giá.

### 3.4. Metric và ngưỡng nghiệm thu

Đặt A là actual, E là reference; tính sai số sau khi chuyển A sang float64.

| Nhóm | Metric | Ngưỡng đạt |
|---|---|---|
| Contract | Shape, dtype, finite, min/max | `(H,W)`, float32, toàn finite, `[0,1]` cho mọi case hợp lệ |
| Hard-mask | Số pixel khác baseline | 0; `assert_array_equal` |
| Hard-mask | Sai lệch diện tích | `abs(sum(A) - (x1-x0)*(y1-y0)) = 0` |
| Hard-mask | IoU = giao/hợp của tập pixel 1 | 1; quy ước cả hai rỗng → 1 |
| Soft-mask | MaxAE = max(abs(A-E)) | <= 1e-6 |
| Soft-mask | MAE = mean(abs(A-E)) | <= 1e-7 |
| Soft-mask | RMSE = sqrt(mean((A-E)^2)) | Báo cáo bổ sung |
| Constant/identity | Mask toàn 0/1; r=0/1; r=14/15 | Khớp tuyệt đối theo tính chất tương ứng |
| Input lỗi | Exception đúng loại | 100% case invalid đạt |

Đây là ngưỡng đề xuất, chưa phải số đo. Nếu không đạt, kiểm tra kernel/sigma/border/dtype trước; không nới ngưỡng chỉ để test pass. IoU chỉ áp dụng cho hard-mask; threshold soft-mask có thể che sai số feathering. Không dùng PSNR/SSIM hoặc đánh giá cảm quan thay thế kiểm tra từng pixel.

## 4. Kế hoạch test Python

Các file cần tạo/sửa:

| File | Công việc |
|---|---|
| `src/region_engine/spatial.py` | Validation và docstring; giữ raster hóa + create_soft_mask |
| `src/region_engine/README.md` | Contract tọa độ, input lỗi, ý nghĩa feather |
| `tests/unit/test_create_bbox_mask.py` | Hai baseline web, hình học, invalid, invariant |
| `tests/fixtures/bbox_mask/` | Manifest, hard expected và soft reference cố định |
| `scripts/generate_bbox_mask_baselines.py` | Sinh fixture từ nguồn/oracle độc lập |
| `scripts/evaluate_bbox_mask.py` | Đọc fixture, chạy actual, ghi metrics và plot; exit khác 0 nếu fail |
| `tests/integration/test_region_blending.py` | Case bbox → mask → blending qua interface sẵn có |

NumPy/OpenCV là dependency runtime hiện có; SciPy, matplotlib và pytest đã được khai báo trong requirements-dev. Ghi phiên bản thực tế khi chạy; xác minh môi trường trước khi cài thêm. Không cần GPU, checkpoint hay kết nối mạng để chạy test sau khi fixture được lưu.

Trình tự lệnh dự kiến tại root repository, sau khi tạo các script nêu trên:

```powershell
python scripts/generate_bbox_mask_baselines.py
python -m pytest tests/unit/test_create_bbox_mask.py -q
python scripts/evaluate_bbox_mask.py --output-dir artifacts/module-2/bbox-mask
python -m pytest tests/unit/test_create_soft_mask.py tests/unit/test_blend_regions.py tests/unit/test_region_engine.py tests/integration/test_region_blending.py -q
```

Script CLI cần hỗ trợ đúng `--output-dir` như kế hoạch. Lưu stdout/stderr, exit code và JUnit XML nếu cần tổng hợp số test. Test integration dùng ảnh gốc đen, ảnh processed có màu hằng `[200,100,50]`: mask=0 giữ đen, mask=1 nhận đúng màu, mask giữa hai mức tuân theo contract blend hiện có. Gaussian blending chỉ là minh họa tích hợp; baseline mask vẫn độc lập.

## 5. Lưu plot và ảnh trước/sau

Thư mục dự kiến: `artifacts/module-2/bbox-mask/`.

```text
bbox-mask/
  README.md
  manifest.json
  metrics.csv
  metrics.json
  test-results.xml
  arrays/                 # actual, expected, hard-mask float .npy
  images/                 # original.png, processed.png, blended_*.png
  plots/
    01_web_baselines.png
    02_clipping_and_empty.png
    03_feather_comparison.png
    04_before_after.png
    05_error_heatmap.png
    06_edge_profile.png
```

Demo ảnh tái lập, không phụ thuộc tải mạng: canvas RGB `(256,384,3)`, `R=floor(255*x/383)`, `G=floor(255*y/255)`, `B=80+80*((x//32+y//32)%2)`. Ảnh processed tăng mỗi kênh 50 trong int16 rồi clip/cast uint8. Bbox `(96,64,288,192)`; thêm bbox sát mép `(-32,32,128,224)`.

Plot trước/sau gồm: ảnh gốc kèm đường bbox, hard-mask, soft-mask, ảnh processed toàn ảnh, ghép hard-mask, ghép soft-mask. So sánh r=0,3,15,25 để thấy chuyển tiếp. Lưu ảnh gốc sạch riêng; đường bbox chỉ là annotation trên plot.

Heatmap hiển thị `abs(actual-reference)` bằng thang màu có nhãn; mask dùng vmin=0/vmax=1; các case web 5×5 dùng interpolation nearest. Edge profile vẽ actual/reference trên một hàng đi qua giữa bbox. Mọi plot có shape, bbox, r/k/sigma và nguồn case. Dùng matplotlib backend Agg, lưu PNG; dữ liệu đo lấy từ `.npy`, không lấy từ PNG đã lượng tử hóa.

Mở kiểm tra trực quan ít nhất plot before/after, clipping và heatmap trước khi giao. Kiểm tra nhãn không bị cắt, đúng RGB, thay đổi nằm ở bbox và dải feather hợp lệ. README artifact dẫn tới toàn bộ ảnh và ghi lệnh tái tạo.

## 6. Kết luận và nghiệm thu

Thuật toán được chọn là raster hóa bbox theo khoảng nửa kín, clipping trong miền ảnh và Gaussian feathering tái sử dụng. Cách này phù hợp API hiện tại, chạy CPU và có thể kiểm chứng bằng baseline số độc lập. Mask bbox chọn toàn bộ hình chữ nhật, không mô tả biên chính xác của đối tượng bên trong.

Kết quả đã hoàn thành ở giai đoạn lập kế hoạch: khảo sát implementation, xác minh nguồn chính thức, chọn hai testcase web có output rõ ràng, đề xuất contract và thiết kế quy trình test/plot. **Chưa có kết quả chạy thực tế hoặc ảnh đầu ra của đợt triển khai này.**

Khi thực hiện, cập nhật bảng sau bằng số đo thực, không thay bằng giá trị kỳ vọng:

| Kết quả cần báo cáo | Kết quả thực tế |
|---|---|
| Số test pass/fail/skip và lý do | 110 passed, 0 failed, 0 skipped trong nhóm bbox + Module 2/3 liên quan. Toàn bộ suite chưa collect được do môi trường thiếu `langgraph`, gây 2 lỗi import ở test agent/pipeline. |
| Pixel mismatch và IoU cho baseline web/hard-mask | 0 mismatch trên 117 case; IoU thấp nhất 1.0. |
| MaxAE, MAE, RMSE soft-mask, case sai nhất | `random_056`, `r=25`: MaxAE `1.7616806258e-7`, MAE `3.3740173375e-8`, RMSE `4.4368500555e-8`; đạt ngưỡng `1e-6`/`1e-7`. |
| Validation và regression Module 2/3 | 110 test pass; `ruff check` cho các file thay đổi pass. |
| Đường dẫn ảnh/plot đã kiểm tra | `artifacts/module-2/bbox-mask/plots/`: 6 plot; `images/`: original, processed, blended hard/soft. Đã mở kiểm tra before/after, clipping và heatmap. |
| Kết luận nghiệm thu | Đạt trong phạm vi Module 2. Cần cài dependency runtime `langgraph` nếu muốn chạy toàn bộ test suite. |

Checklist thực hiện theo thứ tự:

- [x] Tìm nguồn thuật toán và baseline công khai.
- [x] Chốt đề xuất contract, oracle và metric trong kế hoạch.
- [x] Triển khai validation và cập nhật tài liệu API.
- [x] Tạo fixture độc lập, test và script đánh giá.
- [x] Chạy Python test, lưu kết quả và xử lý lỗi; toàn bộ suite còn bị chặn bởi dependency `langgraph` thiếu trong môi trường.
- [x] Tạo/mở kiểm tra ảnh trước/sau, lưu artifact.
- [x] Ghi kết luận thực nghiệm và giới hạn còn lại vào tài liệu này.
