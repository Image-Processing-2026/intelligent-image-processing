# Kế hoạch triển khai create_quadrant_mask — Module 2

Ngày lập: 2026-09-21. Trạng thái: **đã triển khai, kiểm thử và nghiệm thu phần Module 2 trong phạm vi dependency hiện có**.

Hàm mục tiêu: `create_quadrant_mask(image_shape, quadrant, feather_radius=25)`.

Mục tiêu: chọn vùng hình học `top`, `bottom`, `left`, `right`, `center`; trả mask toàn ảnh float32 `(H,W)`, hữu hạn trong `[0,1]`, dùng trực tiếp với Module 3. Giữ tên API hiện tại dù các vùng này là nửa ảnh/vùng giữa, không phải bốn góc phần tư theo nghĩa toán học.

## 1. Nguồn trình bày thuật toán

Đây là phép chọn miền chỉ số trên lưới pixel rồi Gaussian feathering; không cần model hoặc thuật toán phân đoạn ngữ nghĩa.

| Nguồn chính thức | Cách sử dụng |
|---|---|
| [NumPy 2.2 — Indexing on ndarrays, Slicing and striding](https://numpy.org/doc/2.2/user/basics.indexing.html#slicing-and-striding) | Nguồn chính cho slicing theo từng trục, cận cuối không bao gồm và gán giá trị vào vùng chọn |
| [scikit-image 0.25.2 — draw.rectangle](https://scikit-image.org/docs/0.25.x/api/skimage.draw.html#skimage.draw.rectangle) | Ví dụ raster hóa hình chữ nhật có input/output ma trận rõ ràng; chuyển thể làm baseline vùng giữa |
| [OpenCV 4.13.0 — GaussianBlur](https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html) | Tham chiếu kernel, sigma và xử lý biên cho soft-mask |
| [SciPy — gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html) | Oracle Gaussian độc lập với OpenCV, cấu hình cùng miền hỗ trợ và cách phản xạ |

Tra cứu ngày 2026-09-21. NumPy là nguồn sát nhất cho bước chọn vùng. Các nguồn trên không định nghĩa riêng hàm năm lựa chọn của dự án; cách chia tại `H//2`, `W//2` và vùng giữa là contract dự án. Không khẳng định có một bài báo hoặc bộ benchmark chuẩn cho chính API này.

## 2. Triển khai thuật toán

### 2.1. Hiện trạng và phạm vi thay đổi

- `src/region_engine/spatial.py` đã có đủ năm nhánh, tô uint8 255 và gọi `create_soft_mask`.
- Nhánh `else` đang tô toàn ảnh khi tên vùng sai; cần thay bằng exception để không vô tình xử lý toàn ảnh.
- `image_shape[:2]` đang chấp nhận shape ba chiều; cần thống nhất shape đúng hai phần tử với `create_bbox_mask` hiện có.
- File này hiện đã có `_validate_integer_vector`, `_validate_feather_radius` và `create_bbox_mask` có validation; tái sử dụng thay vì viết lại các quy tắc.
- `src/region_engine/detector.py` đang gọi `top`, `bottom`, `center` bằng `(h,w)`; các caller này tương thích contract đề xuất.
- `src/region_engine/__init__.py` đã export hàm. Test hiện tại trong `tests/unit/test_region_engine.py` chỉ kiểm tra hai pixel của vùng top trên ảnh 100×100.

Không thay đổi `create_soft_mask`, `blend_regions` hoặc fallback prompt trong phạm vi này. Fallback toàn ảnh của `segment_by_prompt` là hành vi riêng, không đồng nghĩa với chấp nhận tên quadrant sai.

### 2.2. Contract đề xuất

| Tham số/tình huống | Quy định |
|---|---|
| `image_shape` | Tuple/list hoặc ndarray 1D, đúng hai số nguyên `(H,W)`, H,W > 0 |
| Kiểu scalar shape | Nhận Python int và NumPy integer; loại bool, float, string, complex |
| Shape `(H,W,3)` hoặc sai số chiều/độ dài | ValueError; caller dùng `image.shape[:2]` |
| Container hoặc scalar sai kiểu | TypeError |
| `quadrant` | String khớp chính xác một trong `top`, `bottom`, `left`, `right`, `center` |
| Tên khác, chuỗi rỗng, `TOP`, ` top `, `full` | ValueError, thông báo liệt kê năm giá trị hợp lệ |
| Quadrant không phải string | TypeError, kể cả None/list/ndarray |
| `feather_radius` | Integer không âm, loại bool; sai kiểu → TypeError, âm → ValueError |
| Output | Mảng mới float32 `(H,W)`, finite, mọi phần tử thuộc `[0,1]` |
| Vùng chọn rỗng vì ảnh rất nhỏ | Mask toàn 0, vẫn validate feather đầy đủ |
| Side effect | Không sửa input; các lần gọi không chia sẻ buffer output |

Chọn strict string để giữ API dễ phát hiện lỗi; chuẩn hóa ngôn ngữ/viết hoa nếu cần sẽ nằm ở lớp caller. Thay đổi từ fallback toàn ảnh sang exception và từ shape tùy ý sang `(H,W)` phải ghi trong README/docstring.

### 2.3. Quy ước tọa độ và kích thước lẻ

Gốc ảnh ở trên trái, y tăng xuống dưới, x tăng sang phải. Mọi khoảng dùng cận cuối không bao gồm.

| Vùng | y được chọn | x được chọn | Diện tích hard-mask |
|---|---|---|---|
| top | `[0,H//2)` | `[0,W)` | `(H//2)*W` |
| bottom | `[H//2,H)` | `[0,W)` | `(H-H//2)*W` |
| left | `[0,H)` | `[0,W//2)` | `H*(W//2)` |
| right | `[0,H)` | `[W//2,W)` | `H*(W-W//2)` |
| center | `[H//4,(3*H)//4)` | `[W//4,(3*W)//4)` | `((3*H)//4-H//4)*((3*W)//4-W//4)` |

Ảnh lẻ: bottom/right nhận hàng/cột dư. Top+bottom và left+right phủ ảnh đúng một lần trước feathering. Không yêu cầu hai nửa có cùng số pixel.

Center dùng `floor(3*H/4)`, không dùng `3*(H//4)`; hai biểu thức khác nhau trên kích thước không chia hết cho 4. Ví dụ `(5,7)` chọn y={1,2}, x={1,2,3,4}. Không ép vùng giữa phải đối xứng hoặc có diện tích đúng 25% với mọi kích thước nguyên.

Ảnh `(1,1)`: top/left/center là `[[0]]`, bottom/right là `[[1]]`. Không tự thêm pixel vào vùng rỗng; giữ đúng slicing hiện tại và kiểm thử rõ hành vi này.

### 2.4. Luồng implementation dự kiến

1. Validate shape bằng helper có sẵn, kiểm tra H/W dương.
2. Kiểm tra quadrant là string rồi kiểm tra thuộc tập hợp năm tên trước khi tra mapping.
3. Validate feather trước khi phân nhánh hoặc trả kết quả.
4. Ánh xạ quadrant thành bbox nửa kín:

```python
boxes = {
    "top": (0, 0, w, h // 2),
    "bottom": (0, h // 2, w, h),
    "left": (0, 0, w // 2, h),
    "right": (w // 2, 0, w, h),
    "center": (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4),
}
return create_bbox_mask((h, w), boxes[quadrant], feather_radius=radius)
```

5. Để `create_bbox_mask` raster hóa và gọi `create_soft_mask` đúng một lần. Việc lặp lại validation scalar trong bbox chấp nhận được; không cần refactor helper toàn module chỉ để bỏ vài kiểm tra nhỏ.
6. Cập nhật type hint, docstring, ví dụ README và regression test caller.

Giữ Gaussian hiện có: r=0/1 không blur; r chẵn dương tăng lên số lẻ kế tiếp, r lẻ giữ nguyên; kernel k, sigma=k/3, `BORDER_REFLECT_101`. Mặc định r=25 là kernel 25×25, sigma=25/3 và bán kính hỗ trợ 12 pixel, không phải kernel 51×51.

Feathering cho phép trọng số lan sang phía ngoài vùng chọn. Với ảnh nhỏ hơn kernel, giá trị có thể mềm trên gần như toàn ảnh; không yêu cầu mỗi vùng luôn có pixel bằng 1. Gaussian và quy tắc làm tròn kernel là contract sẵn có của dự án, không phải mặc định chung của mọi thư viện.

## 3. Testcase có input/output baseline rõ ràng

### 3.1. Baseline công khai và cách chuyển thể

**WEB-RIGHT — chuyển thể từ ví dụ NumPy.** [Nguồn NumPy 2.2, Slicing and striding](https://numpy.org/doc/2.2/user/basics.indexing.html#slicing-and-striding) có input `x=[0,1,2,3,4,5,6,7,8,9]`, phép `x[5:]`, output `[5,6,7,8,9]`.

Input API dự án: `image_shape=(1,10), quadrant="right", feather_radius=0`. Chuyển các vị trí được chọn thành trọng số 1, còn lại 0; expected float32:

```text
[[0,0,0,0,0,1,1,1,1,1]]
```

Kiểm tra thêm `x[actual[0] == 1]` bằng output công khai. Đây là phép chuyển thể từ output slicing sang mask, không phải NumPy đã công bố output của hàm quadrant.

**WEB-CENTER — chuyển thể từ ví dụ rectangle.** [Nguồn scikit-image 0.25.2](https://scikit-image.org/docs/0.25.x/api/skimage.draw.html#skimage.draw.rectangle) có shape `(5,5)`, start `(1,1)`, extent `(3,3)`, expected:

```text
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

Đệm thêm một hàng 0 phía dưới và một cột 0 bên phải của baseline đã công bố. Kết quả là baseline chuyển thể cho `image_shape=(6,6), quadrant="center", feather_radius=0`, vì `6//4=1`, `(3*6)//4=4`:

```text
0 0 0 0 0 0
0 1 1 1 0 0
0 1 1 1 0 0
0 1 1 1 0 0
0 0 0 0 0 0
0 0 0 0 0 0
```

Lưu riêng baseline nguyên gốc và thao tác padding trong manifest. Không dùng center `(5,5)` để so trực tiếp với hình chữ nhật công khai 3×3: contract center `(5,5)` chỉ chọn 2×2. Không có baseline soft-mask theo tham số dự án trong hai ví dụ này.

### 3.2. Baseline nội bộ cho đầy đủ năm vùng

Fixture EVEN: shape `(4,6)`, r=0; mọi vị trí không được liệt kê đều bằng 0, vị trí liệt kê bằng 1, dtype float32.

| Quadrant | Tập pixel expected | Tổng |
|---|---|---|
| top | y={0,1}, mọi x | 12 |
| bottom | y={2,3}, mọi x | 12 |
| left | mọi y, x={0,1,2} | 12 |
| right | mọi y, x={3,4,5} | 12 |
| center | y={1,2}, x={1,2,3} | 6 |

Fixture ODD: shape `(5,7)`, r=0:

| Quadrant | Tập pixel expected | Tổng |
|---|---|---|
| top | y={0,1}, mọi x | 14 |
| bottom | y={2,3,4}, mọi x | 21 |
| left | mọi y, x={0,1,2} | 15 |
| right | mọi y, x={3,4,5,6} | 20 |
| center | y={1,2}, x={1,2,3,4} | 8 |

Fixture TINY: `(1,1)` theo mục 2.3; `(1,5)` center toàn 0, top toàn 0, bottom toàn 1, left `[1,1,0,0,0]`, right `[0,0,1,1,1]`. Với `(5,1)`, center/left toàn 0, right toàn 1, top là cột `[1,1,0,0,0]`, bottom là cột `[0,0,1,1,1]`.

Bổ sung shape `(2,2)`, `(3,3)`, `(6,10)`, `(7,9)`, `(8,12)` để bao phủ phần dư theo 4; dùng oracle dưới đây. Chạy tất cả năm vùng với r thuộc `{0,1,2,3,24,25}` và kiểm tra gọi không truyền feather tương đương r=25.

### 3.3. Oracle độc lập

Hard reference dùng lưới `(yy,xx)=np.indices((H,W))` và các bất đẳng thức, không gọi production/bbox hoặc copy mapping bbox của implementation:

```text
top:    yy < H//2
bottom: yy >= H//2
left:   xx < W//2
right:  xx >= W//2
center: H//4 <= yy < (3*H)//4 và W//4 <= xx < (3*W)//4
```

Hai fixture EVEN/ODD phải lưu literal expected theo bảng để kiểm tra cả oracle, tránh oracle và production cùng mắc lỗi làm tròn. Soft reference tính từ hard float64 bằng SciPy:

```python
if r in (0, 1):
    expected = hard.astype(np.float64)
else:
    k = r if r % 2 else r + 1
    expected = scipy.ndimage.gaussian_filter(
        hard.astype(np.float64), sigma=k / 3.0,
        radius=k // 2, mode="mirror", output=np.float64,
    )
```

`mirror` của SciPy phản xạ không lặp pixel biên, phù hợp `BORDER_REFLECT_101`; radius chỉ định để tránh khác miền hỗ trợ do truncate mặc định. Xem [SciPy gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).

Generator baseline không import `create_quadrant_mask`, `create_bbox_mask` hoặc `create_soft_mask`. Lưu `.npy` expected cố định và manifest gồm ID, nguồn, cách chuyển thể, shape, quadrant, r/k/sigma/border, phiên bản thư viện, SHA-256. Script evaluate chỉ đọc baseline, không tự cập nhật golden khi fail.

### 3.4. Invariant và input lỗi

- Hard: top+bottom=1, left+right=1, hai vùng đối ứng không chồng nhau.
- Soft: tổng hai vùng đối ứng xấp xỉ 1 trong atol=1e-6, rtol=0; không yêu cầu chúng không chồng vì có feather.
- Hoán vị trục: top(H,W).T = left(W,H), bottom(H,W).T = right(W,H), center(H,W).T = center(W,H); hard exact, soft atol=1e-6.
- Phản chiếu top/bottom chỉ yêu cầu đối xứng khi H chẵn; left/right khi W chẵn. Không áp đặt lên ảnh lẻ.
- r=0 bằng r=1, r=24 bằng r=25, gọi mặc định bằng r=25: khớp tuyệt đối từng cặp.
- Kiểm tra float32, shape, finite/range, output độc lập và input shape list/ndarray read-only không bị thay đổi.
- Parametrize input lỗi theo contract, gồm bool, float nguyên, NaN/Inf, ndarray 2D, H/W âm/0, tên sai và feather sai kể cả vùng TINY rỗng.

### 3.5. Metric và ngưỡng đạt

Đặt A=actual, E=expected; chuyển sang float64 khi tính sai số.

| Metric | Ngưỡng nghiệm thu |
|---|---|
| Shape/dtype/finite/range | Đúng contract ở mọi case hợp lệ |
| Hard pixel mismatch | 0, dùng `assert_array_equal` |
| Hard IoU = giao/hợp của tập pixel 1 | 1; cả hai rỗng quy ước 1 |
| Hard sai lệch diện tích | 0 so với công thức mục 2.3 |
| Soft MaxAE = max(abs(A-E)) | <= 1e-6 |
| Soft MAE = mean(abs(A-E)) | <= 1e-7 |
| Soft RMSE = sqrt(mean((A-E)^2)) | Báo cáo bổ sung |
| Invariant và invalid input | Tất cả test đạt đúng kỳ vọng |

Các ngưỡng là tiêu chí đề xuất, chưa phải kết quả đo. Khi fail, kiểm tra rounding/kernel/border/dtype trước khi cân nhắc đổi tolerance có giải thích. Không dùng IoU của soft-mask threshold để thay sai số từng pixel; không yêu cầu tổng soft-mask giữ nguyên diện tích hard-mask. PSNR/SSIM không cần thiết cho phép chọn vùng xác định này.

## 4. Test Python và đánh giá

### 4.1. File và trình tự công việc

| File | Công việc dự kiến |
|---|---|
| `src/region_engine/spatial.py` | Hoàn thiện validation và mapping năm vùng sang bbox |
| `src/region_engine/README.md` | Contract, ảnh lẻ/tiny, exception và ý nghĩa feather |
| `tests/unit/test_create_quadrant_mask.py` | Baseline web chuyển thể, literal, oracle, invariant, invalid |
| `tests/fixtures/quadrant_mask/` | `.npy` và manifest provenance/hash |
| `scripts/generate_quadrant_mask_baselines.py` | Sinh golden bằng nguồn/literal/SciPy độc lập |
| `scripts/evaluate_quadrant_mask.py` | Metrics theo case, plot, báo fail bằng exit code khác 0 |
| `tests/integration/test_region_blending.py` | Quadrant → mask → blend/apply_region_op |

SciPy, matplotlib, pytest đã khai báo trong requirements-dev; NumPy/OpenCV là runtime dependency. Xác minh interpreter và phiên bản thực tế, không cài thêm scikit-image chỉ để tái hiện ma trận literal. Toàn bộ test sau khi lưu fixture chạy CPU/offline.

Sau khi triển khai các file trên, chạy từ root repository:

```powershell
python scripts/generate_quadrant_mask_baselines.py
python -m pytest tests/unit/test_create_quadrant_mask.py -q
python scripts/evaluate_quadrant_mask.py --output-dir artifacts/module-2/quadrant-mask
python -m pytest tests/unit/test_create_bbox_mask.py tests/unit/test_create_soft_mask.py tests/unit/test_blend_regions.py tests/unit/test_region_engine.py tests/integration/test_region_blending.py -q
```

Các script mới phải hỗ trợ lệnh như trên. Lưu log và JUnit XML cho lượt nghiệm thu; ghi rõ số pass/fail/skip và lý do thiếu dependency nếu có. Không nhận kết quả skip làm bằng chứng integration đã đạt.

### 4.2. Integration và caller regression

Dùng O toàn 0 `(4,6,3)`, P toàn màu `[200,100,50]`, r=0, kiểm thử cả năm vùng: expected lấy màu P tại đúng tập pixel literal EVEN, còn lại giữ O. Test thêm `apply_region_op` với phép xử lý trả P; tránh chỉ so hai đường đi dùng cùng production mask mà không kiểm tra vị trí chọn.

Case r=3 kiểm tra shape/dtype và công thức blend theo mask đã nghiệm thu; bảo toàn pixel M=0/1. Các caller `segment_by_prompt` với `sky`, `ground`, `center` phải tiếp tục chạy; đây chỉ là regression heuristic hiện tại, không nghiệm thu segmentation model.

## 5. Save plot ảnh trước/sau

Thư mục đầu ra dự kiến: `artifacts/module-2/quadrant-mask/`.

```text
quadrant-mask/
  README.md
  manifest.json
  metrics.csv
  metrics.json
  test-results.xml
  logs/
  arrays/                   # hard, actual, reference .npy
  images/                   # original, processed, blended từng vùng .png
  plots/
    01_five_regions_hard_soft.png
    02_even_odd_tiny.png
    03_before_after_all_regions.png
    04_feather_comparison.png
    05_error_heatmaps.png
    06_edge_profiles.png
```

Demo xác định, không cần tải ảnh: canvas RGB `(256,384,3)`, với x=0..383, y=0..255:

```text
R = floor(255*x/383)
G = y
B = 80 + 80*((x//32 + y//32) mod 2)
processed = uint8(clip(int16(original) + 50, 0, 255))
```

Lưu ảnh gốc và processed toàn ảnh riêng. Với từng quadrant, lưu hard-mask r=0, soft-mask r=25, ghép hard và ghép soft. Plot before/after có năm hàng theo vùng, các cột original/processed/hard blend/soft blend; plot mask tách riêng để dễ nhìn. So sánh feather r=0,3,15,25 cho top và center.

Plot even/odd/tiny có nhãn chỉ số pixel và số pixel chọn; ảnh nhỏ dùng interpolation nearest. Mask dùng thang cố định `[0,1]`; heatmap `abs(actual-reference)` có colorbar, MaxAE và scale ghi rõ. Edge profile top lấy một cột đi qua biên ngang; left lấy một hàng; center lấy cả hàng/cột qua giữa vùng.

Dùng matplotlib backend Agg, PNG cho xem ảnh; đánh giá số từ `.npy` để tránh lượng tử hóa PNG. Ghi quadrant, shape, r/k/sigma và nguồn fixture trên plot. Mở kiểm tra tối thiểu before/after, even/odd/tiny và error heatmap; sửa nhãn bị cắt hoặc hiển thị RGB sai trước khi giao. README dẫn tới ảnh/plot và lệnh tái tạo.

## 6. Kết luận thuật toán và kết quả

Chọn mapping năm tên vùng thành miền chữ nhật theo floor integer, tái sử dụng bbox rasterization và Gaussian feathering. Cách này giữ hành vi hình học hiện tại, bổ sung validation và loại bỏ fallback toàn ảnh khi tên sai. Mask chọn vị trí, không nhận biết đối tượng: top không bảo đảm là bầu trời, center không bảo đảm là chủ thể.

Kết quả triển khai: đã bổ sung validation strict, mapping năm vùng qua `create_bbox_mask`, fixture hard/soft độc lập, script đánh giá và integration regression. Plot before/after, even/odd/tiny và heatmap đã được mở kiểm tra trực quan.

| Hạng mục nghiệm thu | Kết quả hiện tại |
|---|---|
| Unit test pass/fail/skip | 197 passed, 0 failed, 0 skipped trong regression Module 2 + Module 3 liên quan. |
| Baseline công khai/literal: pixel mismatch, IoU | 0 mismatch trên 52 case hard; IoU thấp nhất 1.0. |
| Soft reference: MaxAE/MAE/RMSE, case sai nhất | `odd_5x7_right`, `r=24`: MaxAE `1.2105293745e-7`, MAE `5.3634999981e-8`, RMSE `7.1613495548e-8`; đạt ngưỡng `1e-6`/`1e-7`. |
| Tính bù nhau, hoán vị trục, tiny/odd | Đạt trong unit test; invariant hard exact, soft `atol=1e-6`. |
| Caller regression và Module 3 integration | Đạt: `sky`, `ground`, `center` và blend cả năm vùng. |
| Plot trước/sau đã mở kiểm tra | Đạt; lưu tại `artifacts/module-2/quadrant-mask/plots/` và `images/`. |
| Kết luận nghiệm thu | Đạt trong phạm vi Module 2; `ruff check` cũng pass. |

Toàn bộ test suite của repository chưa collect được do môi trường thiếu
`langgraph`, gây lỗi import ở các test agent/pipeline không thuộc phạm vi
quadrant-mask. Regression liên quan đã chạy độc lập và đạt đầy đủ.

Checklist:

- [x] Khảo sát code và tìm tài liệu chính thức.
- [x] Xác định contract, baseline, metric và kế hoạch xuất plot.
- [x] Triển khai hàm và cập nhật README/docstring.
- [x] Tạo fixture/test/script đánh giá.
- [x] Chạy test Python và xử lý các lỗi phát hiện.
- [x] Lưu, mở kiểm tra plot và ảnh trước/sau.
- [x] Cập nhật kết quả thực nghiệm, link artifact và giới hạn còn lại vào tài liệu này.
