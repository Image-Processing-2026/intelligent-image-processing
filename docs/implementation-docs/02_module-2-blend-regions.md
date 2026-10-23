# Kế hoạch triển khai blend_regions — Module 2

Ngày lập: 2026-09-21. Trạng thái: **đã triển khai, kiểm thử và tạo artifact cho blend_regions**.

Function mục tiêu:

```python
blend_regions(original_image, processed_image, soft_mask)
```

Mục đích: ghép ảnh đã xử lý vào ảnh gốc theo trọng số từng pixel. Module 2 cung cấp function; `apply_region_op` của Module 3 gọi function sau khi xử lý ảnh. Kết quả phải giữ nguyên nơi mask bằng 0, lấy ảnh đã xử lý nơi mask bằng 1 và hòa trộn nơi mask nằm giữa hai giá trị đó.

Hiện trạng tại thời điểm lập kế hoạch: function đã có công thức NumPy cơ bản và một test alpha 0.5, nhưng thiếu validation, baseline độc lập và test mask thay đổi theo vị trí. Nhánh `None` trả trực tiếp cùng object `processed_image`; nhánh RGB đang dùng `np.repeat` để nhân mask ra ba kênh.

## 1. Nguồn trình bày thuật toán

**Nguồn chính về mô hình toán:** [W3C — Compositing and Blending Level 1, bản 21-03-2024](https://www.w3.org/TR/2024/CRD-compositing-1-20240321/), mục 5.1, 5.1.1 và 9.1.4. Đây là tài liệu chính thức ở trạng thái Candidate Recommendation Draft, không gọi là Recommendation đã hoàn tất. Mục 5.1.1 có các ví dụ màu đầu vào và kết quả số; dùng hai ví dụ với nền đục làm nguồn testcase.

**Nguồn triển khai dễ đọc:** [OpenCV — Adding (blending) two images](https://docs.opencv.org/4.13.0/d5/dc4/tutorial_adding_images.html). Trang trình bày nội suy tuyến tính giữa hai ảnh với alpha hằng. Với dự án này, mở rộng alpha thành một mảng theo vị trí; `cv2.addWeighted` với một alpha hằng không thay thế được mask không gian.

**Nguồn đối chiếu phụ:** [Pillow — Image.blend](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.blend), API hòa trộn hai ảnh bằng alpha hằng. Chỉ dùng đối chiếu bổ sung trên case tương ứng, không coi mọi cách làm tròn hoặc lượng tử hóa alpha của thư viện khác là contract của dự án.

Các nguồn được tra cứu ngày 2026-09-21. Baseline công khai được chọn là **ví dụ số W3C**, không phải dataset ảnh tự nhiên. Các test ma trận, mask ngẫu nhiên và ảnh demo ở dưới do dự án thiết kế; phải ghi rõ nguồn gốc đó trong manifest.

## 2. Đặc tả và kế hoạch triển khai thuật toán

### 2.1. Input, output và hành vi

| Thành phần | Contract đề xuất |
|---|---|
| `original_image` | NumPy RGB `uint8`, shape `(H, W, 3)`, không rỗng |
| `processed_image` | NumPy RGB `uint8`, cùng shape với ảnh gốc; là ảnh đầy đủ, không phải crop |
| `soft_mask` | NumPy `float32`, shape `(H, W)` hoặc `(H, W, 1)`, hữu hạn, mọi giá trị trong `[0,1]` |
| `soft_mask=None` | Giữ hỗ trợ hiện có: trả một bản sao của `processed_image` sau khi kiểm tra hai ảnh |
| Output | Mảng RGB `uint8` mới, cùng shape với ảnh đầu vào |
| Side effect | Không sửa hoặc trả view dùng chung vùng nhớ với input |
| Sai kiểu | `TypeError`: không phải ndarray, sai dtype; mask bool/uint8/float64 không tự chuyển ngầm |
| Sai nội dung | `ValueError`: shape không hợp lệ, ảnh khác kích thước, mask sai kích thước, NaN/Inf hoặc ngoài range |

Phạm vi nghiệm thu là RGB ba kênh theo giao diện dự án. Không nhận grayscale/RGBA ở lần này; kiểm tra caller trước khi áp dụng validation vì code cũ vô tình chấp nhận một số shape khác. Nếu cần hỗ trợ thêm, phải bổ sung contract và test riêng.

Mask `(H,W,3)` bị từ chối để bảo đảm một trọng số chung cho ba kênh. Không resize ảnh/mask, tự blur mask, đổi thứ tự RGB/BGR, tự chuẩn hóa mask 0/255 hoặc clip mask sai giá trị. Các thao tác đó có thể che lỗi tích hợp.

Nếu giữ nhánh `None`, cập nhật type hint thành `np.ndarray | None`. Trả bản sao thay vì cùng object là thay đổi ownership có chủ đích; nội dung pixel vẫn giữ nguyên. Caller `apply_region_op` hiện tự xử lý `None` trước khi gọi hàm này.

### 2.2. Công thức và quy tắc tính

Ký hiệu `O` là ảnh gốc, `P` là ảnh đã xử lý, `M` là mask. Với mỗi pixel và kênh màu:

```text
F[y,x,c] = M[y,x] * P[y,x,c] + (1 - M[y,x]) * O[y,x,c]
Y = uint8(clip(F, 0, 255))
```

Đây là trường hợp source-over với nền đục: ảnh processed đóng vai lớp trên, mask là alpha của lớp trên, ảnh original là nền có alpha bằng 1. Output vẫn là ảnh RGB đục; không có phép chia cho mask và không nhân alpha thêm lần nữa.

Quyết định riêng của dự án:

- Tính toán bằng `float32` như code hiện có; chỉ cast về `uint8` ở cuối.
- Giữ quy tắc cắt phần thập phân của số không âm, không chuyển sang làm tròn gần nhất: 127.5 → 127.
- Hòa trộn trực tiếp giá trị RGB đang lưu để tương thích pipeline hiện tại. Không tự chuyển sRGB sang linear-light. W3C cung cấp mô hình compositing; quy tắc số học và encoding cụ thể ở đây là contract của dự án.
- Nếu hai ảnh cùng giá trị ở một kênh/pixel, output phải giữ đúng giá trị đó, kể cả mask bất kỳ. Cần bảo vệ tính chất này khỏi lỗi float32 sát ngưỡng nguyên.

### 2.3. Luồng xử lý dự kiến

1. Validate hai ảnh: ndarray, uint8, RGB ba kênh, không rỗng và cùng shape.
2. Nếu mask là `None`, trả `processed_image.copy()`.
3. Validate kiểu, shape, dtype, tính hữu hạn và range của mask.
4. Với mask 2D, dùng `mask[..., None]`; mask `(H,W,1)` dùng trực tiếp. Broadcasting áp dụng một trọng số cho ba kênh, không cần `np.repeat`.
5. Chuyển ảnh sang float32 trước mọi phép nhân/cộng/trừ để tránh overflow uint8.
6. Tính biểu thức hai trọng số như trên. Với các kênh có `O == P`, giữ lại chính xác giá trị gốc, chẳng hạn dùng `np.copyto` trên kết quả float với điều kiện tương ứng.
7. Clip sai số số học về `[0,255]`, cast uint8, trả mảng mới.

Không thêm tham số sigma/radius vào `blend_regions`; mask đã được tạo bởi `create_soft_mask` trước đó. Kiểm thử số học của blending phải dùng mask cố định, không phụ thuộc việc chạy Gaussian mỗi lần.

### 2.4. Checklist triển khai

- [x] Thêm validation, thông báo lỗi và docstring theo contract.
- [x] Giữ đúng chiều trọng số: `M=1` chọn processed.
- [x] Dùng broadcasting, tính float32, clip/cast một lần cuối.
- [x] Bảo toàn pixel với `M=0`, `M=1` và các kênh có `O=P`.
- [x] Nhánh `None` trả bản sao; không thay đổi bất kỳ input nào.
- [x] Kiểm tra các caller Module 3; giữ implementation và kết quả đã có của `create_soft_mask`.

## 3. Testcase có input/output baseline rõ ràng

### 3.1. Test chuyển thể trực tiếp từ ví dụ W3C

Nguồn chung: [W3C, mục 5.1.1 — Examples of simple alpha compositing](https://www.w3.org/TR/2024/CRD-compositing-1-20240321/). Dữ liệu dưới đây tái hiện vùng giao nhau trong các ví dụ, không lấy màu bằng cách đọc screenshot.

| ID | Input chuẩn hóa từ nguồn | Output số từ nguồn | Chuyển thể sang API dự án |
|---|---|---|---|
| WEB-OPAQUE | Nền đỏ `(1,0,0)`, alpha nền 1; lớp trên xanh dương `(0,0,1)`, alpha 1 | RGB `(0,0,1)`, alpha output 1 | O=`[[[255,0,0]]]`, P=`[[[0,0,255]]]`, M=`[[1.0]]`; Y=`[[[0,0,255]]]` |
| WEB-HALF | Nền đỏ `(1,0,0)`, alpha nền 1; lớp trên xanh dương `(0,0,1)`, alpha 0.5 | RGB `(0.5,0,0.5)`, alpha output 1 | O=`[[[255,0,0]]]`, P=`[[[0,0,255]]]`, M=`[[0.5]]`; F=`[[[127.5,0,127.5]]]`; Y=`[[[127,0,127]]]` |

Hai ảnh O/P có dtype uint8, M có dtype float32. Thực hiện thêm phiên bản tile 32×32 của từng pixel để có ảnh xem.

Phân biệt nguồn: `(0.5,0,0.5)` là baseline công khai; `[127,0,127]` là baseline chuyển thể bằng quy tắc uint8 của dự án, không phải số uint8 W3C công bố. Không dùng ví dụ có alpha nền nhỏ hơn 1 vì function hiện không nhận alpha của ảnh gốc.

### 3.2. Case số tự thiết kế: mask thay đổi theo vị trí

O và P là ảnh 2×2 màu hằng: O mỗi pixel `[10,20,30]`, P mỗi pixel `[110,220,230]`; M float32:

```text
[[0.00, 0.25],
 [0.50, 1.00]]
```

Expected uint8 tính tay:

```text
[[[ 10,  20,  30], [ 35,  70,  80]],
 [[ 60, 120, 130], [110, 220, 230]]]
```

Test này phát hiện đảo original/processed, dùng alpha hằng, sai trục hoặc broadcast sai. Lặp lại với mask `(2,2,1)`; expected không đổi.

### 3.3. Baseline số nguyên độc lập cho alpha dạng t/256

Với `t` nguyên trong `[0,256]`, tạo `M=t/256` bằng float32. Các alpha này biểu diễn chính xác trong nhị phân. Expected từng kênh tính hoàn toàn bằng số nguyên int64:

```text
expected = ((256 - t) * O + t * P) // 256
```

Sinh một ảnh có 257 cột bao phủ `t=0..256`, với nhiều cặp màu tăng/giảm và các mức 0,1,127,128,254,255. Toán hạng integer đủ nhỏ để float32 biểu diễn chính xác trong phép tính production ở cấu hình này; yêu cầu output bằng baseline từng byte, không dung sai.

Baseline này do dự án thiết kế dựa trên công thức, không gọi `blend_regions` để tạo expected và không phụ thuộc một thư viện hòa trộn khác.

### 3.4. Baseline cho mask float32 tổng quát

Với các fixture nhỏ, đọc đúng giá trị float32 đã lưu rồi chuyển từng alpha sang phân số chính xác:

```python
from fractions import Fraction

m = Fraction.from_float(float(mask[y, x]))
value = m * int(processed[y, x, c]) + (1 - m) * int(original[y, x, c])
expected[y, x, c] = value.numerator // value.denominator
```

Dùng vòng lặp scalar độc lập trong script sinh fixture, không import production function. Đây là oracle toán học chính xác cho giá trị alpha thực sự trong file, tránh nhầm `float32(0.1)` với số thập phân 0.1 chính xác. Lưu cả expected uint8 và reference trước lượng tử hóa dạng float64 phục vụ vẽ/đo.

Với ảnh demo lớn có thể dùng NumPy float64 làm reference thực dụng; không gọi đó là tính toán phân số chính xác. Không lấy ảnh JPEG hoặc ảnh chụp màn hình làm golden pixel vì giải mã/nén có thể thay đổi dữ liệu.

### 3.5. Ma trận kiểm thử bổ sung

| ID | Input xác định / cách sinh | Baseline và kỳ vọng |
|---|---|---|
| ZERO | O=`arange(18,uint8).reshape(2,3,3)`, P=`255-O`, mask 0 | Bằng O tuyệt đối |
| ONE | Cùng O/P, mask 1 | Bằng P tuyệt đối |
| NONE | Cùng O/P, mask None | Bằng P; không dùng chung vùng nhớ |
| SAME | O=P chứa đủ mức 0..255; mask dùng seed cố định | Giữ nguyên mọi kênh tuyệt đối |
| HALF-EXTREME | O đen, P trắng, M=0.5; đổi ngược O/P | `[127,127,127]`, không phải 128 |
| SPATIAL-2X2 | Dữ liệu tại 3.2 | Expected tính tay |
| DYADIC | Cấu hình 3.3 | Oracle int64, khớp tuyệt đối |
| RANDOM | `default_rng(20260921)`; sinh O/P uint8 `(17,23,3)`, rồi M float32 `(17,23)` | Oracle Fraction; lưu input `.npy` để cố định dữ liệu |
| NEAR-INTEGER | O=10, P=110 cho cả ba kênh; M là float32 0.1 và hai giá trị `nextafter` liền kề | Oracle Fraction; kiểm tra sai số float32 tại ranh giới cast |
| NONCONTIG | Lấy `[:,::2]` của cả O/P/M fixture RANDOM | Cùng dữ liệu và expected được cắt tương ứng |
| READONLY | Bản sao RANDOM đặt `writeable=False` | Cùng expected, không cố sửa input |
| TINY | Pixel W3C; dòng 1×N và cột N×1 với alpha t/256 | Expected nguồn hoặc integer oracle |
| INTEGRATION | O/P cố định; binary RECT → `create_soft_mask` → blend | Oracle độc lập tính từ chính mask đã tạo, kiểm tra ghép đúng |
| INVALID-IMAGE | Sai dtype, khác shape, grayscale, RGBA, rỗng, list | `TypeError` cho kiểu/dtype; `ValueError` cho shape |
| INVALID-MASK | uint8/bool/float64/list; scalar; sai H/W; ba kênh | `TypeError` cho kiểu/dtype; `ValueError` cho shape |
| INVALID-VALUE | Float32 chứa -0.01, 1.01, NaN, +Inf, -Inf | `ValueError`; không im lặng clip |

Với mọi case hợp lệ, kiểm tra shape/dtype, input không bị sửa, output không alias input và output nằm giữa O/P theo từng kênh. Case SAME và các điểm mask đúng 0/1 phải khớp tuyệt đối, không áp dụng ngoại lệ sai số.

## 4. Metric và test Python để đánh giá

### 4.1. Metric

Gọi A là ảnh thực tế và R là expected uint8. Trước khi trừ phải cast sang int16/float64 để tránh wraparound uint8.

```text
E_max = max(abs(A - R))
MAE = mean(abs(A - R))
RMSE = sqrt(mean((A - R)^2))
exact_channel_rate = mean(A == R)
exact_pixel_rate = mean(all(A == R, axis=2))
```

Ngưỡng nghiệm thu do dự án đề xuất:

| Nhóm | Điều kiện đạt |
|---|---|
| W3C, tính tay, alpha t/256, ZERO/ONE/NONE/SAME | `assert_array_equal`; E_max=MAE=RMSE=0, exact rates=100% |
| Float32 tổng quát so oracle Fraction | Mặc định khớp tuyệt đối; chỉ cho ngoại lệ một mức uint8 tại ngưỡng nguyên theo quy tắc bên dưới |
| Contract và input không hợp lệ | Tất cả kiểm tra cấu trúc, ownership, exception đều đạt |
| Tích hợp mask mềm | Khớp oracle trong quy tắc float32; phần mask=0/1 vẫn khớp tuyệt đối |

**Quy tắc ngoại lệ số học:** nếu một kênh khác baseline, chỉ chấp nhận khi chênh đúng 1, giá trị reference trước cast cách số nguyên gần nhất không quá `1e-4` trên thang 0..255, và output thuộc `{floor(reference), ceil(reference)}`. Các phép kiểm tra này tính bằng Fraction khi dùng oracle chính xác. Mọi sai khác ngoài điều kiện trên đều fail. Không dùng `atol=1` cho toàn bộ ảnh rồi bỏ qua vị trí lỗi.

Lý do: float32 có thể đẩy kết quả qua ranh giới nguyên trước khi cắt phần thập phân. Báo cáo phải liệt kê số kênh sai khác, vị trí, alpha và lý do ngoại lệ; không kết luận khớp tuyệt đối khi chỉ đạt dung sai. `1e-4` là ngân sách sai số đề xuất cho vài phép tính float32 trên thang 255, phải giữ cố định khi test; nếu cần thay đổi phải giải thích bằng phân tích số học.

MAE/RMSE là số liệu bổ sung, E_max đơn lẻ không đủ để nghiệm thu. Không so PSNR/SSIM với ảnh original để đánh giá độ đúng của blending: output chủ ý khác original. Nếu thêm PSNR thì so với output baseline, `data_range=255`; metric này không thay các assertion số học.

### 4.2. File và trình tự thực hiện

```text
src/region_engine/mask_utils.py             # Chỉ hoàn thiện blend_regions
tests/unit/test_blend_regions.py           # Golden, validation, ownership, số học
tests/integration/test_region_blending.py  # create_soft_mask → blend; Module 3 wrapper
tests/fixtures/blend_regions/
    manifest.json
    <case>_original.npy
    <case>_processed.npy
    <case>_mask.npy                        # Bỏ file này với case None
    <case>_expected.npy
    <case>_reference_float64.npy
scripts/generate_blend_baselines.py        # W3C/tính tay/int64/Fraction độc lập
scripts/evaluate_blend_regions.py          # Đo metric, phân loại sai khác, xuất plot
artifacts/module-2/blend-regions/           # Kết quả riêng, không ghi đè soft-mask
```

Manifest ghi nguồn, loại baseline, shape/dtype, quy tắc cast, màu RGB, cấu hình RNG, SHA-256 các file và phiên bản Python/NumPy. W3C ghi URL phiên bản cố định, mục và dữ liệu chuẩn hóa gốc. Baseline chỉ sinh bằng lệnh chủ động, không tự cập nhật expected khi pytest fail.

Trình tự:

1. Kiểm tra môi trường; tái sử dụng NumPy, pytest và Matplotlib hiện có. Fraction thuộc thư viện chuẩn Python, không cần dependency mới cho oracle.
2. Tạo fixture và test từ baseline W3C/tính tay/số nguyên/phân số.
3. Chạy test với implementation hiện tại để ghi nhận điểm thiếu; không mặc định mọi test sẽ fail.
4. Hoàn thiện implementation theo 2, chạy lại test tập trung.
5. Kiểm tra `apply_region_op` bằng operation giả xác định, chẳng hạn ảnh processed được dựng sẵn; tránh phụ thuộc model/API bên ngoài.
6. Chạy kiểm tra hồi quy Region Engine và Processing Engine vì hai module dùng chung hàm ghép.
7. Xuất metrics/plot; mở kiểm tra ảnh PNG thực tế.
8. Điền kết quả thực đo và kết luận ở 6.

Lệnh dự kiến sau khi các file được tạo, chạy từ root repository:

```powershell
python scripts/generate_blend_baselines.py --output tests/fixtures/blend_regions
python -m pytest tests/unit/test_blend_regions.py -v
python -m pytest tests/integration/test_region_blending.py tests/unit/test_region_engine.py tests/unit/test_create_soft_mask.py tests/unit/test_processing_engine.py -v
python scripts/evaluate_blend_regions.py --fixtures tests/fixtures/blend_regions --output artifacts/module-2/blend-regions
python -m ruff check src/region_engine/mask_utils.py tests/unit/test_blend_regions.py tests/integration/test_region_blending.py scripts/generate_blend_baselines.py scripts/evaluate_blend_regions.py
```

Trước bàn giao code, chạy thêm `python -m pytest tests/unit/ -v` theo quy trình repository. Nếu có lỗi ngoài phạm vi hoặc môi trường, ghi riêng nguyên nhân và lệnh đã chạy; không ghi “toàn bộ đạt” khi còn test chưa chạy/fail.

## 5. Lưu ảnh và plot trước/sau

Thư mục đích dự kiến: `artifacts/module-2/blend-regions/`.

```text
environment.json
metrics.csv
summary.json
report.md
arrays/<case>_actual.npy
images/<case>_original.png
images/<case>_processed.png
images/<case>_blended.png
plots/w3c_half_comparison.png
plots/spatial_mask_comparison.png
plots/hard_vs_soft_blending.png
plots/blend_line_profile.png
plots/quantization_boundary.png
```

Nội dung plot:

- `comparison`: 6 ô original, processed, mask, actual, baseline, absolute error. Với RGB error map, hiển thị max lỗi ba kênh tại mỗi pixel và colorbar đơn vị mức uint8.
- `hard_vs_soft_blending`: dùng ảnh synthetic RGB 128×192 gồm gradient xác định, processed tăng sáng có clip, vùng chữ nhật `[32:96,48:144]`. So sánh cùng hai ảnh qua binary mask và soft-mask. Lưu cả mask thực dùng để oracle đọc độc lập.
- `blend_line_profile`: đường cắt hàng giữa ảnh, vẽ kênh R của original/processed/actual/baseline và mask trên ô riêng. Không đưa alpha 0..1 và màu 0..255 lên cùng trục không chú thích.
- `quantization_boundary`: bảng/plot các case alpha 0.1 và `nextafter`, ghi reference trước cast, output uint8 và trường hợp ngoại lệ nếu có.

Xuất Matplotlib bằng backend `Agg`, DPI 160–200; mask dùng `vmin=0,vmax=1`, ảnh giữ RGB uint8, không auto-normalize riêng từng ảnh, không đổi RGB sang BGR khi hiển thị. Dùng `interpolation="nearest"` khi xem pixel. Đóng figure sau khi save và mở file PNG kiểm tra bố cục/chú thích.

Các file PNG ảnh gốc/processed/blended cho phép xem trước/sau độc lập với plot. Dữ liệu đo lấy từ mảng gốc/`.npy`, không đo từ screenshot hoặc figure có chữ. “Trước” là original; processed là ảnh trung gian đã xử lý toàn khung; “sau” là kết quả blend theo vùng.

## 6. Kết luận thuật toán và báo cáo kết quả

### 6.1. Kết luận ở giai đoạn lập kế hoạch

Alpha blending phù hợp với giao diện hiện có: mỗi pixel lấy tổ hợp có trọng số giữa original và processed. Cần nghiệm thu cả công thức, chiều alpha, broadcast, dtype, lượng tử hóa và ownership. Một case alpha hằng 0.5 chưa đủ chứng minh mask theo vùng hoạt động đúng.

Giới hạn: function không phát hiện vùng, không tự làm mềm mask và không sửa lỗi xử lý ảnh. Mask nhị phân vẫn tạo biên cứng; mask mềm tạo dải chuyển tiếp nhưng không bảo đảm loại bỏ mọi halo. Chỉ pixel có mask đúng 0 mới được giữ hoàn toàn; vùng alpha nhỏ do feathering vẫn thay đổi một phần. Blending trực tiếp RGB và cắt phần thập phân là lựa chọn tương thích hiện tại, không phải xử lý ánh sáng tuyến tính hay làm tròn tối ưu cho mọi ứng dụng.

### 6.2. Bảng kết quả chờ thực hiện

| Hạng mục | Trạng thái hiện tại | Bằng chứng cần điền |
|---|---|---|
| Nguồn thuật toán | Đã tra cứu | W3C, OpenCV, Pillow ở 1 |
| Baseline công khai | Đã xác định input/output | Hai ví dụ W3C tại 3.1 |
| Implementation hoàn thiện | Đã hoàn thiện; Ruff đạt | `src/region_engine/mask_utils.py` |
| Test trực tiếp và tích hợp | 63 passed | `test_blend_regions.py`, `test_region_blending.py`, regression Module 2 |
| E_max/MAE/RMSE, exact rates | 9/10 case exact; case còn lại sai 3 kênh một mức | `artifacts/module-2/blend-regions/metrics.csv` |
| Ngoại lệ float32 sát số nguyên | 3/3 kênh khác biệt thỏa điều kiện boundary | `summary.json`, `quantization_boundary.png` |
| Ảnh trước/sau và plot | Đã tạo và kiểm tra PNG | `artifacts/module-2/blend-regions/` |
| Nghiệm thu | Đạt contract; không có lỗi ngoài phạm vi | Checklist dưới đây |

### 6.3. Điều kiện hoàn thành

- [x] Hai testcase W3C và case 2×2 đạt expected đã ghi.
- [x] Bộ alpha t/256 khớp byte tuyệt đối với oracle số nguyên.
- [x] ZERO/ONE/NONE/SAME đạt chính xác và output không alias input.
- [x] Case float32 tổng quát đạt oracle; ngoại lệ sát ranh giới được ghi nhận riêng.
- [x] Validation, shape, dtype, non-contiguous và read-only đạt.
- [x] Tích hợp Module 3 và kiểm tra hồi quy liên quan đạt.
- [x] Fixture có provenance/hash và test không cần mạng.
- [x] Metrics, arrays, ảnh trước/sau và plot đã lưu, mở kiểm tra.
- [x] Báo cáo tách rõ đúng số học, sai khác lượng tử hóa và chất lượng thị giác.

Kết luận thực nghiệm: đã chạy 10 case, 10 đạt theo contract; 9 case khớp tuyệt đối và 1 case có 3 kênh sai khác một mức do phép tính float32 đẩy giá trị sát số nguyên qua ranh giới. Cả 3 khác biệt đều thỏa điều kiện boundary; E_max=1, MAE=0.333333 và RMSE=0.577350 trên case đó. Pixel mask 0/1, kênh `O=P` và nhánh `None` được bảo toàn; output không alias input. Plot hard/soft cho thấy mask mềm tạo dải chuyển tiếp ở biên. Implementation đạt contract với giới hạn đã nêu: hòa trộn trực tiếp RGB và cắt phần thập phân, không tự resize/blur/clip mask.
