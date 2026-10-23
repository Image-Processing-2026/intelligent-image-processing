# Kế hoạch triển khai create_soft_mask — Module 2

Ngày lập: 2026-09-21. Trạng thái: **đã triển khai và kiểm chứng**, implementation, fixture, metrics và plot đã được tạo.

## Mục tiêu và phạm vi

Hoàn thiện `create_soft_mask(binary_mask, feather_radius=15)` tại `src/region_engine/mask_utils.py`: chuyển mặt nạ nhị phân thành mặt nạ mềm bằng Gaussian filtering, để Module 3 dùng làm trọng số alpha khi ghép ảnh.

Đầu ra bàn giao sau khi thực hiện kế hoạch gồm: function hoàn thiện, test Python có baseline độc lập, dữ liệu sai số, plot trước/sau và báo cáo kết luận thực nghiệm. Không triển khai detector, SAM, MediaPipe hoặc sửa thuật toán của `blend_regions` trong phạm vi này.

## 1. Nguồn trình bày thuật toán

### 1.1 Nguồn chính: OpenCV

- [OpenCV — Smoothing Images, mục Gaussian Blurring](https://docs.opencv.org/4.13.0/d4/d13/tutorial_py_filtering.html): đọc trước để hiểu lọc ảnh bằng kernel Gaussian và xem minh họa.
- [OpenCV — GaussianBlur và getGaussianKernel](https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html): nguồn đặc tả API, tham số và công thức kernel. Đây là nguồn chính cho implementation vì dự án đang dùng OpenCV.
- [OpenCV — BorderTypes](https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html): tra quy tắc ngoại suy mép ảnh; `BORDER_DEFAULT` tương ứng `BORDER_REFLECT_101`.

“Chuẩn” ở đây là đặc tả chính thức của thư viện dùng để triển khai. Không có một bộ tham số feathering duy nhất đúng cho mọi ảnh. Lựa chọn sigma, độ rộng kernel và kiểu biên bên dưới là quyết định của dự án, không phải một tiêu chuẩn bắt buộc từ OpenCV.

### 1.2 Nguồn kiểm chứng độc lập: SciPy

- [SciPy 1.14.1 — gaussian_filter](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.ndimage.gaussian_filter.html): có ví dụ input/output số cụ thể; hỗ trợ chỉ định `sigma`, `radius`, `mode`, `output` để đối chiếu.
- [SciPy — tests/test_filters.py](https://github.com/scipy/scipy/blob/main/scipy/ndimage/tests/test_filters.py): tham khảo các test Gaussian về shape, dtype và trường hợp không làm mờ. Link `main` có thể thay đổi; nếu lấy thêm fixture từ đây, ghi lại commit và giấy phép khi triển khai.

Các nguồn trên được tra cứu ngày 2026-09-21. Chưa tìm được trong các nguồn đã kiểm tra một bộ golden binary-mask feathering khớp hoàn toàn API hiện tại. Vì vậy sẽ phân biệt rõ: **test công khai nguyên bản**, **baseline toán học tự thiết kế**, và **baseline sinh bằng SciPy**. Không gọi dữ liệu tự tạo là testcase có sẵn trên mạng.

## 2. Đặc tả và kế hoạch triển khai

### 2.1 Contract đề xuất

```python
def create_soft_mask(binary_mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    ...
```

| Thành phần | Quy định |
|---|---|
| `binary_mask` | `np.ndarray` 2D, shape `(H, W)`, `H > 0`, `W > 0` |
| Dtype đầu vào | `bool` hoặc `uint8` |
| Giá trị hợp lệ | Boolean; hoặc toàn bộ giá trị thuộc `{0, 1}`; hoặc thuộc `{0, 255}` |
| Giá trị không hợp lệ | Từ chối mask trộn `{0, 1, 255}` hoặc mức xám như 128 |
| `feather_radius` | Số nguyên Python/NumPy không âm; từ chối bool và số thực |
| Đầu ra | Mảng mới `(H, W)`, `np.float32`, hữu hạn, nằm trong `[0, 1]` |
| Side effect | Không sửa input, không đọc/ghi file trong function |
| Lỗi | `TypeError` cho sai kiểu; `ValueError` cho sai shape, giá trị hoặc tham số âm |

Việc từ chối mask float là contract đề xuất dựa trên docstring hiện có, không phải giới hạn của Gaussian filtering. Nếu cần nhận float từ model, thực hiện chuyển sang binary mask rõ ràng ở caller; không tự threshold dữ liệu không hợp lệ trong hàm này.

### 2.2 Quyết định về feather_radius

**Giữ hành vi với đầu vào hợp lệ của code hiện tại để không âm thầm thay đổi mức feathering ở các caller.**

| Giá trị tham số `f` | Hành vi |
|---|---|
| `0` | Chỉ chuẩn hóa, không blur |
| `1` | Kernel 1×1, kết quả bằng mask chuẩn hóa |
| Số lẻ dương | `k = f` |
| Số chẵn dương | `k = f + 1` |
| Âm | Báo lỗi; đây là thay đổi có chủ đích so với code cũ |

Với `f > 0`: `sigma_x = sigma_y = k / 3`, bán kính hỗ trợ thực tế `r = (k - 1) // 2`.

Ví dụ mặc định: `f=15 → k=15 → r=7 → sigma=5`. Tên `feather_radius` được giữ để tương thích nhưng docstring phải giải thích nó đang là tham số kích thước kernel. Không đổi thành `k=2*f+1` trong lần triển khai này. Nếu nhóm muốn bán kính đúng nghĩa, đó là thay đổi API/hành vi cần cập nhật caller và baseline cùng lúc.

`sigma=k/3` kế thừa code hiện tại, chưa được chứng minh là lựa chọn đẹp nhất. Kernel hữu hạn này có thể tạo chuyển tiếp khác cấu hình cắt tại 3 hoặc 4 sigma; đánh giá bằng plot trước khi đề xuất thay đổi riêng.

### 2.3 Thuật toán

Gọi `B` là binary mask đã chuẩn hóa về 0/1. Với `i=-r,...,r`:

```text
g[i] = exp(-i² / (2 sigma²)) / sum_j exp(-j² / (2 sigma²))
K[u,v] = g[u] * g[v]
M[y,x] = sum_u sum_v K[u,v] * B_extended[y-u,x-v]
```

`B_extended` dùng phản chiếu không lặp pixel biên (`BORDER_REFLECT_101`). Kernel có trọng số không âm, tổng bằng 1; đầu ra lý tưởng là tổ hợp lồi của các giá trị 0/1.

Luồng implementation:

1. Validate kiểu mảng, số chiều, kích thước, dtype, tập giá trị và tham số.
2. Sao chép/chuyển mask sang `float32`.
3. Với encoding 0/255, chia cho 255; với boolean hoặc 0/1, giữ nguyên giá trị.
4. Nếu `f=0`, trả mảng mới đã chuẩn hóa.
5. Xác định `k`, `sigma` theo bảng trên.
6. Gọi OpenCV với cấu hình tường minh:

   ```python
   soft_mask = cv2.GaussianBlur(
       mask_f32,
       (k, k),
       sigmaX=k / 3.0,
       sigmaY=k / 3.0,
       borderType=cv2.BORDER_REFLECT_101,
   )
   ```

7. Clip sai số số học về `[0, 1]`, bảo đảm `float32`, trả kết quả.

Không blur trên `uint8` rồi mới chuẩn hóa vì có thể mất các mức alpha nhỏ. Không min-max normalize sau blur: mask toàn 1 phải tiếp tục là toàn 1, và một vùng rất nhỏ có thể có đỉnh nhỏ hơn 1 một cách đúng đắn.

### 2.4 Công việc trong code

- [x] Cập nhật docstring, validation và cấu hình Gaussian tường minh.
- [x] Giữ nguyên chữ ký hàm và hành vi cho input hợp lệ đang dùng.
- [x] Kiểm tra caller tại `spatial.py`, `face_detector.py`, `detector.py` phù hợp contract.
- [x] Comment tiếng Việt có dấu; thông báo lỗi/log tiếng Anh.
- [x] Không thêm SciPy/Matplotlib vào đường chạy production; chúng phục vụ test và báo cáo.

## 3. Testcase và output baseline

### 3.1 TC-WEB-01: ví dụ số công khai của SciPy

Nguồn: [mục Examples của SciPy 1.14.1](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.ndimage.gaussian_filter.html).

Input xác định: `a = np.arange(0, 50, 2, dtype=np.int64).reshape(5, 5)`.

Thao tác nguồn: `gaussian_filter(a, sigma=1)`; mode mặc định `reflect`, `truncate=4`, output giữ kiểu số nguyên.

Output baseline công khai:

```text
[[ 4,  6,  8,  9, 11],
 [10, 12, 14, 15, 17],
 [20, 22, 24, 25, 27],
 [29, 31, 33, 34, 36],
 [35, 37, 39, 40, 42]]
```

Test Python: gọi đúng cấu hình nguồn và `np.testing.assert_array_equal(actual, expected)`.

**Vai trò:** xác minh tái hiện được ví dụ công khai và môi trường reference. Đây không phải test trực tiếp cho `create_soft_mask`: input không nhị phân, dtype và kiểu biên cũng khác. Không ép baseline này thành 0/1 hoặc đổi tham số rồi vẫn gọi là output gốc.

### 3.2 TC-MATH-01: binary impulse với baseline toán học rõ ràng

Đây là testcase tự thiết kế dựa trên định nghĩa Gaussian, không phải fixture tải từ mạng.

Input `uint8` 5×5: toàn 0, riêng `[2, 2] = 255`. Tham số `f=3`, do đó `k=3`, `sigma=1`.

Đặt `q=exp(-1/2)`, `a=q/(1+2q)`, `b=1/(1+2q)`. Expected chính xác:

```text
[[0,   0,   0,   0, 0],
 [0, a*a, a*b, a*a, 0],
 [0, a*b, b*b, a*b, 0],
 [0, a*a, a*b, a*a, 0],
 [0,   0,   0,   0, 0]]
```

Tính expected bằng NumPy float64 từ biểu thức trên, không gọi `create_soft_mask`, `GaussianBlur` hoặc `getGaussianKernel`. Case này kiểm tra được chuẩn hóa, sigma, kích thước kernel, vị trí và tính đối xứng; không dùng nó để kiểm tra biên ảnh vì impulse nằm đủ xa mép.

### 3.3 Baseline độc lập cho binary mask: SciPy float64

Đối với các case có `f>0`, dùng cấu hình đối chiếu:

```python
reference = scipy.ndimage.gaussian_filter(
    normalized_mask.astype(np.float64),
    sigma=(k / 3.0, k / 3.0),
    order=0,
    mode="mirror",
    radius=(k - 1) // 2,
    output=np.float64,
)
```

Quy tắc biên `mirror` của SciPy tương ứng phản chiếu không lặp pixel biên của OpenCV. Phải truyền `radius` để khớp kích thước kernel; không dùng mặc định `reflect`/`truncate=4`. Đối chiếu này là suy luận từ đặc tả hai thư viện và phải được xác minh bằng các case sát mép.

Đây là **baseline sinh độc lập**, chưa phải golden file đã có trên mạng. Khi triển khai, sinh và lưu `.npy` kèm phiên bản thư viện, cấu hình và SHA-256. Test thường ngày chỉ đọc fixture đã lưu; chỉ tái sinh baseline bằng lệnh chủ động, không cập nhật expected tự động khi test fail.

### 3.4 Ma trận testcase trực tiếp

Mỗi testcase phải lưu input cụ thể, tham số và expected hoặc loại exception mong đợi. Không dùng ảnh minh họa làm baseline số.

| ID | Input và tham số | Baseline / điều cần kiểm tra |
|---|---|---|
| ZERO | Zeros `(32, 48)`, `f=15` | Toàn 0 |
| ONE | Ones `(32, 48)`; lặp lại với 255 và bool, `f=15` | Toàn 1 kể cả mép ảnh |
| IDENTITY | `[[0,255],[255,0]]`, `f=0,1` | `[[0,1],[1,0]]`, float32 |
| IMPULSE | TC-MATH-01, `f=3` | Ma trận giải tích ở trên |
| STEP | Zeros `(64,64)`, cột `32:` bằng 1, `f=15` | SciPy; profile ngang chuyển tiếp không giảm, có giá trị trung gian |
| RECT | Zeros `(64,64)`, `[16:48,16:48]=1`, `f=15` | SciPy; vùng sâu bên trong bằng 1, vùng xa biên bằng 0 |
| TOUCH-EDGE | Zeros `(32,32)`, `[:16,:12]=1`, `f=15` | SciPy; kiểm tra đặc biệt các hàng/cột sát mép |
| CORNER | Zeros `(5,5)`, `[0,0]=1`, `f=3` | SciPy; phát hiện chọn sai kiểu biên |
| THIN | Zeros `(65,65)`, cột 32 bằng 1, `f=15` | SciPy; cho phép đỉnh nhỏ hơn 1 |
| TINY | Mảng 1×1 với 0/1; 1×9 và 9×1 có xung giữa, `f=15` | Hằng số cho 1×1, SciPy cho hai case còn lại |
| EVEN | RECT, `f=14` và `f=15` | Cùng baseline SciPy với `k=15` |
| ENCODING | Cùng RECT với bool, uint8 0/1 và uint8 0/255 | Cùng expected SciPy |
| RANDOM | RNG `np.random.default_rng(20260921)`, `integers(0,2,(31,47),dtype=np.uint8)`, `f=3,15,25` | SciPy; lưu input để tái hiện độc lập RNG |
| NONCONTIG | Lấy `RECT[:, ::2]`, `f=15` | SciPy; input không cần contiguous |
| BAD-SHAPE | Mảng rỗng `(0,10)`, 1D `(10,)`, 3D `(10,10,1)` | `ValueError` |
| BAD-TYPE | Python list; mảng float32, int16, complex | `TypeError` |
| BAD-VALUE | uint8 chứa 128 hoặc đồng thời 0,1,255 | `ValueError` |
| BAD-PARAM | `f=-1`; `f=2.5`, `True`, `"15"` | Âm: `ValueError`; sai kiểu: `TypeError` |

Với mọi input hợp lệ: kiểm tra shape, dtype, hữu hạn, range; so sánh input trước/sau bằng `assert_array_equal` và xác nhận output không dùng chung vùng nhớ với input.

## 4. Metric và kế hoạch test Python

### 4.1 Metric đúng đắn

Gọi `A` là output thực tế và `R` là reference. Chuyển sang float64 trước khi tính sai số.

```text
E_max = max(abs(A - R))
MAE   = mean(abs(A - R))
RMSE  = sqrt(mean((A - R)^2))
```

Ngưỡng nghiệm thu đề xuất của dự án, không phải ngưỡng chuẩn từ OpenCV/SciPy:

| Kiểm tra | Điều kiện đạt |
|---|---|
| Sai số từng pixel | `np.testing.assert_allclose(A, R, rtol=0, atol=1e-6)` |
| Sai số cực đại | `E_max <= 1e-6` |
| MAE và RMSE | Ghi vào báo cáo; cùng phải `<= 1e-6` |
| Contract | Shape đúng, dtype float32, mọi giá trị hữu hạn và trong `[0,1]` |
| Không blur | `f=0,1` bằng chính xác mask đã chuẩn hóa |
| Invalid input | Đúng loại exception đã đặc tả |
| STEP | Profile không giảm trong sai số `1e-6`; có alpha giữa 0 và 1 |

Không đánh giá độ đúng bằng cách so output mềm với binary input qua IoU/SSIM: feathering chủ ý thay đổi biên. PSNR có thể ghi bổ sung với reference mềm nhưng không cần cho nghiệm thu; `E_max` trực tiếp hơn. Không áp đặt bảo toàn tổng mask cho mọi case sát mép vì quy tắc ngoại suy có thể ảnh hưởng tổng trên miền ảnh hữu hạn.

Nếu fail, kiểm tra encoding, sigma, radius, mode và dtype trước; không tự nới tolerance để làm test pass. Các test hình học bổ sung giúp tránh việc hai implementation vô tình cùng chạy sai cấu hình.

### 4.2 File dự kiến

```text
src/region_engine/mask_utils.py               # Hoàn thiện function
tests/unit/test_create_soft_mask.py           # Contract + golden + giải tích
tests/reference/test_gaussian_reference.py    # Tái hiện TC-WEB-01
tests/fixtures/soft_mask/                     # Input và expected cố định
    manifest.json                            # Nguồn, encoding, k/sigma/mode, hash
    <case_id>_input.npy
    <case_id>_expected.npy
scripts/generate_soft_mask_baselines.py       # Sinh reference độc lập bằng SciPy
scripts/evaluate_soft_mask.py                 # Đo sai số và xuất plot
artifacts/module-2/create-soft-mask/           # Kết quả thực nghiệm
```

Các file trên đã được tạo. Hàm sinh expected không import implementation đang kiểm thử; fixture được lưu kèm phiên bản SciPy/NumPy và SHA-256. Các test không cần tải dữ liệu qua mạng lúc chạy.

### 4.3 Trình tự thực hiện

1. Ghi nhận Python, NumPy, OpenCV, SciPy, Matplotlib, pytest và hệ điều hành đang dùng. Thêm SciPy, Matplotlib vào dependencies phục vụ đánh giá; chốt phiên bản tương thích Python/NumPy của dự án khi cài.
2. Tái hiện TC-WEB-01 bằng expected công khai. Nếu không khớp, điều tra phiên bản/dtype trước khi dùng reference.
3. Viết fixture input; tạo expected bằng công thức giải tích hoặc SciPy và lưu manifest. Đối chiếu case impulse giữa hai nguồn expected trước khi dùng cho production test.
4. Viết các test contract, baseline, edge case trước khi sửa implementation; ghi nhận những test hiện chưa đạt.
5. Hoàn thiện function theo mục 2.
6. Chạy test tập trung và kiểm tra hồi quy Module 2.
7. Chạy script đánh giá, xuất plot và mở ảnh để kiểm tra trực quan.
8. Ghi kết quả thực đo vào mục 6 của tài liệu này.

Lệnh dự kiến chạy từ root repository sau khi đã tạo các file:

```powershell
python -m pytest tests/reference/test_gaussian_reference.py -v
python scripts/generate_soft_mask_baselines.py --output tests/fixtures/soft_mask
python -m pytest tests/unit/test_create_soft_mask.py tests/unit/test_region_engine.py -v
python scripts/evaluate_soft_mask.py --fixtures tests/fixtures/soft_mask --output artifacts/module-2/create-soft-mask
python -m ruff check src/region_engine/mask_utils.py tests/unit/test_create_soft_mask.py tests/reference/test_gaussian_reference.py scripts/generate_soft_mask_baselines.py scripts/evaluate_soft_mask.py
```

Trước khi bàn giao code, chạy thêm `python -m pytest tests/unit/ -v` theo quy trình repo và phân biệt lỗi có sẵn/lỗi môi trường với regression do thay đổi này.

## 5. Lưu plot trước/sau để xem

Thư mục dự kiến: `artifacts/module-2/create-soft-mask/`.

```text
environment.json
metrics.csv
summary.json
report.md
plots/
    step_comparison.png
    step_profile.png
    rect_comparison.png
    touch_edge_comparison.png
    impulse_comparison.png
    thin_comparison.png
    feather_parameter_comparison.png
arrays/
    <case_id>_actual.npy
```

Quy cách xuất ảnh:

- Mỗi `comparison.png` có 4 ô: **binary input — output thực tế — baseline — absolute error**.
- Ba ô mask dùng cùng `cmap="gray"`, `vmin=0`, `vmax=1`, `interpolation="nearest"`. Không dùng auto-contrast riêng cho từng ô.
- Error map có colorbar riêng và ghi `E_max`, tránh hiểu màu sai số phóng đại là sai khác lớn.
- `step_profile.png` vẽ một hàng giữa ảnh với input/output/baseline trên cùng trục, alpha từ 0 đến 1.
- `feather_parameter_comparison.png` dùng cùng RECT với `f=0,3,15,25` và ghi cả `k`, sigma trong tiêu đề.
- Lưu bằng Matplotlib backend `Agg`, DPI khoảng 160–200; đóng figure sau khi lưu. Mở kiểm tra plot đã lưu, không chỉ kiểm tra file tồn tại.
- Lưu `.npy` để giữ chính xác dữ liệu float; PNG chỉ là hình xem. Không đọc ngược PNG 8-bit để tính metric.
- Có thể thêm minh họa ghép hai ảnh màu hằng bằng hard mask/soft mask để thấy đường chuyển tiếp. Đây chỉ là demo ứng dụng, không thay thế test baseline và không mở rộng phạm vi sửa `blend_regions`.

“Trước/sau” của function này là **mask nhị phân trước xử lý / mask mềm sau xử lý**, vì function không nhận ảnh RGB.

## 6. Kết luận và báo cáo kết quả

### 6.1 Kết luận thuật toán ở giai đoạn lập kế hoạch

Gaussian filtering phù hợp yêu cầu tạo alpha liên tục từ binary mask. Kết quả phụ thuộc encoding, kích thước kernel, sigma và cách xử lý biên; phải cố định đủ bốn yếu tố để có baseline tái lập được.

Giới hạn: feathering chỉ làm mềm vùng đã có, không sửa sai phân đoạn. Nó lan trọng số ra ngoài biên nhị phân và có thể làm giảm alpha tối đa của vùng mảnh/nhỏ. Nó giúp giảm độ gắt khi ghép nhưng không bảo đảm xóa mọi halo hoặc chênh lệch màu. Việc đúng số học không đồng nghĩa mọi ảnh ghép đều đẹp.

### 6.2 Kết quả thực nghiệm

| Hạng mục | Kết quả hiện tại | Bằng chứng cần bổ sung |
|---|---|---|
| Nguồn thuật toán | Đã tra cứu OpenCV chính thức | Các link ở mục 1 |
| Test công khai | TC-WEB-01 pass | `1 passed` trong `tests/reference/test_gaussian_reference.py` |
| Implementation hoàn thiện | Đã thực hiện | `ruff check` pass; contract và Gaussian config đã cập nhật |
| Test trực tiếp binary mask | 28 pass; fixture 20/20 pass | `pytest` nhóm Module 2 và `tests/unit/test_soft_mask_fixtures.py` |
| Sai số lớn nhất / MAE / RMSE | `1.145438606e-7` / `1.479433137e-8` / `2.846199019e-8` | Case `rect`, trong `artifacts/module-2/create-soft-mask/metrics.csv` |
| Plot trước/sau | Đã tạo và mở kiểm tra | `artifacts/module-2/create-soft-mask/plots/` |
| Đánh giá trực quan | Đạt | STEP có chuyển tiếp đơn điệu; vùng biên và corner khớp baseline; tham số lớn làm dải chuyển tiếp rộng hơn |
| Kết luận nghiệm thu | Đạt trong phạm vi contract | 20/20 case trong `atol=1e-6`; xem `summary.json` và `report.md` |

### 6.3 Điều kiện hoàn thành

- [x] TC-WEB-01 tái hiện đúng output công khai.
- [x] Case impulse khớp baseline giải tích.
- [x] Tất cả case hợp lệ khớp baseline độc lập trong tolerance và đúng contract.
- [x] Tất cả case không hợp lệ phát sinh đúng exception.
- [x] Test hồi quy liên quan đạt, không đổi kết quả hợp lệ ngoài kế hoạch.
- [x] Fixture có nguồn gốc, cấu hình, phiên bản và hash rõ ràng.
- [x] Có metrics, dữ liệu float và plot đã mở xem.
- [x] Báo cáo ghi rõ case fail nếu có; không có case fail trong phạm vi đã chạy.

Kết luận thực nghiệm: Đã chạy 20 case fixture SciPy, 20 đạt và 0 không đạt ở `atol=1e-6`. Sai số cực đại là `1.145438606e-7`, MAE `1.479433137e-8`, RMSE `2.846199019e-8` ở case `rect`. Case impulse cũng khớp baseline giải tích với `E_max=1.296377847e-8`. Với `k` là số lẻ sau quy đổi, `sigma=k/3`, border `BORDER_REFLECT_101`, implementation đạt contract. Giới hạn: sai khác float32 nhỏ hơn baseline float64; feathering vẫn có thể làm giảm alpha cực đại của vùng rất mảnh/nhỏ.
