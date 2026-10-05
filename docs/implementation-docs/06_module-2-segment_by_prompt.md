# Kế hoạch triển khai segment_by_prompt — Module 2

Ngày lập: 2026-09-21. Trạng thái: **đã triển khai contract, command routing, lazy backend seam, union/feather, caller, mock tests/fixtures/scripts/artifact; real-model chưa nghiệm thu vì chưa có weights/dataset**.

API: `segment_by_prompt(image, text_prompt, feather_radius=15) -> np.ndarray`.

Mục tiêu: dùng prompt để tìm đối tượng bằng GroundingDINO, phân đoạn bằng MobileSAM, hợp nhất các instance phù hợp và feather một lần. Output là mask toàn ảnh float32, finite, `(H,W)`, trong `[0,1]`, dùng được với Module 3. CPU là cấu hình nghiệm thu mặc định.

## 1. Nguồn trình bày thuật toán

| Nguồn chính thức | Vai trò |
|---|---|
| [GroundingDINO — repository tác giả](https://github.com/IDEA-Research/GroundingDINO) | Detector nhận ảnh và văn bản; lọc bbox theo độ tương đồng với token/phrase |
| [Grounded Segment Anything — repository tác giả](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Nguồn sát nhất với toàn pipeline text → bbox → segmentation |
| [Hugging Face Transformers — Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino) | API thực thi dự kiến, processor ảnh/văn bản và postprocess tọa độ |
| [MobileSAM — repository tác giả](https://github.com/ChaoningZhang/MobileSAM) | Segmenter nhẹ, checkpoint TinyViT và SamPredictor |
| [MobileSAM predictor.py](https://github.com/ChaoningZhang/MobileSAM/blob/master/mobile_sam/predictor.py) | Quy ước bbox XYXY, set_image/predict và mask trở lại kích thước ảnh |
| [Meta SAM 2](https://github.com/facebookresearch/sam2) | Backend thay thế để đánh giá sau, không bắt buộc cài trong giai đoạn đầu |

Tra cứu ngày 2026-09-21. Không có một thuật toán riêng tên `segment_by_prompt` định nghĩa đầy đủ contract của dự án. Nguồn Grounded SAM trình bày cách ghép detector và segmenter; thay SAM bằng MobileSAM, chọn instance, fallback và feather là quyết định tích hợp cần kiểm thử riêng.

GroundingDINO tìm vị trí theo ngôn ngữ; MobileSAM nhận bbox/point, không trực tiếp hiểu text. Không gọi riêng MobileSAM là text segmentation. Ảnh demo trong README có overlay không phải mask ground truth để đo độ chính xác.

## 2. Triển khai thuật toán

### 2.1. Hiện trạng repository và thay đổi đã triển khai

- `src/region_engine/detector.py` đã bỏ fallback heuristic cho semantic prompt; chỉ exact command hình học/toàn ảnh mới không load model. Prompt semantic được normalize NFC, gộp whitespace, alias exact và giới hạn 256 token ước lượng.
- `src/region_engine/segmentation_backend.py` đã thêm adapter GroundingDINO Tiny + MobileSAM local trên CPU, lazy cache/lock/reset, threshold `0.35/0.25`, NMS IoU `>0.8`, lỗi rõ ràng và không tải mạng ngầm.
- Luồng semantic giữ bbox XYXY pixel, clip/reject bbox rỗng, set image một lần, predict từng bbox, OR union toàn ảnh, rồi gọi `create_soft_mask` đúng một lần.
- Executor nhận zero semantic mask thì bỏ qua action hiện tại; exception backend được truyền lên, không fallback toàn ảnh. Test heuristic cũ trong quadrant đã được gỡ để quadrant chỉ kiểm tra geometry.
- `create_soft_mask`, `blend_regions`, bbox và quadrant được tái sử dụng; không sửa thuật toán Gaussian trong phạm vi này.

### 2.2. Chốt backend và phạm vi

Giai đoạn đầu dùng **GroundingDINO Tiny qua Transformers + MobileSAM vit_t**, CPU float32, `eval()` và `torch.inference_mode()`. Chọn Transformers để tránh buộc người dùng Windows phải build extension của implementation GroundingDINO gốc; cấu hình `disable_custom_kernels=True` và xác minh inference CPU thực tế trên phiên bản được pin. Đây là hướng triển khai, chưa phải khẳng định môi trường hiện tại đã chạy được.

SAM2 là tùy chọn giai đoạn sau: adapter riêng, checkpoint và test riêng; không tự đổi backend khi MobileSAM lỗi. Không yêu cầu triển khai đồng thời cả hai segmenter để hoàn tất giai đoạn đầu.

Giữ public signature ba tham số. Cấu hình model/threshold đặt trong config nội bộ có version và manifest; test inject fake adapter qua factory nội bộ thay vì thêm tham số model vào public API.

### 2.3. Contract input, prompt và fallback

| Thành phần | Quy định đề xuất |
|---|---|
| Image | ndarray RGB uint8 `(H,W,3)`, H/W dương; sai kiểu/dtype → TypeError, sai shape/rỗng → ValueError |
| Noncontiguous/read-only | Nhận; copy contiguous khi backend cần, không sửa input |
| Text prompt | String không rỗng sau strip; sai kiểu → TypeError, chuỗi trống → ValueError |
| Chuẩn hóa | Unicode NFC, gộp whitespace, strip, lowercase; giữ các thuộc tính có nghĩa |
| Alias tiếng Việt | Mapping exact sau chuẩn hóa: `người`→`person`, `bầu trời`/`trời`→`sky`, `mèo`→`cat`, `chó`→`dog` |
| Prompt khác | Truyền như mô tả đối tượng; không khẳng định hỗ trợ tiếng Việt tổng quát hay tự gọi dịch vụ dịch |
| Phạm vi ngôn ngữ | Một đối tượng hoặc cụm mô tả mỗi lần gọi; chưa bảo đảm phủ định, đếm, quan hệ phức tạp hoặc nhiều câu |
| Prompt quá dài | Kiểm tra token theo giới hạn model đã khóa; ValueError trước inference, không cắt ngầm phần mô tả |
| Feather | Integer không âm, loại bool, theo contract hiện có; validate trước load model |
| Không có bbox đạt ngưỡng hoặc mọi mask hợp lệ đều rỗng | Mảng zero float32 `(H,W)` |
| Thiếu model/dependency hoặc inference lỗi | Exception rõ ràng; không trả zero/full/geometric mask để che lỗi |
| Output | Một mảng mới float32 `(H,W)`, finite, `[0,1]`, union tất cả instance của prompt |

Để giữ tương thích, chỉ các **exact command** `full`, `all`, `toàn`, `toàn bộ`, `full_image` trả ones; `top`, `bottom`, `left`, `right`, `center`, `giữa` gọi quadrant tương ứng. Validate ảnh và feather trước command. Không load neural model cho command hình học. Không dùng substring: `all people` và `person in the center` là semantic prompt, không chọn toàn ảnh hoặc center ngầm.

`sky`, `ground`, `floor`, `background` là prompt semantic; không tự ánh xạ sang nửa ảnh hoặc lấy phần bù khi model thất bại. Đặc biệt sky là vùng nền rộng, cần bộ test semantic riêng; pass trên person không đồng nghĩa pass trên sky.

Đề xuất `SegmentationUnavailableError(RuntimeError)` cho dependency/checkpoint/init và `SegmentationInferenceError(RuntimeError)` cho inference/output sai. Giữ nguyên cause bằng `raise ... from exc`. Kết quả zero chỉ cho pipeline chạy thành công mà không chọn được đối tượng. Không dùng `except Exception: return ones/zeros`.

### 2.4. Dependency, checkpoint và lifecycle

1. Kiểm tra Python/Windows/architecture và bộ Torch CPU + torchvision tương thích; thêm optional extra `segmentation` hoặc file requirements riêng có hướng dẫn rõ.
2. Thêm Transformers, tokenizer dependencies, MobileSAM từ commit cố định và dependency của MobileSAM; kiểm chứng với NumPy/OpenCV hiện có. Pin phiên bản sau smoke test, không chọn version chưa thử rồi ghi “đã hỗ trợ”.
3. Chuẩn bị snapshot `IDEA-Research/grounding-dino-tiny` gồm weights, config, tokenizer, image processor; chuẩn bị MobileSAM `mobile_sam.pt`. Tải bằng script riêng, lưu revision/hash/license/source URL trong manifest.
4. Runtime dùng local path dưới `models/segmentation/`, chỉ đọc file đã có; không tải model/tokenizer ngầm trong public function. Không sử dụng remote inference API.
5. Lazy-load/cached model một lần mỗi process. Model cache theo backend/config/model hash; init thất bại không được cache thành “không tìm thấy”.
6. MobileSAM predictor có state ảnh: giữ lock cho toàn bộ `set_image → predict mọi bbox → reset_image` và reset trong finally. Không để hai request trộn embedding của hai ảnh. Không cache embedding bằng object id vì ảnh có thể bị sửa giữa lần gọi.
7. Có close/reset cho test và application shutdown. Import Module 2 mask utilities không buộc load torch/checkpoint.

### 2.5. Luồng inference và hệ tọa độ

1. Validate input, xử lý exact command nếu có; semantic prompt thêm dấu chấm kết thúc theo định dạng backend, ví dụ `person.`.
2. Processor tạo tensor từ ảnh RGB nguyên kích thước và prompt; lưu `(H,W)` gốc.
3. GroundingDINO inference CPU. Cấu hình ban đầu: box threshold=0.35, text threshold=0.25; khóa trước evaluation, không chỉnh theo từng ảnh test.
4. Dùng `post_process_grounded_object_detection` với target_sizes `[(H,W)]`. Adapter trả bbox **XYXY pixel float** cùng score/phrase. Với backend HF không chuyển lại normalized cxcywh. Nếu sau này dùng repository gốc, cần adapter riêng vì output raw có quy ước khác.
5. Reject output NaN/Inf hoặc shape sai bằng exception. Clip bbox vào miền ảnh, bỏ bbox mất diện tích. Không expand bbox mặc định như face detector. Lọc phrase rỗng; không đòi phrase phải bằng nguyên prompt vì thuộc tính/token có thể được trích khác.
6. Sắp theo score giảm dần rồi tọa độ; NMS class-agnostic IoU>0.8 cho một prompt để bỏ bbox gần trùng. Không âm thầm lấy top-1; lưu mọi detection được giữ trong báo cáo. Bbox rỗng sau lọc → zero mask, không cần chạy SAM.
7. `predictor.set_image(rgb)` một lần cho ảnh. Với từng bbox gốc float, dùng public `predict(box=xyxy, multimask_output=False)`; predictor tự chuyển bbox sang hệ tọa độ encoder. Không transform bbox thêm lần hai. Dùng API tensor batch chỉ khi đã kiểm tra `apply_boxes_torch` tương ứng.
8. Yêu cầu mỗi mask trả về đúng `(H,W)`; dùng output full-resolution của predictor, không dùng low-resolution logits làm output cuối. Với API mặc định nhận boolean mask; nếu adapter đổi sang logits phải threshold theo model contract, không coi logits là trọng số `[0,1]`.
9. Hợp nhất boolean masks bằng OR trên một accumulator toàn ảnh. Không cộng mask để vùng overlap vượt 1, không chọn mask đầu tiên, không blur từng instance trước union.
10. Gọi `create_soft_mask(union_uint8, feather_radius)` đúng một lần; kiểm tra contract cuối và trả mảng mới.

Với `multimask_output=False`, mỗi box chỉ có một mask; điểm chất lượng của SAM chỉ ghi diagnostic ở giai đoạn đầu, không tự đặt thêm ngưỡng bỏ mask chưa hiệu chuẩn. Mask sai kích thước/kiểu là lỗi adapter, không resize âm thầm để che sai hệ tọa độ.

Mặc định feather=15 nghĩa là k=15, sigma=5, `BORDER_REFLECT_101`; tên tham số lịch sử không phải bán kính kernel. r=0/1 không blur, số chẵn dương tăng lên số lẻ kế tiếp. Không dùng feather để cải thiện metric segmentation nhị phân.

### 2.6. Module 3 và migration test

Executor nhận zero mask thì bỏ qua action hiện tại, tiếp tục action sau; không đổi zero thành None. Exception model phải được báo theo error flow, không fallback full-image. Các exact command được giữ như trên; cập nhật README để phân biệt hình học với ngữ nghĩa.

Thay test heuristic sky/ground trong unit quadrant bằng test direct quadrant hoặc fake semantic adapter trong test detector. Giữ test center không cần model. Không đổi assertion chỉ để chấp nhận heuristic cũ; phải kiểm tra sky thật không còn tự chọn nửa trên.

## 3. Testcase trên mạng và baseline

### 3.1. Dataset chính: Penn-Fudan, prompt person

Nguồn [Penn-Fudan Pedestrian Database](https://www.cis.upenn.edu/~jshi/ped_html/) cung cấp ảnh và mask người. [Tutorial chính thức PyTorch](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) minh họa rõ cặp:

```text
Input:    PNGImages/FudanPed00046.png
Prompt:   person
Baseline: PedMasks/FudanPed00046_mask.png
```

Mask annotation mã hóa instance, 0 là nền. Baseline binary union của API này là `GT = (label_mask != 0)`, shape giữ nguyên, không đọc palette PNG thành RGB rồi threshold màu hiển thị. Với feather=0, expected là GT float32; **metric model đo gần GT, không yêu cầu đúng từng pixel như hàm hình học**.

Thêm `FudanPed00001.png` ↔ `FudanPed00001_mask.png` và các cặp cùng tên được liệt kê trong tutorial. Sau tải, kiểm tra file/shape, số instance, hash rồi ghi manifest; không tự điền số người hoặc pixel area khi chưa đọc mask. Hai tên trên là fixture cụ thể có đường dẫn baseline công khai, không phải output do model dự án sinh.

Bộ định lượng: sắp tất cả ID hợp lệ theo tên; lấy 10 cặp đầu làm calibration và 20 cặp kế làm evaluation, disjoint. Khóa danh sách/hash trước nhìn prediction. Case 00046 dùng minh họa riêng; nếu xuất hiện trong split khác phải ghi rõ, không gộp hai lần. Tính metric union toàn bộ người, đồng thời báo lỗi bỏ sót instance nhỏ.

### 3.2. Regression detector từ ví dụ công khai

[Ví dụ Transformers Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino) dùng COCO `val2017/000000039769.jpg`, model tiny, lớp `a cat`/`a remote control`, threshold 0.4/0.3 và có output bbox cat `[344.78,22.90,637.30,373.62]`, score khoảng 0.468.

Đây là baseline bbox **của ví dụ model**, không phải mask GT. Tái hiện đúng prompt/config/version của ví dụ ở adapter test riêng; không so prompt `cat` với output multi-class rồi yêu cầu giống hệt. Chỉ dùng kiểm tra hệ tọa độ/đường chạy, không dùng số bbox này thay nhãn phân đoạn. Ngưỡng regression dự kiến IoU>=0.95 cho box match; score chỉ diagnostic vì nguồn in số đã làm tròn và chưa khóa revision.

Nếu dùng ảnh này để đánh giá cat segmentation, phải tải annotation COCO tương ứng, lấy union masks đúng category/annotation ID và ghi manifest. Không vẽ bbox tô kín làm segmentation baseline. Đây là mở rộng, không thay baseline Penn-Fudan bắt buộc.

### 3.3. Sky cần baseline semantic riêng

Nguồn [MIT Scene Parsing / ADE20K devkit](https://github.com/CSAILVision/sceneparsing) có ảnh và semantic labels. Đề xuất lấy 5 ảnh validation có sky và 5 ảnh không sky theo thứ tự tên, xác định bằng class metadata + label map trước inference.

Baseline `GT=(label_map == sky_id)` trên pixel valid; tra ID từ metadata của bản dataset đã tải, không đoán ID hoặc tự trừ 1. Void/ignore pixel loại khỏi metric. Manifest phải chứa ID ảnh, đường label map, class mapping, valid mask và hash. Hiện mới xác minh nguồn dataset; chưa tải/chốt ID cụ thể, nên nhóm này chưa được tính là fixture đã chuẩn bị.

Nhóm sky bắt buộc trước khi kết luận hỗ trợ `sky`; nếu chưa chuẩn bị/chạy được thì chỉ nghiệm thu person và ghi rõ sky chưa nghiệm thu. Không thay sky bằng quadrant để đạt đầu ra có vẻ hợp lý.

### 3.4. No-object và ảnh chuyển thể

- Fake GroundingDINO trả 0 boxes: expected zero mask, SAM không được gọi.
- Ảnh đen RGB `(64,96,3)`, prompt `person`: GT zero; inference thật phải báo false-positive area/count nếu có.
- Ảnh negative tự nhiên: chỉ dùng nhãn đầy đủ cho lớp target hoặc xác minh thủ công có ghi provenance. Việc dataset không gán nhãn một lớp ngoài taxonomy không chứng minh lớp đó vắng mặt.
- Case sát mép: từ Penn-Fudan 00046, dùng GT tìm x nhỏ nhất của foreground, crop cả ảnh và label từ cột đó đến hết ảnh; baseline là label được crop cùng phép biến đổi. Đây là fixture chuyển thể, không phải ví dụ nguyên gốc, không lấy detection để chọn crop.
- Case nhiều instance: chọn ảnh đầu tiên theo tên trong tập cố định có ít nhất hai nonzero instance ID; ghi ID trước chạy model. GT union mọi instance, không chỉ đối tượng lớn nhất.

### 3.5. Unit test với baseline tính tay

Ảnh dummy RGB `(4,6,3)`, prompt `person`, feather=0. Fake detector trả hai bbox `(1,0,3,2)` và `(2,1,5,3)`; fake SAM trả M1=1 tại y={0,1}, x={1,2} và M2=1 tại y={1,2}, x={2,3,4}. Union expected:

```text
0 1 1 0 0 0
0 1 1 1 1 0
0 0 1 1 1 0
0 0 0 0 0 0
```

Tổng=9, pixel overlap `[1,2]` vẫn bằng 1. Thêm case bbox trùng qua NMS, bbox ngoài ảnh, score dưới/ngang/trên threshold theo adapter đã pin, phrase rỗng, NaN/Inf, sai mask shape, SAM lỗi sau box thứ hai, prompt whitespace/rỗng/quá token và input readonly.

Kiểm thử riêng adapter tọa độ trên ảnh `(H=100,W=200)`: normalized cxcywh `(0.5,0.5,0.4,0.2)` tương ứng XYXY `(60,40,140,60)`. Test này dành cho phép chuyển raw format nếu adapter có sử dụng; đường HF postprocessed phải giữ nguyên XYXY và không scale lại. Assert MobileSAM nhận bbox gốc float, ảnh đúng RGB, `set_image` đúng một lần, reset cả khi fail.

Golden soft-mask từ union literal bằng SciPy float64, không gọi production: k làm lẻ như contract, `gaussian_filter(union, sigma=k/3, radius=k//2, mode="mirror")`; r=0/1 giữ union. Tham chiếu [SciPy gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html). Đây là baseline hậu xử lý, không đo độ đúng ngữ nghĩa.

### 3.6. Metric và ngưỡng nghiệm thu

| Tầng | Metric | Gate đề xuất |
|---|---|---|
| Contract | Dtype/shape/finite/range/ownership | 100% case hợp lệ đạt |
| Mock hard union | Pixel mismatch/diện tích | 0 mismatch; diện tích fixture =9 |
| Mock soft | MaxAE/MAE với SciPy | <=1e-6 / <=1e-7; báo RMSE |
| Real segmentation | IoU=TP/(TP+FP+FN), Dice=2TP/(2TP+FP+FN) | Person evaluation mean IoU>=0.70, mean Dice>=0.80; gate kỹ thuật đề xuất, chưa phải kết quả |
| Coverage | Tỷ lệ ảnh positive có mask nonempty | >=90% trên person evaluation; không bỏ ảnh miss khỏi trung bình |
| Negative | False-positive pixel area và tỷ lệ ảnh có FP | Ảnh đen smoke phải zero; negative tự nhiên báo riêng |
| Integration | Zero mask giữ ảnh; lỗi propagate; union blend một lần | Fixture hard khớp từng byte |
| CPU | Cold load, DINO, SAM encode/decode, feather, peak RAM | Báo p50/p95; chưa cam kết SLA |

Với sky dùng gate mean IoU>=0.70 trên nhóm có sky như mục tiêu ban đầu, báo riêng nhóm không sky; 10 ảnh là smoke, không đủ chứng minh chất lượng tổng quát. Gate phải khóa trước lượt evaluation; nếu fail ghi chưa đạt, không hạ ngưỡng hoặc đổi test sau khi xem kết quả.

Đo IoU/Dice trên **union binary trước feather**, hoặc public output với feather=0; lưu cả output feather=15 để kiểm tra giao diện. Trên pixel valid, cả GT/pred rỗng: IoU/Dice=1 nhưng báo riêng negative; GT nonempty và pred rỗng: 0. Báo macro trung bình theo ảnh và micro theo tổng pixel riêng, không dùng accuracy nền lớn làm thước đo chính.

Để tìm nguyên nhân: (a) DINO recall bbox với IoU>=0.5 và matching một-một; (b) MobileSAM với bbox GT; (c) pipeline với bbox DINO. Bbox GT từ mask dùng max-index+1 theo quy ước nửa kín. GT-box run là diagnostic, không thay kết quả end-to-end. AP cần instance score/evaluator riêng, không gọi IoU union là COCO AP.

## 4. Test Python và quy trình đánh giá

### 4.1. File cần tạo/sửa

| File | Công việc |
|---|---|
| `src/region_engine/detector.py` | Validation, command routing, semantic pipeline, union và lỗi |
| `src/region_engine/segmentation_backend.py` | Adapter DINO/MobileSAM, config/cache/lock/reset |
| `src/region_engine/README.md` | Contract, model setup, prompt/fallback và giới hạn |
| `src/agent/executor.py` | Zero semantic mask → bỏ qua action; lỗi không thành full-image |
| `pyproject.toml` và requirements tùy chọn | Extra segmentation và bộ version đã kiểm chứng |
| `tests/unit/test_segment_by_prompt.py` | Fake model, input, union, tọa độ và lifecycle |
| `tests/integration/test_prompt_segmentation_model.py` | Model thật + baseline dataset |
| `tests/integration/test_prompt_region_processing.py` | Module 3 và executor |
| `tests/fixtures/prompt_segmentation/` | Manifest, mock literal, hash và ground truth |
| `scripts/prepare_prompt_segmentation_assets.py` | Model/tokenizer/checkpoint/dataset preparation |
| `scripts/generate_prompt_mask_baselines.py` | Mask GT và mock soft oracle độc lập |
| `scripts/evaluate_prompt_segmentation.py` | Metric, intermediate output, ảnh và plot |

Lệnh dự kiến sau khi triển khai script:

```powershell
python scripts/prepare_prompt_segmentation_assets.py --manifest tests/fixtures/prompt_segmentation/assets.json
python scripts/generate_prompt_mask_baselines.py
python -m pytest tests/unit/test_segment_by_prompt.py -q
python -m pytest tests/integration/test_prompt_region_processing.py -q
python -m pytest tests/integration/test_prompt_segmentation_model.py -m real_model -q
python scripts/evaluate_prompt_segmentation.py --manifest tests/fixtures/prompt_segmentation/assets.json --device cpu --mock-only --output-dir artifacts/module-2/segment-by-prompt
python scripts/evaluate_prompt_segmentation.py --manifest tests/fixtures/prompt_segmentation/assets.json --device cpu --output-dir artifacts/module-2/segment-by-prompt-real
python -m pytest tests/unit/test_create_soft_mask.py tests/unit/test_blend_regions.py tests/unit/test_create_bbox_mask.py tests/unit/test_create_quadrant_mask.py tests/integration/test_region_blending.py -q
```

Đã tạo các script/test nêu trên. Unit mock chạy không mạng/không weights;
profile real-model phải fail rõ nếu thiếu asset/dependency, không skip rồi báo
hoàn thành. `--mock-only` chỉ kết luận union/feather plumbing; không đại diện
cho chất lượng semantic. Lưu log, exit code, JUnit XML và số pass/fail/skip.

Manifest lưu revision và hash model, tokenizer, processor, MobileSAM commit, image/label hash, prompt trước/sau chuẩn hóa, thresholds/NMS, resize config, seed, split, CPU/OS và versions. Pin dữ liệu trước inference; script evaluation không overwrite GT hoặc golden khi fail.

CPU benchmark tách thời gian tải model/decode ảnh khỏi warm inference, 3 warm-up và 10 lượt đo trên một fixture cố định; ghi số bbox và kích thước ảnh. Nếu thời gian quá lớn, báo số lượt đã thực hiện thực tế; không dùng thời gian GPU từ paper làm baseline CPU.

## 5. Save plot và ảnh trước/sau

Thư mục dự kiến: `artifacts/module-2/segment-by-prompt/`.

```text
segment-by-prompt/
  README.md
  run_manifest.json
  metrics.csv
  metrics.json
  detections.json
  test-results.xml
  logs/
  arrays/                 # GT, binary union, soft, mock reference .npy
  images/                 # original, processed, blended PNG từng case
  plots/
    01_person_gt_vs_prediction.png
    02_multi_instance_union.png
    03_edge_object.png
    04_negative_unchanged.png
    05_sky_gt_vs_prediction.png
    06_before_after.png
    07_mock_error_and_edge_profile.png
    08_cpu_timings.png
```

Mỗi fixture có original sạch, GT, bbox DINO+score, từng SAM mask, binary union, soft-mask và overlay FP/FN/TP. Plot sky chỉ tạo khi đã có fixture và chạy thật; không sinh ảnh giả thay cho kết quả còn thiếu.

Demo processed=`clip(int16(original)+40,0,255).astype(uint8)`; blend một lần với mask. Plot so original, xử lý toàn ảnh, ghép hard, ghép soft; no-object giữ nguyên ảnh. Ghi prompt chuẩn hóa, model/config, shape và feather. Mask thang cố định `[0,1]`, categorical GT dùng nearest, heatmap có colorbar.

Lưu float `.npy` để đo; PNG chỉ để xem. Soft error/profile dùng mock cùng binary input để đánh giá Gaussian; semantic FP/FN dùng GT người gán, không trộn hai loại lỗi. Dùng matplotlib Agg, kiểm tra RGB, nhãn không bị cắt và ảnh đúng ID trước khi giao. Mở ít nhất person/multiple/negative/before-after, cùng sky nếu có; README dẫn tới toàn bộ artifact và lệnh tái tạo.

## 6. Kết luận thuật toán và kết quả

Phương án chọn là GroundingDINO Tiny → bbox pixel → MobileSAM → union binary → Gaussian feathering, chạy CPU. Không tìm thấy đối tượng trả zero mask để giữ ảnh; lỗi model được báo riêng. Các lệnh chọn vùng hình học chỉ được dùng khi người gọi yêu cầu exact command.

Giới hạn: kết quả phụ thuộc prompt, detector và segmenter; đối tượng nhỏ/che khuất/vùng stuff như sky có thể khó; CPU có thể chậm dù segmenter nhẹ. Dataset nhỏ và prompt person không đủ chứng minh mọi prompt đều đúng. Ground truth binary không phải alpha matte vật lý, nên soft-mask là quy ước ghép ảnh của dự án.

Kết quả hiện tại: đã triển khai routing prompt, validation, backend adapter,
NMS, union/feather, lifecycle/error contract, executor behavior và bộ mock
fixture/evaluator. Real-model chưa được tuyên bố nghiệm thu vì môi trường chưa
có snapshot GroundingDINO, checkpoint MobileSAM hoặc dataset Penn-Fudan.
Artifact hiện có chỉ là mock plumbing, không phải semantic inference thật.

| Kết quả cần cập nhật | Trạng thái |
|---|---|
| CPU dependencies/checkpoint/tokenizer chạy được | Chưa xác minh runtime; asset manifest còn placeholder |
| Unit contract/union/soft oracle | 21 semantic unit pass; mock fixture 3/3 pass |
| Person dataset IoU/Dice/coverage | Chưa đo; chưa có Penn-Fudan asset |
| Sky positive/negative | Chưa chuẩn bị fixture cụ thể/chưa chạy |
| Negative và model-error fallback | Mock empty + error propagation đã pass |
| Module 3/executor regression | 3 prompt integration pass; regression Module 2 chạy riêng |
| CPU latency/RAM | Đã ghi mock timing trong artifact; chưa phải real-model SLA |
| Plot đã mở kiểm tra | Đã sinh mock plots; không sinh plot sky giả |
| Nghiệm thu | Hậu xử lý mock đạt; real semantic còn chờ asset/runtime |

Checklist:

- [x] Tìm tài liệu gốc và baseline công khai có input/mask rõ ràng.
- [x] Lập contract, backend, fallback, metric và quy trình test/plot.
- [ ] Chuẩn bị dependency/model/fixture, pin revision/hash và smoke CPU.
- [x] Triển khai pipeline và cập nhật caller/test heuristic cũ.
- [x] Chạy unit, integration mock, evaluation mock và regression liên quan.
- [x] Xuất/mở kiểm tra artifact mock; không tạo ảnh giả cho sky/real model.
- [x] Ghi kết luận; nêu rõ real-model, dataset và benchmark chưa chạy.
