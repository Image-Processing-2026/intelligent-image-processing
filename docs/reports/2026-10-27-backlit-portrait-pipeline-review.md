# Rà soát pipeline AI Image Doctor: chân dung ngược sáng và tóc bị nâng thành xám

Ngày: 27/10/2026, múi giờ Asia/Bangkok. Phạm vi: điều tra và đề xuất, chưa triển khai sửa pipeline.

## 1. Kết luận chính

Hệ thống chẩn đoán đúng vấn đề chính nhưng chưa đủ cơ sở để kết luận toàn bộ khâu phân tích và đề xuất đã tốt. Trong ca này, vùng lỗi được mô tả quá rộng, kế hoạch nâng sáng toàn bộ người, phép gamma nâng cả tóc đen, và tiêu chí dừng không kiểm tra được sự thay đổi đặc điểm đó.

**Đã tái lập ảnh kết quả chính xác từng pixel** bằng mask Gemini + GrabCut và hai lần gamma 1.6 rồi 1.2. Cùng lúc, bộ kiểm tra tác dụng phụ trả về danh sách rỗng. Vì vậy, đây là một khoảng trống có thể tái hiện trong mục tiêu xử lý và kiểm chứng, không chỉ là nhận xét chủ quan về ảnh.

Không đúng nếu nói hệ thống hoàn toàn không nhìn lại ảnh: planner vòng 2 nhận ảnh gốc và ảnh đã chỉnh vòng 1. Tuy nhiên, đường xử lý một phiên bản không có một bước duyệt trực quan độc lập, bắt buộc trên ảnh cuối trước khi trả kết quả. Critic hiện có phục vụ xếp hạng nhiều phiên bản, chưa làm nhiệm vụ này.

Ưu tiên đề xuất: (1) trạng thái review trung thực và cổng duyệt đầu ra; (2) mục tiêu và ràng buộc theo vùng con; (3) giới hạn tác động cộng dồn, kiểm tra giữ tông đen; (4) benchmark trước khi đổi thuật toán. Không cần thay bằng AI sinh ảnh.

## 2. Nguồn bằng chứng và giới hạn

- Session do người dùng cung cấp: `session-c09a16ccae804c44a69123de7da1a5d0.json`; schema `iid.session-export/1`, session `c09a16ccae804c44a69123de7da1a5d0`.
- Hai PNG người dùng đính kèm, cùng kích thước 800 × 533: ảnh gốc `codex-clipboard-23dc8a08-b979-4d52-bd3f-7d59f29e4596.png` và ảnh kết quả `codex-clipboard-50d2f921-0c25-4c8b-a8a8-850564f8ad43.png`.
- Mã nguồn checkout hiện tại, HEAD `cc6da5fed2dbbf64c546a60d2d4da6be1cb3572d`; workspace có thay đổi chưa commit từ trước. Không có bản chụp đầy đủ code/config tại thời điểm phiên chạy. Độ trùng pixel chứng minh đường gamma/mask được tái lập, không chứng minh mọi thành phần runtime đều giống checkout.
- [CONTEXT.md](../../CONTEXT.md), [ngữ cảnh kiến trúc](../../intelligent_image_processing_context.md), [ADR](../decisions.md). ADR-001 yêu cầu AI chỉ điều phối; ADR-002 dự kiến kết hợp đánh giá no-reference với kiểm tra trực quan. Tài liệu ngữ cảnh còn ghi suy luận thẩm mỹ/nhiếp ảnh nâng cao là phần deferred: cần phân biệt mở rộng tính năng với việc hoàn thiện kiểm chứng MVP.
- Đọc mã, phân tích session, replay classical cục bộ và chạy test hiện có. Không gọi lại Gemini, không thay đổi phiên trong ứng dụng, không chạy lại toàn bộ pipeline qua API.

Trong report: **xác nhận** = có bằng chứng trong log/code hoặc tái lập; **rủi ro** = suy ra từ code, chưa xác nhận xảy ra trong phiên; **đề xuất** = chưa được chứng minh hiệu quả trên bộ ảnh.

## 3. Phiên này đã chạy như thế nào?

Nguồn số liệu mục này: `session.options`, run `analyze.result.diagnosis` và `process.result.history` trong JSON.

| Giai đoạn | Bằng chứng thực tế |
|---|---|
| Ý định | Người dùng chọn `mood=brighten`, sửa `backlit_subject` |
| Chẩn đoán | Portrait, ngược sáng; lỗi `backlit_subject@person`, severity hợp nhất 3 |
| Định vị | `gemini_grabcut`, confidence gán theo backend 0.4; diện tích mask 0.269523 |
| Độ sáng đầu vào | Vùng person L* ≈ 23.57; nền L* ≈ 63.68; đây là số đo vùng, không phải riêng mặt |
| Kế hoạch vòng 1 | `gamma_correct`, target `person`, gamma 1.6; kế hoạch khai báo feather 30 |
| Sau vòng 1 | Theo dõi lỗi đo từ 2 xuống 1; `RE_PROCESS`; vùng person L* ≈ 36.15 |
| Kế hoạch vòng 2 | Cùng vùng person, gamma 1.2 trên kết quả vòng trước |
| Sau vòng 2 | Theo dõi lỗi đo từ 2 xuống 0 so với gốc; `SHIP`; không rollback |
| Đánh giá cuối | Heuristic 78.46/100, BRISQUE 19.5514, NIQE 5.2619, `quality_improved=false`, `side_effects=[]` |
| Phiên bản | `num_variants=1`, không có variant ranking |

Severity 3 ban đầu và số 2 trong theo dõi không phải cùng phép đánh giá: một bên là chẩn đoán hợp nhất VLM + đo, một bên là mức đo được dùng để so trước/sau. Không diễn giải 2 → 0 như một kết luận độc lập rằng mặt đã đẹp tự nhiên.

## 4. Thí nghiệm tái lập

Script hỗ trợ: [reproduce_backlit_hair_case.py](reproduce_backlit_hair_case.py). Script đọc treatment và bbox từ JSON, cố định RNG của OpenCV, dựng mask bằng `resolve_subject_mask(..., use_semantic=False)`, áp tuần tự hai gamma, so với PNG kết quả. Script không gọi mạng và không ghi ảnh.

Chạy tại thư mục repo trong PowerShell:

```powershell
.\.venv\Scripts\python.exe docs/reports/reproduce_backlit_hair_case.py `
  --session 'C:\Users\Tien\Downloads\session-c09a16ccae804c44a69123de7da1a5d0.json' `
  --original 'C:\Users\Tien\AppData\Local\Temp\codex-clipboard-23dc8a08-b979-4d52-bd3f-7d59f29e4596.png' `
  --result 'C:\Users\Tien\AppData\Local\Temp\codex-clipboard-50d2f921-0c25-4c8b-a8a8-850564f8ad43.png'
```

Kết quả đã quan sát: backend `gemini_grabcut`, mask area 0.269523; sai số MAE toàn ảnh **0.0**, trùng pixel **true**. Bộ kiểm tra tác dụng phụ trả `[]` ở từng vòng và cả so sánh gốc → cuối.

ROI tóc được chọn thủ công ở đỉnh đầu: x=[435,490), y=[65,90), 1.375 pixel. Đây là mẫu minh họa có thể kiểm tra lại, không phải segmentation toàn bộ tóc.

| Trung bình kênh trong ROI, thang 0–255 | Gốc | Kết quả |
|---|---:|---:|
| R | 9.696 | 42.920 |
| G | 10.692 | 45.659 |
| B | 16.478 | 58.303 |

SHA-256 của dữ liệu đầu vào để nhận diện lại đúng bộ bằng chứng:

```text
session JSON: e6cfc841fbb9d1c3ba33b8a3a25c21ba65c735aa9ad3f372d2a3b9a3cbc3c50b
original PNG: 9f97d5bad26876c3edec37b6ddb18334a0a2a01740b29812900734e15c22f18d
result PNG:   c95733739864aef754e9008c8f9209b08da809d60e2930a073c1f20df229aef1
```

Các file gốc nằm ngoài repo và ảnh nằm ở thư mục Temp, nên đường dẫn có thể hết hiệu lực. Muốn lưu fixture lâu dài cần chọn nơi lưu phù hợp và giữ nguyên byte; chưa sao chép ảnh cá nhân vào repo trong lần điều tra này.

## 5. Phát hiện và nguyên nhân

### F1 — Xác nhận: vùng đối tượng chưa đủ để đại diện mục tiêu nhiếp ảnh

Defect gắn vào `person`, dù bằng chứng bằng lời nói về khuôn mặt tối. Prompt planner yêu cầu dùng vùng defect làm target. Không có mục tiêu có cấu trúc kiểu “nâng mặt, giữ tóc đen, giữ khối áo”. Kết quả là cả tóc, mặt, áo nhận cùng phép biến đổi.

Nguồn: [prompt Plan](../../src/agent/vlm_diagnostician.py), dòng 58–98; [schema defect/action](../../src/agent/state.py), dòng 76–118 và 154 trở đi; JSON `diagnosis.defects`, `history[].plan`.

### F2 — Xác nhận: gamma RGB nâng mạnh vùng gần đen, tác động cộng dồn

`_raw_gamma` dùng `I_out = 255 × (I_in/255)^(1/gamma)` với LUT từng kênh RGB. Tóc đen có giá trị gần 0 nhưng không bằng 0 nên vẫn được nâng mạnh. Một điểm xám 10 đi qua 10 → 33 → 46. Hai gamma 1.6 và 1.2 gần tương đương gamma 1.92 trong miền float tại nơi mask bằng 1; có mask mềm và lượng tử hóa uint8 thì không đồng nhất tuyệt đối với một gamma duy nhất.

Đây là thay đổi mức tông đo được; không có bằng chứng rằng hệ thống chạy một công cụ nhuộm tóc hay giảm saturation riêng. Biến đổi RGB phi tuyến còn thay đổi tỷ lệ kênh; mức đóng góp của thay đổi sắc độ so với tăng sáng chưa được tách riêng bằng ablation.

Nguồn: [gamma](../../src/processing_engine/exposure_contrast.py), dòng 14–36; [alpha blending](../../src/processing_engine/base.py), dòng 131–210; thí nghiệm mục 4.

### F3 — Xác nhận: tiêu chí ngược sáng chưa đủ kiểm chứng khuôn mặt và tóc

`backlit_measurement` xét L* trung bình vùng, chênh với nền và các ngưỡng. Khi vùng là cả người, giá trị bị trộn bởi tóc và áo. Tăng độ sáng trung bình có thể làm lỗi đo hết dù thuộc tính tóc đã thay đổi. Mask confidence 0.4 là hằng số theo backend, không phải xác suất rằng 40% mask đúng.

Nguồn: [đo ngược sáng](../../src/agent/fusion.py), dòng 443–483; [confidence backend](../../src/region_engine/subject.py), dòng 75–78 và 343–351.

### F4 — Xác nhận: AI nhìn lại ở bước lập kế hoạch, chưa có cổng duyệt ảnh cuối

Vòng 2 planner nhận cả ảnh gốc và hiện tại. Followup perception ưu tiên đo lại, chỉ gọi lại VLM trong một số tình huống còn lỗi mở/tác dụng phụ cần phân xử; session này ghi rõ không cần hỏi lại VLM ở chẩn đoán vòng 2. Sau xử lý, `evaluate_node` gọi evaluator reference/no-reference, không gọi visual reviewer. Nhánh `SHIP` một phiên bản có thể kết thúc ngay.

Nguồn: [image_parts](../../src/agent/vlm_diagnostician.py), dòng 376–389; [followup](../../src/agent/perception.py), dòng 1087–1192; [evaluate và graph](../../src/agent/graph.py), dòng 220–248, 577–595, 644–675.

### F5 — Xác nhận: critic đã có nhưng chỉ xếp hạng nhiều phiên bản

Critic nhận ảnh gốc và các candidate, xem tính tự nhiên và preserve. Tuy nhiên, chỉ được gọi khi có ít nhất hai candidate; schema yêu cầu chọn trong danh sách, không có kết quả “tất cả đều không đạt”. Vì vậy bật nhiều variants cũng không thay thế được một cổng kiểm duyệt có quyền từ chối.

Nguồn: [variant critic](../../src/agent/variants.py), dòng 252–370; JSON `num_variants=1`.

### F6 — Xác nhận: trạng thái review có thể gây hiểu nhầm

`visual_plausibility_passed` mặc định `True`; evaluator no-reference không điền kết quả visual review thực. `vlm_feedback` chứa chuỗi điểm heuristic, không phải nhận xét do VLM viết. Trong export, `eval_score.decision` có thể vẫn là mặc định `SHIP` dù `history[].decision` vòng 1 là `RE_PROCESS`. Cần phân biệt quyết định graph với dữ liệu mặc định của evaluator.

Nguồn: [EvaluationResult](../../src/analyzer_evaluator/schemas.py), dòng 98–106; [return evaluator](../../src/analyzer_evaluator/no_reference_eval.py), dòng 399 trở đi; JSON `history[0]`.

### F7 — Xác nhận: `quality_improved=false` không đồng nghĩa ảnh đã xấu đi

Delta brightness hiện đo khoảng cách trung bình toàn ảnh tới mốc sáng lý tưởng, không đo mục tiêu nâng mặt. Ảnh gốc đã có trung bình toàn ảnh gần mốc đó; cứu người tối có thể làm chỉ số này xấu đi. Khi tracked defects tồn tại, graph ưu tiên tiến triển lỗi mục tiêu, có chủ đích bỏ qua kết luận suy thoái toàn ảnh; hành vi này đã được test.

Không đề xuất sửa bằng “hễ false là rollback”: sẽ làm hỏng chỉnh sáng cục bộ hợp lý. Cần ghép tiến triển lỗi với kiểm tra giữ đặc điểm trên vùng liên quan.

Nguồn: [delta và quality](../../src/analyzer_evaluator/no_reference_eval.py), dòng 182–314; [decide](../../src/agent/graph.py), dòng 480–539; [test quyết định](../../tests/unit/test_decide_defects.py), dòng 117–128.

### F8 — Xác nhận: bộ phát hiện tác dụng phụ chưa bao phủ thay đổi tông tóc

Các loại hiện có: halo, dark noise, banding, channel clipping và oversaturation. Không có kiểm tra chuyển vật vốn đen thành xám hoặc tính đồng nhất vật liệu trước/sau. Trong replay chính xác, danh sách rỗng ở cả so sánh theo vòng lẫn cộng dồn. Chuyển kiểm tra sang ảnh gốc là cần thiết nhưng riêng việc đó chưa giải quyết được ca này.

Nguồn: [side_effects](../../src/analyzer_evaluator/measurement/diff.py), dòng 258–340; mục 4.

### F9 — Xác nhận trong code: feather khai báo và đường thực thi không khớp

Plan ghi `feather_radius=30`, nhưng nhánh semantic có subject bbox đi qua `_resolve_subject_region` rồi `resolve_subject_mask` mà không chuyển tham số này. Subject resolver dùng `DEFAULT_FEATHER_RADIUS=15`. Do đó không được khẳng định mask thực tế đã dùng feather 30 chỉ dựa vào plan. Đây là vấn đề truy vết/thực thi riêng; không phải nguyên nhân chính làm xám phần tóc nằm bên trong mask.

Nguồn: [executor](../../src/agent/executor.py), dòng 239–280; [subject resolver](../../src/region_engine/subject.py), dòng 75, 214–238, 343–351.

### F10 — Xác nhận trong code: preserve còn là danh mục phong cách và so tên vùng

Danh mục preserve hiện có warm/cool tone, low/high key, silhouette, film grain, soft focus, muted/vivid colors; chưa có black point, màu vật thể, texture da/tóc. Guard xét tên vùng bằng nhau hoặc `full`, chưa biểu diễn quan hệ chứa nhau `person ⊃ face/hair`. Chỉ thêm một dòng prompt “giữ tóc đen” chưa tạo được ràng buộc thực thi tương ứng.

Nguồn: [preserve schema](../../src/agent/state.py), dòng 54–65; [preserve guard](../../src/agent/planner.py), dòng 274–350.

### F11 — Rủi ro bổ sung khi trả kết quả, chưa xảy ra trong ca này

- Khi kiểm tra side effects ném exception, wrapper log rồi trả `[]`, nên bên gọi chưa phân biệt được “không phát hiện lỗi” với “không kiểm tra được”. Nguồn: [graph](../../src/agent/graph.py), dòng 342–349. Replay ca này chạy thành công, không có bằng chứng đã gặp exception.
- Rollback đổi `current_image` về ảnh trước nhưng không cập nhật `evaluation_result`; API xuất evaluation hiện có. Có nguy cơ ảnh giao và số đo cuối thuộc hai candidate khác nhau. Phiên này không rollback, nên đây không phải nguyên nhân tóc xám. Nguồn: [graph](../../src/agent/graph.py), dòng 541–574; [record_process](../../src/api/workspace_service.py), dòng 243–300.
- Các đường render lại thực thi recipe trên ảnh gốc rồi trả/lưu ảnh mà chưa có review gate chung. Variant preview có thể khác full-resolution vì segmentation và thao tác không gian phụ thuộc kích thước. Chưa tái hiện sai khác đó trong phiên này. Nguồn: [render API](../../src/api/routes.py), dòng 173–201; [workspace render](../../src/api/workspace.py), dòng 656–690; [preview routing](../../src/agent/graph.py), dòng 577–607.

Ghi chú kiểm toán độc lập: [output evaluation audit](2026-10-27-output-evaluation-audit-notes.md). Report này hợp nhất các phát hiện liên quan; các rủi ro trên cần kiểm thử riêng, không được dùng như bằng chứng rằng chúng đã xảy ra với người dùng.

### F12 — Xác nhận trong code: khuôn mặt có thể dùng mask chữ nhật thay vì đường viền mặt

Người dùng bổ sung quan sát có ảnh vùng tự do sát đối tượng, có ảnh khuôn mặt lại được chỉnh theo cả khung chữ nhật, kéo theo phần không thuộc mặt. Chưa có session/ảnh của ví dụ thứ hai để xác nhận backend cụ thể, nhưng code có các đường trực tiếp tạo ra hành vi này:

- `person`/đối tượng semantic có thể dùng SAM hoặc GrabCut để tạo mask bám đối tượng. Với target `face`, `_infer_region_kind` chuyển sang nhánh face, kể cả khi planner ghi `region_type=semantic`. Nguồn: [executor](../../src/agent/executor.py), dòng 88–104.
- Face mode mặc định là `bbox`. `detect_faces` chuyển bbox thành mask chữ nhật có feather và mặc định nới mỗi phía 15%, tăng mỗi chiều khoảng 30% trước khi clip. Vì thế bbox không chỉ là khung minh họa: bên trong khung là vùng nhận xử lý, có thể gồm tóc/nền/áo. Feather làm mềm biên chứ không loại bỏ pixel không thuộc mặt. Nguồn: [planner](../../src/agent/planner.py), dòng 65–72; [face detector](../../src/region_engine/face_detector.py), dòng 269–360; [controller](../../src/region_engine/controller.py), dòng 428–470.
- Có `face_mode=oval` dựa trên Face Landmarker, nhưng đó là contour hình học khuôn mặt, không phải segmentation da. `sam_refined` chưa được bật. Nếu oval gặp backend unavailable/inference error ở executor, executor thử lại bằng bbox và ghi log. Đây là fallback ở tầng executor, dù controller có thể báo lỗi rõ ràng. Nguồn: [controller](../../src/region_engine/controller.py), dòng 428–512; [fallback executor](../../src/agent/executor.py), dòng 166–179.
- Đường mask dùng cho perception/overlay cũng có thể khác đường action: `_face_subject` gọi `detect_faces(rgb)` trực tiếp, nên ưu tiên bbox mặc định. Không được coi overlay trước xử lý là bằng chứng chắc chắn cho mask oval của action. Nguồn: [subject](../../src/region_engine/subject.py), dòng 330–340.

Đề xuất: tách `detection_box` khỏi `edit_mask`; exposure khuôn mặt nên dùng contour/segmentation có kiểm tra độ phủ và vùng bảo vệ, chỉ dùng bbox làm vùng tìm kiếm. Không tự động áp cùng cường độ lên bbox khi mask chính xác thất bại. Nếu phải hạ chất lượng, cần chính sách rõ ràng (bỏ qua, best-effort bảo thủ, hoặc yêu cầu vùng khác) và xuất metadata. Preview phải hiển thị chính mask thực thi; ghi cả requested mode, actual mode, fallback reason, expansion và feather. Chỉ đổi mặc định sang oval chưa giải quyết đầy đủ vì oval vẫn có vùng không phải da và còn nhánh fallback.

### F13 — Năng lực phân vùng công bố và khả năng dùng làm mặc định

Stack semantic của repo là GroundingDINO Tiny + MobileSAM. [Grounding DINO](https://arxiv.org/abs/2303.05499) công bố open-set detection theo tên lớp/mô tả; không phải danh sách hữu hạn các lớp được bảo đảm. [SAM](https://github.com/facebookresearch/segment-anything) nhận prompt điểm/box và sinh mask; [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) giữ pipeline/prompt decoder tương ứng với encoder nhẹ hơn. Tên vật thể được hiểu ở detector/Gemini, còn SAM giải quyết đường biên từ gợi ý hình học. Khả năng zero-shot không bảo đảm segmentation chính xác mọi đối tượng/bộ phận.

Mask free-form trong session được tái lập là GrabCut, không phải bằng chứng MobileSAM đã chạy. [OpenCV GrabCut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html) phân tách foreground/background từ vùng khởi tạo và có thể cần thêm nhãn sửa, không có danh sách lớp ngữ nghĩa.

Probe `.venv` sau khi nạp `.env` tại thời điểm điều tra: `face=true`, `semantic_models=false`; thư mục DINO và checkpoint MobileSAM tồn tại, `torch`/`transformers` tìm được, `mobile_sam` không tìm được. Đây là bằng chứng môi trường kiểm tra, không khẳng định tiến trình server đang dùng cùng interpreter. Chưa cài dependency, chưa thực hiện real inference MobileSAM.

Có thể đổi policy thành ưu tiên mask bám biên cho đối tượng; bbox vẫn hữu ích làm prompt đầu vào. Với mặt: face detector/landmark → MobileSAM trong ROI → kiểm tra mask → landmark contour/GrabCut phù hợp → bbox chỉ trong trường hợp được chính sách chấp nhận. Với đối tượng khác: Gemini/DINO → MobileSAM → kiểm tra mask → GrabCut → fallback phù hợp hoặc bỏ qua. Không dùng một thứ tự cứng cho mọi đối tượng.

“Fail” phải bao gồm mask sai/rò vùng, không chỉ exception/empty. Diện tích, độ phủ ROI, phần vượt vùng tìm kiếm, số thành phần và vùng bảo vệ là tín hiệu kiểm tra; không đủ để chứng nhận đúng ngữ nghĩa. Cần benchmark riêng face/hair/skin, vì mask cả người hoặc cả đầu không giải quyết tự động nhu cầu chỉ nâng da mặt. Chính sách bbox cuối cùng phải giảm tác động hoặc bỏ qua nếu rủi ro xử lý nhầm lớn; không giữ nguyên tham số mạnh khi fallback.

## 6. Đề xuất thiết kế

### P0 — Hoàn thiện kiểm chứng và tính trung thực của kết quả

1. Thêm `review_output` trước khi chấp nhận candidate cuối, nhận ảnh gốc, candidate, intent, treatment, số đo và crop vùng chỉnh. Phân biệt đạt mục tiêu, giữ đặc điểm, artifact; reviewer có quyền đề nghị sửa, từ chối candidate hoặc báo không đánh giá được. Dùng bằng chứng quan sát ngắn theo vùng, không yêu cầu lộ chuỗi suy nghĩ nội bộ.
2. Trạng thái review nên là `not_run | passed | failed | unavailable`; lưu provider/model, phạm vi ảnh/crop, artifact hash và findings. Chưa chạy hoặc timeout không được trở thành `passed`. Nếu chưa đủ kiểm chứng thì trả kết quả best-effort kèm trạng thái rõ ràng, không lặp vô hạn.
3. `SHIP` cần đồng thời: mục tiêu đạt, các ràng buộc giữ đặc điểm đạt, không có tác dụng phụ nghiêm trọng, review bắt buộc đã pass. Vẫn giữ iteration/call budget; reviewer cũng có thể sai, cần kết hợp đo đạc và benchmark người đánh giá.
4. Tách điểm metric, trạng thái review và quyết định orchestrator; không đặt tên `vlm_feedback` cho nhận xét heuristic. Quyết định cuối gắn với đúng artifact được trả.
5. Sau rollback, chọn lại hoặc tính lại evaluation cho ảnh giữ. Dùng cùng chính sách kiểm chứng cho đường trả ảnh chính, variant và render lại; không suy rộng review preview thành review full-resolution. Tách lỗi kiểm tra tác dụng phụ thành trạng thái unavailable thay vì danh sách rỗng.

Đây là đề xuất kiến trúc, chưa có bằng chứng rằng thêm một VLM call sẽ tự động bắt được mọi lỗi tóc.

### P1 — Cải thiện kế hoạch theo mục tiêu và bảo vệ vật liệu

- Tách `defect_region` khỏi `edit_region` và `protected_regions`; thêm liên kết action → defect để vẫn theo dõi được tiến triển khi chỉnh mặt cho lỗi thuộc person. Việc này cần cập nhật `_targeted_defects`, preserve scope, schemas và executor đồng bộ.
- Với ca này: xác nhận mặt là vùng cần tăng khả năng nhìn; mặt có thể nhận cường độ cao hơn thân, tóc nhận ít tác động hơn. Không mặc định crop mặt hình chữ nhật là mask da: bbox vẫn có tóc, mắt, môi; skin heuristic cũng không phải detector đáng tin trên mọi màu da/ánh sáng.
- Thêm ràng buộc giữ tông đen theo vùng bảo vệ, không chỉ khóa điểm 0. Gamma đã giữ chính xác 0 nhưng vẫn làm sáng mạnh các giá trị 5–30. Đánh giá phân vị luminance, màu và chi tiết trong ROI ổn định; không dùng một ngưỡng đen tuyệt đối cho mọi ảnh.
- Dùng mask chỉnh sửa mềm kết hợp vùng vật thể, độ sáng và confidence. Mọi threshold/cường độ phải hiệu chỉnh bằng dataset; không coi tất cả pixel tối là tóc cần giữ vì như vậy sẽ bỏ sót mặt thiếu sáng.
- Giữ gamma hiện tại cho tương thích; nếu thêm chính sách bảo vệ black point hoặc tone curve, phải có mode rõ ràng, metadata và test riêng. Không âm thầm đổi toán học của công cụ mà playbook đang giả định.

### P1 — Phân tích bối cảnh nhiếp ảnh có cấu trúc trước khi lập kế hoạch

Đề xuất bổ sung bước phân tích cảnh, ánh sáng, vật liệu và ý định để planner có cơ sở chọn cách sửa. Mục tiêu là kết luận có cấu trúc, bằng chứng quan sát và ràng buộc kiểm chứng được; không yêu cầu mô hình xuất chuỗi suy nghĩ nội bộ dài hoặc chỉ thêm câu “hãy suy luận kỹ”. Đây là đề xuất thiết kế từ ca audit, chưa phải năng lực đã triển khai hay đã được benchmark.

| Khía cạnh | Nội dung cần xác định | Tác động đến kế hoạch |
|---|---|---|
| Ánh sáng | Dấu hiệu hướng sáng, vùng trong bóng, chênh sáng chủ thể/nền | Quyết định chỉnh cục bộ hay toàn ảnh và vùng sáng cần giữ |
| Vật liệu và màu vốn có | Vùng tối do tóc/áo đen hay do thiếu ánh sáng | Không biến mọi pixel tối thành mục tiêu nâng sáng |
| Hình khối | Bóng đổ và tương phản tạo khối mặt/đối tượng | Giữ phân bố sáng tối, tránh làm sáng phẳng |
| Ngữ cảnh và ý định | Chủ thể chính, không khí ảnh, phần người dùng muốn thay đổi | Xác định ưu tiên và điều kiện dừng phù hợp |
| Khả năng phục hồi | Chi tiết còn lại, clipping, nhiễu bị lộ khi nâng sáng | Giới hạn can thiệp theo dữ liệu thực có |
| Độ chắc chắn của vùng | Mask có thể lẫn tóc, nền hoặc đối tượng khác | Chọn lại mask, giảm cường độ hoặc bỏ qua thao tác |

Yếu tố vật lý phải được biểu diễn như giả thuyết có bằng chứng và độ bất định: một ảnh đơn không đủ xác định chắc chắn phản xạ vật liệu, nguồn sáng hay phơi sáng gốc. Không suy diễn tóc tối là tóc đen trong mọi ảnh, hoặc nguồn sáng nằm ở một vị trí chính xác khi không có dấu hiệu đủ rõ. Khi các giải thích còn mơ hồ, ưu tiên can thiệp bảo thủ hoặc làm rõ ý định nếu điều đó ảnh hưởng đáng kể kết quả.

Luồng đề xuất:

```text
Quan sát + số đo + ý định
    → phân tích cảnh/ánh sáng/vật liệu và mức bất định
    → mục tiêu thay đổi + vùng bảo vệ + dấu hiệu thất bại
    → lập kế hoạch trong toolbox và kiểm tra mask
    → xử lý
    → review theo đúng mục tiêu/ràng buộc đã đặt
```

Ví dụ đầu ra mong muốn cho ca ngược sáng (schema minh họa, chưa phải API hiện có):

```json
{
  "goal": "Làm rõ khuôn mặt trong ảnh ngược sáng",
  "evidence": ["Mặt tối hơn nền", "Tóc có tông rất tối ở ảnh gốc"],
  "scene_hypotheses": [
    {
      "claim": "Mặt bị thiếu sáng tương đối do điều kiện ngược sáng",
      "confidence": "medium",
      "basis": "Quan sát ảnh và chênh sáng theo vùng"
    }
  ],
  "edit_regions": ["face"],
  "protected_regions": ["hair", "background"],
  "preserve": ["Tông tối tự nhiên của tóc", "Khối sáng tối", "Độ sáng nền"],
  "uncertainties": ["Mask mặt có thể lẫn tóc; cần kiểm tra trước khi áp dụng"],
  "expected_effect": "Mặt rõ hơn trong khi tóc và nền giữ cảm giác tự nhiên",
  "failure_signs": ["Tóc chuyển xám", "Mặt phẳng", "Lộ viền mask"]
}
```

Các nhãn như `hair` phải được phân giải và kiểm tra thành mask thực, hoặc được ghi là chưa định vị được; không chỉ lưu văn bản rồi coi ràng buộc đã có hiệu lực. Schema thực tế cần dùng region ID, evidence/source, mức bất định và tiêu chí kiểm chứng nhất quán. Không dùng confidence tự khai của VLM như xác suất đã hiệu chỉnh.

Nối bước này với phần thực thi: planner phải liên kết action với mục tiêu và vùng được phép chỉnh; validator kiểm tra khả năng toolbox/mask đáp ứng ràng buộc; executor ghi vùng thực sự bị tác động; reviewer đánh giá expected effect và failure signs trên ảnh cuối. Nếu mô hình nói “giữ tóc tối” nhưng vẫn tăng gamma toàn person mà không có cơ chế bảo vệ, bước mới chỉ thêm lời giải thích.

Triển khai trước bằng structured output trong luồng phân tích/lập kế hoạch hiện tại, cùng edit/protected regions và review gate. Chỉ thử tăng ngân sách suy luận hoặc thêm lượt phản biện riêng khi đã có baseline đo lường. Không mặc định thêm nhiều lời gọi AI sẽ tạo ra quyết định tốt hơn.

Nghiệm thu bằng A/B trên bộ hold-out: so baseline với phân tích có cấu trúc, sau đó mới so biến thể có ngân sách suy luận/lượt phản biện bổ sung. Đo mức chọn đúng vùng và công cụ, tỉ lệ sửa đúng ý định nhưng phá đặc điểm khác, bỏ lọt/chặn sai của review, cùng độ trễ và chi phí. Bộ ca cần chứa vật vốn đen trong ảnh đủ sáng, mặt thiếu sáng thật, tóc bạc thật, ánh sáng màu, low-key và ảnh đã tốt. Không chấm chất lượng bằng độ dài phần giải thích.

### P1 — Quản lý tác động cộng dồn và mask thực tế

- Lưu ngân sách nâng sáng/độ biến đổi theo vùng so với ảnh gốc. Các phép gamma cùng vùng chỉ là trường hợp đơn giản; với nhiều mask/công cụ không được chỉ nhân các gamma rồi coi đó là toàn bộ tác động.
- Có candidate gốc làm mốc và giữ candidate tốt nhất đã được đánh giá. Quyết định cuối chọn từ những ảnh đạt các ràng buộc, không mặc định ảnh vòng cuối là tốt nhất; rollback một vòng không đảm bảo chọn được ảnh tốt nhất toàn lịch sử.
- Đo tác dụng phụ theo vòng và cộng dồn. Đồng thời bổ sung loại lỗi còn thiếu như mất tông đen, thay đổi màu vật thể và mất texture; kiểm tra cộng dồn hiện có vẫn không bắt được ca đã tái lập.
- Xử lý chênh lệch feather: hoặc hỗ trợ tham số ở subject resolver/cache key, hoặc báo rõ bị bỏ qua và ghi feather thực tế. Xuất metadata backend, mask hash, vùng thực thi và tham số thực thi.

### P2 — Thử nghiệm thuật toán truyền thống có kiểm soát

Không đề xuất thay gamma bằng CLAHE toàn người một cách máy móc. Có thể benchmark tone curve giữ vùng gần đen, xử lý luminance tách chroma, hoặc chỉnh illumination có bảo toàn cấu trúc. Xử lý riêng luminance cũng không tự giữ màu hoàn hảo: cần kiểm tra gamut, clipping và độ bão hòa.

Hai hướng nghiên cứu có nguồn sơ cấp:

- [LIME, Guo (2016)](https://arxiv.org/abs/1605.05034) ước lượng illumination map rồi tinh chỉnh bằng prior cấu trúc. Đây là căn cứ để thử chỉnh chiếu sáng theo không gian; không bảo đảm tự phân biệt đúng tóc đen và mặt tối trên ca này.
- [Local Laplacian Filters, Paris, Hasinoff, Kautz (2011)](https://people.csail.mit.edu/sparis/publi/2011/siggraph/) trình bày tone mapping và chỉnh chi tiết có xét cạnh. Có thể so với gamma theo vùng về halo, giữ chi tiết và chi phí CPU; chưa cần đưa vào toolbox MVP trước khi đo hiệu quả.

[LIVE, UT Austin](https://live.ece.utexas.edu/research/Quality/nrqa.htm) mô tả BRISQUE/NIQE dựa trên thống kê ảnh tự nhiên. Suy luận thiết kế của report: giữ chúng làm tín hiệu chất lượng hỗ trợ, không xem điểm đó là chứng nhận rằng màu tóc của một chủ thể cụ thể đã được bảo toàn.

## 7. Lộ trình và tiêu chí nghiệm thu đề xuất

| Ưu tiên | Việc làm | Điều kiện chứng minh hoàn thành |
|---|---|---|
| P0 | Sửa trạng thái và nguồn review | Không có trường hợp review chưa chạy nhưng export báo passed |
| P0 | Cổng duyệt candidate cuối | Test reviewer fail/unavailable không bị coi là pass; log trỏ đúng artifact |
| P1 | Edit/protected regions và theo dõi lỗi | Chỉnh face cho defect person vẫn theo dõi đúng; guard hiểu vùng chồng lấn |
| P1 | Phân tích bối cảnh có cấu trúc | Kết luận có evidence/bất định, nối được với action và review; A/B đo chất lượng quyết định thay vì độ dài giải thích |
| P1 | Bảo vệ tông đen/cộng dồn | Ca này cải thiện mặt trong khi giữ tóc đạt tiêu chí đã hiệu chỉnh |
| P1 | Đồng nhất tham số mask | Plan và execution log chỉ rõ feather thực; cache phân biệt cấu hình cần thiết |
| P2 | Đổi/tăng công cụ | A/B benchmark cho thấy lợi ích ngoài ca tóc này và chi phí CPU chấp nhận được |

Bộ đánh giá nên bao gồm: chân dung tóc đen và tóc bạc thật, áo đen, nhiều màu da, mặt trong bóng râm, ngược sáng mạnh, low-key/silhouette có chủ ý, ánh sáng màu, ảnh nhiễu và ảnh đã tốt. Ca này là regression case, không đủ làm tập hiệu chỉnh và tập đánh giá cuối đồng thời.

Tiêu chí gồm: mức nhìn rõ mặt theo ý định; mức giữ tóc/áo và texture; drift màu/halo/clipping trên vùng bảo vệ; tỉ lệ review bỏ lọt và chặn sai; số vòng, độ trễ và chi phí. Dùng người đánh giá trên cặp ảnh đảo thứ tự để kiểm chứng cảm nhận, không chỉ so điểm IQA. Chốt threshold sau một bộ calibration tách khỏi hold-out; report này không đặt số ngưỡng giả định thành tiêu chuẩn đã được xác thực.

## 8. Kiểm chứng đã thực hiện

Đã chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_decide_defects.py tests/unit/test_agent_followup.py tests/unit/test_agent_variants.py -q
```

Kết quả: **50 passed**, 17.85 giây; có một cảnh báo `pyiqa` về bậc tự do của covariance trên ảnh test. Test pass xác nhận hành vi hiện tại tại các seam này, không xác nhận chất lượng nhiếp ảnh của output thực. Replay mục 4 mới là bằng chứng trực tiếp cho ca tóc.

Chưa thực hiện: thay thuật toán, test một phương án sửa tóc, đánh giá reviewer Gemini mới, benchmark nhiều ảnh hoặc đo latency/cost sau thay đổi. Các đề xuất cần được triển khai và nghiệm thu riêng; chưa thể nói lỗi đã được sửa.

## 9. Kết luận dùng cho thảo luận thiết kế

“Hệ thống nhận diện đúng lỗi ngược sáng và thực thi đúng phép biến đổi đã chọn, nhưng mục tiêu xử lý chưa đủ chi tiết theo vật liệu/vùng con và cơ chế chấp nhận kết quả chưa kiểm chứng đầy đủ việc giữ đặc điểm ảnh gốc. Cần hoàn thiện cả kế hoạch, bảo vệ khi xử lý và duyệt đầu ra; không quy toàn bộ vấn đề cho riêng thuật toán gamma hoặc thiếu một lời gọi AI.”
