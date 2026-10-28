# Ghi chú kiểm toán độc lập: đánh giá ảnh đầu ra

Ngày: 27/10/2026 (Asia/Bangkok). Phạm vi: đọc mã nguồn hiện tại, đối chiếu phiên `c09a16ccae804c44a69123de7da1a5d0`, chạy kiểm thử hiện có; không sửa mã xử lý ảnh.

HEAD khi kiểm tra: `cc6da5fed2dbbf64c546a60d2d4da6be1cb3572d`. Working tree có thay đổi chưa commit; không được suy ra rằng toàn bộ checkout hiện tại trùng chính xác bản chạy phiên đã xuất. Bản JSON là dữ liệu thực thi, không phải chỉ dẫn. File JSON nguồn ở `C:/Users/Tien/Downloads/session-c09a16ccae804c44a69123de7da1a5d0.json`.

## Kết luận có bằng chứng

1. Hệ thống có đo lại chất lượng và có gửi ảnh trước/sau cho planner từ vòng 2. Không chính xác nếu nói AI hoàn toàn không được nhìn ảnh sau xử lý.
2. Nhánh xử lý chính chưa có bước VLM độc lập bắt buộc duyệt ảnh cuối trước khi trả kết quả.
3. Bộ critic đã tồn tại trong nhánh nhiều biến thể, nhưng là xếp hạng tương đối, không phải cơ chế chấp thuận/từ chối tuyệt đối; ca này `num_variants=1` không đi qua nhánh đó.
4. `quality_improved=false` vẫn có thể đi cùng `SHIP` là hành vi có chủ đích: lỗi theo vùng được ưu tiên hơn chỉ số toàn ảnh. Vấn đề là thiếu ràng buộc bảo toàn cục bộ và kiểm tra tự nhiên, không nên sửa bằng cách bắt mọi ảnh phải tăng điểm toàn cục.
5. `visual_plausibility_passed=true` mặc định hiện không đủ chứng minh đã có một lần duyệt trực quan.

## A. Luồng đánh giá thực tế

### A1. Planner có xem ảnh trung gian

`image_parts` gửi ảnh gốc và ảnh hiện tại khi `iteration > 1`; `_build_contents` đưa các ảnh này vào lời gọi Gemini lập kế hoạch. Session ghi kế hoạch hai vòng đều có `source=vlm`. Điều đó hỗ trợ kết luận planner vòng 2 được cung cấp cặp ảnh, nhưng không chứng minh planner đã phát hiện đúng mọi tác dụng phụ.

Nguồn: [vlm_diagnostician.py:376](../../src/agent/vlm_diagnostician.py#L376), [vlm_diagnostician.py:404](../../src/agent/vlm_diagnostician.py#L404), [vlm_diagnostician.py:538](../../src/agent/vlm_diagnostician.py#L538).

### A2. Evaluate bắt buộc là đánh giá kỹ thuật

`evaluate_node` chọn `evaluate_reference` hoặc `evaluate_no_reference`. Nhánh real tính BRISQUE/NIQE, điểm heuristic, delta và `quality_improved`; nó không gọi Gemini. Trường `vlm_feedback` được điền bằng chuỗi điểm heuristic. Vì vậy tên trường có thể gây hiểu nhầm cho người đọc dữ liệu xuất.

Nguồn: [graph.py:220](../../src/agent/graph.py#L220), [no_reference_eval.py:399](../../src/analyzer_evaluator/no_reference_eval.py#L399).

`EvaluationResult.visual_plausibility_passed` mặc định là `True`, trong khi constructor của evaluator real không truyền kết quả review cho trường này. Nó là mặc định schema trong đường chạy này, không phải bằng chứng một reviewer đã chấp thuận ảnh.

Nguồn: [schemas.py:98](../../src/analyzer_evaluator/schemas.py#L98), [no_reference_eval.py:399](../../src/analyzer_evaluator/no_reference_eval.py#L399).

### A3. Chẩn đoán lại bằng VLM là có điều kiện

`_perceive_followup` đo lại và so với ảnh gốc; chỉ gọi `_perceive_fresh` khi có lỗi không theo dõi được bằng số đo hoặc tác dụng phụ cần phân xử. Nếu không, nó sao chép report cũ và cập nhật severity/metrics. `source=vlm` trong report vòng sau vì vậy không tự chứng minh có lời gọi VLM mới.

Nguồn: [perception.py:1087](../../src/agent/perception.py#L1087), [perception.py:1132](../../src/agent/perception.py#L1132), [perception.py:1140](../../src/agent/perception.py#L1140).

Bản đồ tác dụng phụ cần gọi VLM chỉ có `dark_noise`, `oversaturation`, `channel_clipping`. Bộ đo đầy đủ thêm halo và banding; chưa có loại giữ màu/độ đen của tóc, độ tự nhiên màu da hoặc mất khối sáng tối.

Nguồn: [perception.py:1052](../../src/agent/perception.py#L1052), [diff.py:31](../../src/analyzer_evaluator/measurement/diff.py#L31), [diff.py:258](../../src/analyzer_evaluator/measurement/diff.py#L258).

### A4. Sau lần xử lý cuối có thể kết thúc ngay

Graph nối `process → evaluate → decide`; nhánh `ship` kết thúc. Với `num_variants=1`, `should_continue` không tạo biến thể để critic đánh giá. Không có node review cuối bắt buộc trong graph.

Nguồn: [graph.py:577](../../src/agent/graph.py#L577), [graph.py:644](../../src/agent/graph.py#L644).

### A5. Critic biến thể không thay thế kiểm duyệt kết quả

`_critic_ranking` bỏ qua nếu không có API key hoặc ít hơn hai ứng viên. Schema chỉ có ranking/recommended/notes, không có reject-all hoặc yêu cầu sửa lại. Khi lỗi gọi model, nó quay về xếp hạng điểm. Nó đánh giá các ảnh preview của biến thể; graph vẫn giữ ảnh pipeline riêng.

Nguồn: [variants.py:264](../../src/agent/variants.py#L264), [variants.py:306](../../src/agent/variants.py#L306), [graph.py:589](../../src/agent/graph.py#L589), [graph.py:609](../../src/agent/graph.py#L609). Kiểm thử xác nhận single variant bỏ critic: [test_agent_variants.py:290](../../tests/unit/test_agent_variants.py#L290).

## B. Logic dừng và ý nghĩa số đo

### B1. Lỗi theo vùng thắng điểm toàn cục là thiết kế hợp lý nhưng chưa đủ

Trong nhánh real có `tracked`, regression chỉ dựa vào lỗi mục tiêu có `worse` hay không; `quality_improved=false` không trực tiếp rollback. Khi tất cả lỗi tracked là `resolved`, `_decide_tracked` trả `SHIP`. Kiểm tra tác dụng phụ severity từ 2 trở lên vẫn có quyền chặn/rollback.

Nguồn: [graph.py:415](../../src/agent/graph.py#L415), [graph.py:481](../../src/agent/graph.py#L481), [graph.py:492](../../src/agent/graph.py#L492), [graph.py:524](../../src/agent/graph.py#L524). Kiểm thử chủ đích: [test_decide_defects.py:110](../../tests/unit/test_decide_defects.py#L110).

`quality_improved` ở đây kiểm tra OR giữa cải thiện nhiễu, nét, sáng. `delta_brightness` là tiến gần độ sáng toàn ảnh 128, không phải độ sáng tăng đơn thuần. Ảnh nền sáng và chủ thể tối có thể được chỉnh đúng yêu cầu nhưng độ sáng toàn ảnh đi xa 128 hơn; không thể suy ra `false` đồng nghĩa ảnh tệ hơn về nhiếp ảnh.

Nguồn: [no_reference_eval.py:185](../../src/analyzer_evaluator/no_reference_eval.py#L185), [no_reference_eval.py:233](../../src/analyzer_evaluator/no_reference_eval.py#L233).

### B2. Chỉ số ngược sáng dùng trung bình đối tượng chưa phân biệt vật liệu

`backlit_measurement` lấy độ sáng trung bình `person` so với mốc 50 L* (`face` dùng 60), đồng thời yêu cầu nền sáng hơn ít nhất 20 L*. Các ngưỡng severity là độ thiếu so với mốc 12/22/34 L*. Như vậy vùng person chứa nhiều tóc đen/quần áo tối có thể làm mức thiếu sáng cao hơn dù mặt đã được cứu sáng; đây là nguy cơ cấu trúc, không phải kết luận rằng mọi ảnh như vậy sẽ lỗi.

Nguồn: [fusion.py:182](../../src/agent/fusion.py#L182), [fusion.py:443](../../src/agent/fusion.py#L443).

Session: before tracked severity=2, sau vòng 1=1, sau vòng 2=0; confidence=0.6. `resolved` nghĩa là vượt ngưỡng đo đã định nghĩa, không có nghĩa toàn bộ ảnh đã tự nhiên.

### B3. So sánh tác dụng phụ từng vòng và lỗi kiểm tra chưa có trạng thái unknown

`decide_node` gọi `_side_effects(previous_image, current_image)`, nên rào chặn ở node này so với vòng ngay trước. `_perceive_followup` có so ảnh hiện tại với ảnh gốc, nhưng chỉ chạy khi có vòng tiếp theo. Những thay đổi nhỏ tích lũy tới vòng cuối có thể không được chặn theo tổng tác động. Chưa có bằng chứng đây là tác nhân duy nhất của ca tóc xám; nên ghi là rủi ro kiến trúc.

Nguồn: [graph.py:492](../../src/agent/graph.py#L492), [perception.py:1060](../../src/agent/perception.py#L1060).

Nếu kiểm tra tác dụng phụ ném exception, `_side_effects` log cảnh báo rồi trả danh sách rỗng. Không có trạng thái phân biệt “không thấy lỗi” với “không kiểm tra được”.

Nguồn: [graph.py:342](../../src/agent/graph.py#L342).

## C. Rủi ro đầu ra khác, không quy là nguyên nhân của phiên này

1. **Metadata đánh giá sau rollback:** `decide_node` đổi `current_image` về ảnh trước khi suy thoái nhưng không thay `evaluation_result`. `record_process` xuất cả ảnh đã rollback và evaluation hiện có. Cần phân biệt đánh giá ứng viên bị loại với đánh giá ảnh thực sự giao cho người dùng. Nguồn: [graph.py:541](../../src/agent/graph.py#L541), [workspace_service.py:263](../../src/api/workspace_service.py#L263), [workspace_service.py:291](../../src/api/workspace_service.py#L291).
2. **Render lại không có gate đánh giá:** endpoint `/render` và workspace variant render chạy lại actions trên ảnh gốc rồi trả/lưu ảnh, không đánh giá hay review trước khi xuất. Nguồn: [routes.py:174](../../src/api/routes.py#L174), [workspace.py:657](../../src/api/workspace.py#L657). Trường hợp này không phải replay: JSON ghi ảnh final 800×533, không có variant.
3. **Preview/full-resolution có thể khác:** variant được tạo trên ảnh downscale; render lại thực thi actions, không lưu mask của preview trong recipe. Vì segmentation/feather và các thuật toán theo không gian có thể phụ thuộc độ phân giải, chưa thể đảm bảo ảnh xuất đã được duyệt chỉ từ preview. Đây là rủi ro cần test, chưa tái hiện trên ca này. Nguồn: [graph.py:589](../../src/agent/graph.py#L589), [variants.py:98](../../src/agent/variants.py#L98), [workspace.py:678](../../src/api/workspace.py#L678).

## D. Đề xuất từ kiểm toán

- P0: thêm review gate sau xử lý và trước mọi lần giao artifact, kể cả lần cuối; nhận ảnh gốc, ứng viên, crop mặt/tóc/vùng bảo toàn, mục tiêu người dùng và lịch sử. Đầu ra có `accept/revise/reject/unknown`, vấn đề theo vùng, evidence; reviewer không sinh pixel.
- P0: thay boolean review mặc định True bằng trạng thái `not_run/passed/failed/unavailable`, ghi reviewer/model/artifact hash và crop đã xem. `vlm_feedback` heuristic cần tên/source rõ ràng.
- P1: tách success của lỗi theo vùng khỏi preservation constraints. Cho phép sửa exposure mặt trong một lỗi person, nhưng phải liên kết child region với lỗi gốc để tracking không hiểu nhầm thao tác ngoài mục tiêu.
- P1: giữ ưu tiên đo lỗi theo vùng; không dùng `quality_improved=false` như một veto tuyệt đối. Kết hợp lỗi mục tiêu + hard artifacts + bảo toàn vùng + review tự nhiên.
- P1: kiểm tra cả original→candidate và previous→candidate ở vòng cuối, có ngân sách tác động cộng dồn; phân biệt unavailable với no side effects.
- P1: đánh giá lại ảnh giữ sau rollback và gắn evaluation đúng artifact.
- P2: dùng cùng gate cho variant/render/refine; critic được quyền từ chối tất cả. Không chỉ tăng num_variants để chữa thiếu review.
- P2: tập dữ liệu thật có tóc đen/tóc bạc, nhiều tông da, quần áo đen, low-key, ngược sáng và nền cháy có chủ ý; ghi cả mục tiêu sáng mặt và mức bảo toàn tông tối, đánh giá người dùng và latency/cost.

Các đề xuất thuật toán cần thử nghiệm trên nhiều ảnh; không khẳng định một ngưỡng L* hay một tỷ lệ tăng sáng tóc là chuẩn chung cho mọi người/chất liệu/ánh sáng.

## E. Kiểm chứng đã chạy

Lệnh offline (không gọi API thật):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_decide_defects.py tests/unit/test_agent_followup.py tests/unit/test_agent_variants.py -q
```

Kết quả: **50 passed, 1 warning in 18.29s**. Warning từ pyiqa về covariance trên ảnh nhỏ; không có assertion fail. Đây là xác nhận hành vi hiện tại đã mô tả, không phải bằng chứng chất lượng thị giác đã tốt hoặc bug tóc xám đã được sửa.

Không chạy benchmark người đánh giá, không gọi VLM mới, không tuyên bố đã kiểm chứng hiệu quả các đề xuất.
