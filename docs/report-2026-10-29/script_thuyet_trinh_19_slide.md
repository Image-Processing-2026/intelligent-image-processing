# Kịch bản thuyết trình AI Image Doctor theo bốn module

Bản này dùng cho bài trình chiếu 19 slide sau khi bỏ slide phân công. Thời lượng dự kiến là 14 phút trình bày và 1 phút hỏi đáp. Các đoạn lời thoại có thể đọc trực tiếp; phần trong ngoặc vuông là chỉ dẫn khi tập và không đọc thành tiếng.

“Slide” trong kịch bản là vị trí trang hiện tại trong PowerPoint. Số ở góc slide vẫn giữ từ bản cũ: trang đầu là 01, các trang tiếp theo là 03 đến 20. Ví dụ, Slide 2 trong kịch bản tương ứng trang có số 03 ở góc.

## Phân chia phần trình bày

- Thành viên 1 — Module 1 Phân tích và đánh giá: Slide 1–4, từ 0:00 đến 3:30.
- Thành viên 2 — Module 2 Vùng và mặt nạ: Slide 5–8, từ 3:30 đến 7:00.
- Thành viên 3 — Module 3 Xử lý điểm ảnh cổ điển: Slide 9–13, từ 7:00 đến 10:30.
- Thành viên 4 — Module 4 Điều phối và ứng dụng: Slide 14–19, từ 10:30 đến 14:00.
- Cả nhóm trả lời câu hỏi: từ 14:00 đến 15:00.

Thời lượng ở từng slide là mốc tập, bao gồm thời gian chỉ hình và chuyển trang. Nếu một phần vượt giờ, ưu tiên giữ ý chính và câu bàn giao cuối module.

## Thành viên 1 Phân tích và đánh giá

### Slide 1 Giới thiệu đề tài

[Góc slide: 01. Thời lượng: 50 giây. Mốc: 0:00–0:50.]

Em xin kính chào thầy và các bạn. Nhóm em xin trình bày đề tài AI Image Doctor, hệ thống cải thiện chất lượng ảnh bằng cách kết hợp mô hình thị giác ngôn ngữ với các thuật toán xử lý ảnh cổ điển.

Trong thực tế, một ảnh có thể đồng thời bị nhiễu, thiếu sáng hoặc mất nét. Nếu áp một bộ lọc cố định lên toàn ảnh, vùng cần sửa có thể tốt hơn nhưng những vùng khác lại bị ảnh hưởng. Các phương pháp sinh ảnh còn có khả năng bổ sung chi tiết không có trong ảnh gốc.

Vì vậy, nhóm sử dụng VLM để nhận biết vấn đề và đề xuất cách xử lý, còn việc thay đổi điểm ảnh được thực hiện bằng các thuật toán cổ điển. Sau mỗi lượt, hệ thống đánh giá lại để quyết định tiếp tục hay dừng. Bài trình bày sẽ đi qua bốn module, từ đo chất lượng ảnh đến điều phối toàn bộ vòng lặp.

### Slide 2 Dữ liệu và chuẩn hóa ảnh đầu vào

[Góc slide: 03. Thời lượng: 50 giây. Mốc: 0:50–1:40. Chỉ lần lượt bốn ảnh minh họa.]

Trước tiên, để kiểm tra thuật toán, nhóm tạo ảnh lỗi từ ảnh sạch. Như trên hình, các lỗi minh họa gồm nhiễu Gaussian, mờ chuyển động, thiếu sáng và tương phản thấp. Bộ sinh dữ liệu chính có bảy dạng suy giảm; bốn ảnh này là các ví dụ để thấy rõ sự khác nhau giữa các nhóm lỗi.

Điểm thuận lợi của cách tạo dữ liệu này là chúng ta vẫn giữ được ảnh sạch ban đầu. Nhờ đó, kết quả sau xử lý có thể được so sánh với một mốc tham chiếu, thay vì chỉ nhìn bằng mắt và kết luận ảnh có vẻ đẹp hơn.

Với ảnh người dùng tải lên web, hệ thống kiểm tra file, xử lý hướng ảnh theo EXIF và chuyển về RGB. Ảnh sau chuẩn hóa là mảng NumPy kiểu uint8, có chiều cao, chiều rộng và ba kênh màu. Việc thống nhất định dạng giúp các module phía sau nhận cùng một cấu trúc dữ liệu và hạn chế lỗi do khác hệ màu hoặc kiểu dữ liệu.

### Slide 3 Phân tích chất lượng bằng số đo

[Góc slide: 04. Thời lượng: 50 giây. Mốc: 1:40–2:30. Chỉ luồng từ ảnh đến Measurement rồi Calibration.]

Sau khi chuẩn hóa, Module 1 phân tích ảnh bằng các phép đo kỹ thuật. Ảnh được xem xét ở mức toàn cục và theo các ô nhỏ để nhận ra những vấn đề không xuất hiện đồng đều trên toàn ảnh.

Chẳng hạn, độ sáng và tỷ lệ clipping cho biết ảnh có bao nhiêu điểm bị cắt sáng hoặc cắt tối. Nhiễu được ước lượng trên vùng ít cấu trúc để giảm nhầm lẫn giữa hạt nhiễu và chi tiết thật. Độ nét được xem xét qua đặc trưng biên và các phép đo như Laplacian. Ngoài ra còn có các thông tin về màu và tương phản.

Bước Calibration chuyển các phép đo thành mức độ lỗi từ không đến ba, giúp việc diễn giải thống nhất hơn. Đầu ra của module là bằng chứng kỹ thuật. Phần hiểu ngữ cảnh, ví dụ nền tối có phải chủ ý nghệ thuật hay không, sẽ được kết hợp với VLM ở Module 4.

### Slide 4 Đánh giá ảnh sau xử lý

[Góc slide: 05. Thời lượng: 60 giây. Mốc: 2:30–3:30. Chỉ công thức PSNR rồi ví dụ số. Giữ câu bàn giao ở cuối.]

Module 1 còn đánh giá ảnh sau xử lý. Công thức tiêu biểu ở đây là PSNR, bằng mười nhân logarit cơ số mười của hai trăm năm mươi lăm bình phương chia MSE, đơn vị decibel.

MSE là trung bình bình phương chênh lệch giữa ảnh đang đánh giá và ảnh sạch, trên mọi điểm ảnh và kênh màu. Hai trăm năm mươi lăm là mức cực đại của ảnh uint8. Khi sai số giảm, PSNR tăng. Ví dụ, MSE bằng một trăm cho PSNR khoảng hai mươi tám phẩy mười ba decibel.

Công thức này cần ảnh chuẩn phù hợp. Với ảnh thật không có ảnh sạch tương ứng, hệ thống dùng các phép đo không tham chiếu và review. Điểm nội bộ từ không đến một trăm là một thước đo khác, không phải PSNR. Module 1 cung cấp bằng chứng đánh giá, còn graph quyết định dừng.

Để sửa đúng chỗ, chúng ta cần xác định vùng tác động. Em xin chuyển sang thành viên phụ trách Module 2.

[Ghi nhớ khi hỏi đáp: PSNR = 10 log₁₀(255²/MSE), tương đương 20 log₁₀(255/√MSE) trong code. Với MSE gần 0, code trả 100 dB theo quy ước, trong khi công thức lý thuyết tiến tới vô hạn. Nguồn: src/analyzer_evaluator/reference_eval.py, các hàm _compute_mse và _compute_psnr.]

## Thành viên 2 Vùng và mặt nạ

### Slide 5 Yêu cầu vùng được chuyển thành mask

[Góc slide: 06. Thời lượng: 35 giây. Mốc: 3:30–4:05.]

Tiếp nối phần đánh giá, em xin trình bày Module 2, có nhiệm vụ xác định vùng cần xử lý.

Ví dụ, với yêu cầu làm sáng khuôn mặt nhưng giữ nguyên nền, hệ thống cần biết khuôn mặt nằm ở đâu. Region Engine nhận thông tin vùng hoặc đối tượng, chọn cách định vị phù hợp rồi tạo mask, tức mặt nạ xác định mức tác động ở từng điểm ảnh.

Mask có cùng kích thước không gian với ảnh, dùng kiểu float32 và giá trị từ không đến một. Vùng bằng không được giữ nguyên, vùng bằng một nhận toàn bộ tác động; các giá trị trung gian dùng để chuyển tiếp mềm.

### Slide 6 Định vị vùng theo mục tiêu và asset

[Góc slide: 07. Thời lượng: 60 giây. Mốc: 4:05–5:05. Chỉ ba nhánh của sơ đồ.]

Module 2 không dùng một cách định vị duy nhất cho mọi yêu cầu. Cách tạo vùng phụ thuộc vào loại đối tượng, thông tin đầu vào và các mô hình có sẵn trong môi trường chạy.

Với khuôn mặt, hệ thống có thể sử dụng MediaPipe. Khi có Face Landmarker, các mốc khuôn mặt hỗ trợ tạo vùng oval bám theo mặt. Với một đối tượng được mô tả bằng ngôn ngữ, nhánh GroundingDINO kết hợp SAM có thể định vị đối tượng bằng khung bao rồi tạo mặt nạ chi tiết.

GrabCut là một lựa chọn khởi tạo từ khung bao khi có điều kiện phù hợp. Các vùng hình học như vùng chữ nhật hoặc vùng theo vị trí có thể được tạo trực tiếp, không cần mô hình hiểu đối tượng.

Cần lưu ý, các mô hình này chỉ giúp xác định vị trí và phạm vi tác động. Chúng không sinh ra ảnh phục chế. Nếu thiếu asset hoặc không có thông tin vùng hợp lệ, hệ thống không thể mặc định rằng mọi nhánh đều hoạt động đầy đủ.

### Slide 7 Mask mềm giúp ghép vùng tự nhiên

[Góc slide: 08. Thời lượng: 60 giây. Mốc: 5:05–6:05. Chỉ ảnh đầu vào, mask, vùng phủ và ảnh kết quả.]

Hình này minh họa cách mask mềm giới hạn thay đổi vào một vùng. Từ trái sang phải là ảnh đầu vào, mặt nạ, vùng tác động được phủ màu và kết quả nâng sáng theo mask. Mask oval trong ví dụ được vẽ tay để giải thích cơ chế, không phải kết quả nhận diện tự động.

Nếu chỉ dùng mask có biên cứng, chỗ tiếp giáp giữa vùng đã sửa và vùng giữ nguyên có thể lộ rõ. Khi làm mềm biên, các điểm ở rìa nhận tác động giảm dần, nhờ đó kết quả chuyển tiếp tự nhiên hơn.

Công thức bên dưới là phép trộn hai ảnh theo mask. Ở vùng trắng, ảnh kết quả gần với ảnh đã xử lý. Ở vùng đen, kết quả giữ ảnh trước xử lý. Vùng xám là sự pha trộn giữa hai ảnh.

Về trách nhiệm, Module 2 tạo mask; Module 3 mới thực hiện thuật toán và áp dụng phép hòa trộn này. Việc thống nhất mask giúp nhiều công cụ khác nhau dùng chung một cơ chế xử lý cục bộ.

### Slide 8 Kiểm tra mask và phạm vi tác động

[Góc slide: 09. Thời lượng: 55 giây. Mốc: 6:05–7:00. Chỉ sự khác nhau giữa nâng sáng toàn ảnh và vùng mặt.]

Ba ảnh ở đây cho thấy vì sao phải kiểm soát phạm vi xử lý. Khi nâng sáng toàn ảnh, nền và trang phục cũng thay đổi. Nếu mục tiêu chỉ là làm rõ khuôn mặt, việc giới hạn bằng mask sẽ giữ được nhiều hơn những vùng người dùng không yêu cầu sửa.

Tuy nhiên, mask cũng có thể sai. Vì vậy, hệ thống cần kiểm tra tính hợp lệ của vùng trước khi áp thuật toán. Khi không định vị được vùng, thao tác tương ứng được bỏ qua. Với một số trường hợp mask mặt không đạt kiểm tra, code có thể tạm dùng khung bao với cường độ giảm còn một nửa; đây vẫn là phương án dự phòng cần tiếp tục cải thiện.

Sau khi có mask hợp lệ, hệ thống đã biết phần nào cần thay đổi và phần nào cần giữ. Em xin chuyển sang thành viên phụ trách Module 3 để trình bày các thuật toán tạo ra thay đổi đó.

## Thành viên 3 Xử lý điểm ảnh cổ điển

### Slide 9 Năm công cụ xử lý điểm ảnh

[Góc slide: 10. Thời lượng: 30 giây. Mốc: 7:00–7:30.]

Sau phần xác định vùng, em xin trình bày Module 3, là nơi trực tiếp thay đổi điểm ảnh.

Hệ thống có năm nhóm công cụ: khử nhiễu, hiệu chỉnh gamma, CLAHE, tăng nét và điều chỉnh màu. Mỗi công cụ nhận ảnh, mask và tham số; đầu ra giữ nguyên kích thước cùng kiểu dữ liệu uint8.

Planner quy định thứ tự xử lý, còn engine thực thi thuật toán. Một nguyên tắc là khử nhiễu trước khi tăng nét, vì tăng nét quá sớm có thể làm hạt nhiễu rõ hơn.

### Slide 10 Khử nhiễu và đánh đổi chi tiết

[Góc slide: 11. Thời lượng: 45 giây. Mốc: 7:30–8:15. Chỉ vùng mặt và chất liệu quần áo ở các ảnh.]

Hệ thống hỗ trợ các cách khử nhiễu như Gaussian, Median, Bilateral và Non Local Means. Gaussian làm trơn theo lân cận; Median hữu ích với nhiễu dạng xung; Bilateral kết hợp khoảng cách không gian và độ giống nhau về cường độ; còn Non Local Means khai thác sự tương đồng giữa các vùng ảnh.

Trong minh họa này, từ ảnh sạch nhóm thêm nhiễu Gaussian rồi thử các phương pháp xử lý. Ta có thể quan sát lượng hạt giảm đi, nhưng mức giữ chi tiết khác nhau. Đặc biệt, nếu khử nhiễu quá mạnh, da và chất liệu quần áo có thể bị bệt.

Vì vậy, mục tiêu không chỉ là làm ảnh thật mịn. Cần cân bằng giữa giảm nhiễu và giữ cấu trúc ban đầu. Agent giới hạn tham số strength trong khoảng cho phép, nhưng việc chọn mức phù hợp vẫn cần dựa vào mức nhiễu và đánh giá kết quả sau xử lý.

### Slide 11 Gamma và CLAHE cho sáng và tương phản

[Góc slide: 12. Thời lượng: 65 giây. Mốc: 8:15–9:20. Chỉ công thức Gamma, ví dụ số và hai cặp ảnh.]

Với ảnh thiếu sáng, công thức tiêu biểu là hiệu chỉnh Gamma. Mức đầu ra s bằng hai trăm năm mươi lăm nhân với r chia hai trăm năm mươi lăm, tất cả lũy thừa một chia gamma. Trong đó, r và s là mức điểm ảnh vào và ra, còn gamma là tham số dương điều khiển độ sáng.

Sau khi chia cho hai trăm năm mươi lăm, giá trị nằm trong khoảng từ không đến một. Với gamma lớn hơn một, số mũ nhỏ hơn một làm các mức trung gian tăng lên. Ví dụ, mức vào sáu mươi bốn và gamma hai phẩy năm cho kết quả khoảng một trăm bốn mươi sáu phẩy sáu chín. Bảng tra trong code cắt phần thập phân và trả mức một trăm bốn mươi sáu, nên ảnh sáng hơn.

Chương trình tính sẵn bảng tra hai trăm năm mươi sáu mức để áp nhanh lên từng kênh. Nếu dùng mask, kết quả này tiếp tục được hòa trộn theo vùng như Module 2 đã giải thích.

Với tương phản thấp, CLAHE xử lý kênh L trong LAB theo các vùng nhỏ và giới hạn mức khuếch đại. Cả hai công cụ chỉ làm rõ tín hiệu còn tồn tại, không khôi phục chi tiết đã mất do clipping.

[Ghi nhớ khi hỏi đáp: s = 255 × (r/255)^(1/γ), với γ > 0. γ = 1 giữ mức sáng, γ < 1 làm tối các mức trung gian. Giá trị 0 và 255 giữ nguyên. Đây là công thức trước bước hòa trộn mask. Nguồn: src/processing_engine/exposure_contrast.py, các hàm _raw_gamma và apply_gamma.]

### Slide 12 Tăng nét và điều chỉnh màu

[Góc slide: 13. Thời lượng: 40 giây. Mốc: 9:20–10:00.]

Với tăng nét, hệ thống có Unsharp Masking và Laplacian. Có thể hiểu Unsharp là lấy phần khác biệt giữa ảnh và bản làm mờ để tăng tương phản quanh biên. Ảnh sẽ trông rõ hơn, nhưng dùng quá mạnh có thể tạo viền sáng hoặc làm nhiễu tăng lên. Đây chưa phải phương pháp giải chập để khôi phục ảnh mờ nặng.

Với màu sắc, hệ thống điều chỉnh độ bão hòa trong HSV và thay đổi hệ số kênh đỏ, lam để điều khiển tông ấm lạnh. Cặp ảnh con mèo minh họa sự thay đổi tông ấm. Tham số này là điều khiển tương đối trong chương trình, không phải nhiệt độ màu đo bằng Kelvin.

### Slide 13 Kết quả benchmark thành phần

[Góc slide: 14. Thời lượng: 30 giây. Mốc: 10:00–10:30. Chỉ cột trước và sau, không đọc toàn bộ số.]

Biểu đồ cho thấy PSNR tăng trong các ví dụ benchmark thành phần, trong đó trường hợp gamma cải thiện rõ nhất. Tuy nhiên, đây là thử nghiệm nhỏ có ảnh chuẩn và quét tham số để chọn kết quả tốt nhất. Nó cho thấy khả năng của từng thuật toán, chưa đại diện cho hiệu quả tự động của toàn pipeline.

Như vậy, Module 3 đã cung cấp các công cụ xử lý. Thành viên phụ trách Module 4 sẽ giải thích hệ thống chọn công cụ, lặp lại và dừng như thế nào.

## Thành viên 4 Điều phối và ứng dụng

### Slide 14 LangGraph kết nối bốn module

[Góc slide: 15. Thời lượng: 60 giây. Mốc: 10:30–11:30. Chỉ theo chiều mũi tên; chỉ đường quay lại ở cuối.]

Sau khi đã có số đo, mask và các công cụ, Module 4 kết nối chúng thành một luồng xử lý bằng LangGraph. Thứ tự trình bày bốn module vừa rồi là cách phân chia trách nhiệm; thứ tự chạy thực tế được thể hiện ở sơ đồ này.

Đầu tiên, Analyze đo chất lượng ảnh. Perceive kết hợp VLM với bằng chứng kỹ thuật. Clarify làm rõ ý định người dùng, sau đó bước Diagnose and Plan tạo kế hoạch xử lý. Process gọi Module 2 để tạo mask và Module 3 để áp thuật toán. Evaluate dùng Module 1 để đánh giá ảnh ứng viên.

Từ kết quả đó, Decide quyết định xử lý tiếp hay dừng. Nếu cần xử lý tiếp, luồng quay về Analyze. Khi dừng, hệ thống qua Review rồi Finalize để xác định kết cục cuối.

Các bước chia sẻ DoctorState, chứa ảnh gốc, ảnh hiện tại, ảnh trước vòng và lịch sử. Nhờ vậy, vòng sau có thể dùng kết quả vòng trước và hệ thống có cơ sở để hoàn tác khi cần.

### Slide 15 Gemini đề xuất chương trình kiểm tra

[Góc slide: 16. Thời lượng: 40 giây. Mốc: 11:30–12:10.]

Gemini được dùng để diễn giải vấn đề của ảnh và hỗ trợ tạo kế hoạch. Thông tin gửi vào có thể gồm ảnh, crop, số đo và yêu cầu của người dùng, chẳng hạn làm sáng mặt nhưng giữ nguyên nền.

Kế hoạch trả về dưới dạng JSON có cấu trúc, mô tả công cụ, vùng và tham số. Chương trình kiểm tra kế hoạch, giới hạn tham số và giữ đúng thứ tự thực thi. Python mới là thành phần gọi các hàm xử lý ảnh.

Ở vòng sau, hệ thống ưu tiên đo lại và chỉ hỏi VLM thêm khi cần. Nếu thiếu khóa API hoặc gọi mô hình thất bại, có đường dùng luật và ghi nhận nguồn quyết định; không nên coi kết quả đó là suy luận thành công từ Gemini.

### Slide 16 Kiểm soát dừng và hoàn tác

[Góc slide: 17. Thời lượng: 35 giây. Mốc: 12:10–12:45.]

Vòng lặp cần được kiểm soát để tránh xử lý mãi hoặc làm ảnh xấu dần. Sau khi đánh giá ảnh ứng viên, hệ thống có thể hoàn tác về ảnh trước vòng, tiếp tục nếu còn lỗi và còn lượt, hoặc dừng.

Trong code, tham số giới hạn là max_iterations, mặc định bằng ba. Hết lượt chỉ có nghĩa là hệ thống phải dừng, chưa có nghĩa ảnh đã đạt mục tiêu.

Với ảnh thật, trạng thái ACCEPTED cần kết quả review hợp lệ gắn với đúng ảnh được trả ra. Nếu thiếu review hợp lệ, kết quả được ghi là UNVERIFIED. Cách ghi nhận này giúp tránh báo thành công khi chưa đủ bằng chứng.

### Slide 17 Ứng dụng và góp ý sau kết quả

[Góc slide: 18. Thời lượng: 30 giây. Mốc: 12:45–13:15.]

Ở phía ứng dụng, người dùng tải ảnh, xem chẩn đoán, xác nhận điều muốn sửa và điều cần giữ, rồi chạy pipeline. Kết quả và lịch sử được lưu theo phiên.

Sau đó, người dùng có thể góp ý, ví dụ ảnh vẫn còn tối. Nhánh refine điều chỉnh phác đồ và render lại từ ảnh gốc để tránh chồng bộ lọc. Đây là nhánh tinh chỉnh riêng, không tự động tương đương chạy lại toàn bộ vòng review vừa trình bày.

Case memory lưu kinh nghiệm cho các tình huống xử lý; cơ chế này không phải lưu ảnh để huấn luyện lại Gemini.

### Slide 18 Pilot toàn pipeline trên ảnh thật

[Góc slide: 19. Thời lượng: 30 giây. Mốc: 13:15–13:45.]

Với toàn pipeline, nhóm có thử nghiệm pilot gồm bốn ảnh, mỗi ảnh chạy ba lần trên hai phiên bản, tổng cộng hai mươi tư lượt. Biểu đồ thể hiện số lượt có chỉnh sửa, không phải số lượt đạt mục tiêu.

Hai phiên bản có hành vi khác nhau: V2 chỉnh ảnh nhiễu nhiều hơn, nhưng cả hai chưa nâng được trường hợp mặt tối. Do chỉ có bốn ảnh độc lập, kết quả này chưa đủ để kết luận phiên bản nào tốt hơn hoặc đổi cấu hình mặc định.

### Slide 19 Kết luận và hướng tiếp theo

[Góc slide: 20. Thời lượng: 15 giây. Mốc: 13:45–14:00.]

Qua bốn module, hệ thống có thể đo chất lượng, xác định vùng, xử lý và đánh giá lại. Hướng tiếp theo là mở rộng tập kiểm thử độc lập, cải thiện mask và kiểm soát tác dụng phụ. Nhóm em xin cảm ơn thầy và các bạn, và xin nhận câu hỏi.

## Chuẩn bị cho phần hỏi đáp

[Phần dưới dùng để ôn tập, không đọc nối tiếp vào bài. Thành viên 4 tiếp nhận câu hỏi và chuyển cho người phụ trách nếu cần.]

### Vì sao dùng VLM nhưng vẫn dùng thuật toán cổ điển

VLM hỗ trợ hiểu nội dung ảnh và yêu cầu người dùng. Thuật toán cổ điển trực tiếp thực hiện thay đổi điểm ảnh bằng các phép toán có thể mô tả và kiểm tra. Cách kết hợp này hạn chế việc sinh thêm chi tiết giả, nhưng vẫn có thể làm mất chi tiết, sai màu hoặc tạo halo nếu chọn công cụ và tham số chưa phù hợp.

### Module 1 và VLM có làm trùng nhiệm vụ không

Hai phần cung cấp bằng chứng khác nhau. Module 1 đo các đặc trưng kỹ thuật như sáng, nhiễu và nét. VLM diễn giải ý nghĩa ngữ cảnh và mục tiêu chỉnh sửa. Chẳng hạn, vùng tối có thể là lỗi thiếu sáng hoặc là phong cách chụp cần giữ. Việc kết hợp hai nguồn hỗ trợ quyết định có căn cứ hơn.

### Không có ảnh sạch thì đánh giá bằng cách nào

Không thể dùng PSNR và SSIM như đánh giá có ảnh tham chiếu nếu không có ground truth phù hợp. Với ảnh thật, hệ thống dựa vào các phép đo không cần tham chiếu, lỗi theo vùng, yêu cầu người dùng và review kết quả. Những bằng chứng này vẫn có giới hạn nên không thể khẳng định chất lượng tuyệt đối.

### Vì sao phải dùng mask mềm

Mask giúp giới hạn tác động vào vùng cần sửa. Giá trị trung gian ở biên làm mức xử lý chuyển tiếp dần, giảm đường nối rõ giữa vùng đã sửa và vùng giữ nguyên. Tuy nhiên, mask sai vị trí vẫn có thể tác động sai đối tượng; làm mềm biên không thay thế được việc định vị đúng.

### Hệ thống có phục hồi được ảnh mờ hoặc mất thông tin nặng không

Khả năng hiện tại chủ yếu là cải thiện chất lượng từ tín hiệu còn tồn tại. Tăng nét có thể làm biên rõ hơn nhưng không khôi phục đáng tin cậy toàn bộ chi tiết đã mất do mờ nặng. Tương tự, nâng sáng không tái tạo được vùng đã mất thông tin do clipping.

### Tại sao giới hạn vòng lặp bằng ba

Ba là giá trị mặc định trong code, nhằm giới hạn số lượt xử lý và chi phí. Đây không phải bằng chứng rằng ba vòng luôn tối ưu. Hệ thống có thể dừng sớm theo điều kiện đánh giá; khi hết lượt mà chưa đủ bằng chứng thì không được mặc định báo thành công.

### Nếu VLM trả JSON sai hoặc không gọi được API thì sao

Kế hoạch phải qua bước kiểm tra trước khi thực thi. Khi thiếu khóa hoặc gọi lỗi, hệ thống có đường xử lý theo luật và ghi nhận nguồn quyết định. Cần phân biệt kết quả dùng luật với một lần suy luận VLM thành công.

### Tại sao PSNR gamma tăng mạnh nhưng chưa thể nói toàn hệ thống tốt

Benchmark thành phần dùng ảnh chuẩn và chọn tham số tốt qua quét tham số. Nó kiểm tra khả năng của công cụ trong điều kiện có kiểm soát. Toàn pipeline còn phải tự nhận biết lỗi, chọn vùng và tham số, xử lý rồi đánh giá; mỗi bước đều có thể sai. Vì vậy, không dùng kết quả thành phần để thay thế đánh giá toàn hệ thống.

### Hai mươi tư lượt thử có phải hai mươi tư ảnh không

Không. Có bốn ảnh độc lập, mỗi ảnh chạy ba lần trên hai phiên bản. Các lần lặp giúp quan sát tính ổn định trên cùng ảnh, nhưng không làm tăng độ đa dạng dữ liệu thành hai mươi tư ảnh.

### Hệ thống có học lại từ góp ý người dùng không

Nhánh refine thay đổi phác đồ theo góp ý. Case memory lưu kinh nghiệm xử lý để tham khảo trong các tình huống sau. Đây không phải cập nhật trọng số hay huấn luyện lại Gemini.

## Cách tập và bàn giao

- Tập một lượt riêng từng module, sau đó tập cả nhóm với cùng một người bấm slide.
- Dùng vị trí trang hiện tại để chuyển slide; không nhầm với số cũ ở góc trang.
- Với ảnh và biểu đồ, chỉ vùng đang giải thích rồi dừng ngắn để người nghe quan sát; tránh đọc toàn bộ chữ trên slide.
- Nếu chậm giờ, rút bớt ví dụ và tên chỉ số phụ. Giữ phần giới hạn kết quả và câu bàn giao.
- Kết thúc phần nói ở phút 14. Khi hỏi đáp, Module 1 trả lời về dữ liệu và đánh giá, Module 2 về mask, Module 3 về thuật toán, Module 4 về VLM và điều phối.
