# Phơi sáng và tương phản

## Gamma tác động lên vùng trung tính
tags: gamma, exposure, midtones, underexposed, overexposed, gamma_correct
sources: Gonzalez & Woods, Digital Image Processing (4th ed.), ch.3
Biến đổi gamma ánh xạ I_out = 255·(I/255)^(1/γ), giữ nguyên điểm 0 và 255. Với γ > 1 vùng tối
và trung tính sáng lên nhiều nhất, vùng sáng gần như đứng yên, nên đây là cách làm sáng an toàn
hơn cộng độ sáng tuyến tính. Mỗi lần nâng gamma cũng nâng nhiễu ở vùng tối lên theo.

## Vùng cháy sáng không phục hồi được
tags: overexposed, highlights, clipping, highlight_clip_ratio, sky
sources: Gonzalez & Woods, Digital Image Processing (4th ed.), ch.3
Điểm ảnh đã chạm 255 mất hết thông tin kết cấu. Hạ gamma chỉ biến trắng thành xám nhạt, không
tạo lại mây hay chi tiết. Khi highlight_clip_ratio của vùng cao, mục tiêu hợp lý là làm vùng đó
bớt chói và hài hòa hơn, không phải khôi phục nó.

## Chênh sáng và chỉnh theo vùng
tags: backlit_subject, backlight, region, dodge, burn, face, sky, brightness_vs_rest
sources: Ansel Adams, The Negative (Zone System)
Khi chủ thể và nền chênh nhau nhiều (brightness_vs_rest rất âm hoặc rất dương), không mức phơi
sáng toàn ảnh nào đúng cho cả hai. Chỉnh theo vùng tương đương kỹ thuật dodge (làm sáng) và burn
(làm tối) trong phòng tối: nâng chủ thể tối, hạ vùng sáng, mỗi vùng một mức.

## Tương phản cục bộ bằng CLAHE
tags: clahe, contrast, low_contrast, histogram, local contrast, clip_limit
sources: Zuiderveld, Contrast Limited Adaptive Histogram Equalization, Graphics Gems IV (1994)
CLAHE cân bằng histogram trên từng ô 8×8 của kênh L (LAB) rồi nội suy giữa các ô. clip_limit cắt
đỉnh histogram trước khi cân bằng, giới hạn mức khuếch đại: 1.5–2.0 là nhẹ nhàng, trên 3.0 dễ
sinh quầng quanh cạnh và làm nổi nhiễu ở vùng phẳng như bầu trời hay tường.

## Ảnh tối hoặc sáng có chủ ý
tags: low_key, high_key, silhouette, preserve, intentional, mood
sources: Kinh nghiệm nhiếp ảnh phổ biến
Ảnh low-key (phần lớn là bóng tối, ánh sáng chọn lọc) và high-key (sáng, ít bóng) cố ý lệch khỏi
histogram cân bằng để tạo cảm xúc. Silhouette (bóng đen chủ thể trên nền sáng) cũng là chủ ý.
Dấu hiệu chủ ý: ánh sáng có hướng rõ ràng, chủ thể vẫn đọc được hình khối, không có nhiễu nặng.
"Sửa" các ảnh này về trung bình sẽ phá hỏng chúng.
