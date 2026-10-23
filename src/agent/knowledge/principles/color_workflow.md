# Màu sắc và quy trình

## Nhiệt độ màu và cân bằng trắng
tags: color, white balance, temperature, color_cast_warm, color_cast_cool, color_correct
sources: Gonzalez & Woods, Digital Image Processing (4th ed.), ch.6
Nguồn sáng có nhiệt độ màu khác nhau: đèn sợi đốt ~2700–3200K (vàng cam), nắng trưa ~5500K,
bóng râm và trời âm u ~6500–7500K (xanh lam). color_correct chỉ có trục đỏ–lam
(temperature_shift); nó không sửa được ám xanh lá–tím (tint) của đèn huỳnh quang.

## Tông ấm có chủ ý
tags: warm_tone, golden hour, sunset, food, candle, color_cast_warm, preserve
sources: Kinh nghiệm nhiếp ảnh phổ biến
Giờ vàng, hoàng hôn, ánh nến, đèn quán cà phê và ảnh đồ ăn mang tông ấm vì cảm xúc của cảnh.
Chỉ số color_cast của Module 1 không phân biệt được chủ ý với lỗi; phải nhìn ngữ cảnh. Khi
phân vân, khử một phần thay vì khử hoàn toàn.

## Bão hòa và màu da
tags: saturation, skin, portrait, undersaturated, oversaturated, saturation_scale
sources: Gonzalez & Woods, Digital Image Processing (4th ed.), ch.6
Tăng saturation nhân kênh S của HSV cho mọi màu như nhau, nên da người (vốn đã bão hòa vừa
phải) là thứ đầu tiên trông bất thường. Với ảnh có người, giữ saturation_scale trong khoảng
0.95–1.15; với phong cảnh có thể tới 1.2.

## Thứ tự xử lý
tags: workflow, order, denoise, gamma_correct, clahe, sharpen, color_correct
sources: Gonzalez & Woods, Digital Image Processing (4th ed.), ch.3 và ch.5
Thứ tự an toàn: khử nhiễu → phơi sáng (gamma) → tương phản (CLAHE) → làm nét → màu. Mỗi bước
sau khuếch đại những gì bước trước để lại, nên nhiễu phải được xử lý đầu tiên. Planner luôn sắp
xếp lại theo thứ tự này.

## Ít mà đủ qua nhiều vòng
tags: workflow, iteration, subtle, over-processing, severity
sources: Kinh nghiệm hậu kỳ phổ biến
Hệ thống chạy tối đa vài vòng và đo lại sau mỗi vòng. Chỉnh vừa phải rồi để vòng sau xác nhận
tốt hơn chỉnh mạnh một lần: lỗi do xử lý quá tay (halo, da bệt, cháy) khó gỡ hơn lỗi còn sót.

## Mặt nạ mềm và độ rộng viền
tags: region, mask, feather_radius, face, sky, heuristic, backend
sources: ADR-003 của dự án (soft masks)
Mọi vùng được blend bằng mặt nạ mềm. feather_radius 15 hợp với vùng lớn; vùng có cạnh tự nhiên
như khuôn mặt cần 25–35 để không lộ viền. Mặt nạ 'sky' hoặc 'ground' khi không có backend ngữ
nghĩa chỉ là nửa trên/dưới khung hình (backend 'heuristic'), nên chỉnh vùng đó phải nhẹ tay.
