# Nhiễu và độ nét

## Chọn bộ lọc khử nhiễu
tags: noise, denoise, gaussian, median, bilateral, nlm
sources: Tomasi & Manduchi, ICCV 1998; Buades, Coll & Morel, CVPR 2005; Gonzalez & Woods ch.5
Gaussian làm mờ đều, mất cạnh, chỉ hợp với nhiễu nhẹ. Median thay điểm ảnh bằng trung vị lân
cận, là lựa chọn đúng cho nhiễu muối tiêu (chấm trắng/đen rời rạc). Bilateral lọc theo cả
khoảng cách và chênh lệch màu nên giữ cạnh, hợp với nhiễu vừa. Non-Local Means (nlm) trung bình
các mảng giống nhau trên toàn vùng tìm kiếm; về lý thuyết giữ kết cấu tốt ở nhiễu mạnh, nhưng
với cài đặt của toolbox nó cần strength cao và chậm hơn nhiều (xem mục hiệu chỉnh bên dưới).

## Hiệu chỉnh cường độ khử nhiễu của toolbox
tags: noise, denoise, bilateral, nlm, strength, calibration, low light
sources: Đo nội bộ trên data/real/e2e/test_chelsea_dark_noisy.jpg so với skimage chelsea (2026-10)
Cùng một con số strength cho hiệu quả rất khác nhau giữa các phương pháp. Trên ảnh tối nhiễu
nặng (nâng gamma 1.3 sau khi khử), SSIM so với ảnh sạch: nlm 1.0 = 0.26, bilateral 1.0 = 0.37,
bilateral 1.5 = 0.48, bilateral 2.0 = 0.51, nlm 2.0 = 0.51. nlm mất khoảng 300 ms so với 1–3 ms
của bilateral trên ảnh 451×300. Với nhiễu nặng: bilateral 1.5–2.0 là lựa chọn mặc định.

## Đánh đổi giữa khử nhiễu và chi tiết
tags: noise, denoise, strength, sharpness, detail, skin
sources: Gonzalez & Woods, Digital Image Processing (4th ed.), ch.5
Mọi bộ lọc khử nhiễu đều làm mất một phần chi tiết tần số cao. Đánh giá No-Reference của hệ
thống coi việc mất trên 30% độ nét (phương sai Laplacian) là suy thoái và sẽ rollback, nên với
ảnh nhiễu nặng, khử vừa phải qua nhiều vòng an toàn hơn khử mạnh một lần.

## Hạt film là chủ ý
tags: film_grain, grain, noise, preserve, analog
sources: Kinh nghiệm nhiếp ảnh phổ biến
Hạt đều, mịn, đơn sắc trên toàn ảnh (đặc biệt ảnh đen trắng, tông phim) thường là chủ ý thẩm
mỹ. Nhiễu sensor thì khác: lốm đốm màu, tập trung ở vùng tối, đi kèm ảnh thiếu sáng.

## Làm nét không tạo ra chi tiết
tags: blur, sharpen, unsharp_mask, laplacian, halo, motion blur, defocus
sources: Gonzalez & Woods, Digital Image Processing (4th ed.), ch.3 và ch.5
Unsharp mask tăng tương phản tại cạnh nên ảnh trông nét hơn, nhưng không khôi phục chi tiết đã
mất. Mờ chuyển động và lệch nét cần giải chập (deconvolution), không có trong toolbox. Làm nét
quá tay sinh viền sáng (halo) và khuếch đại nhiễu, nên luôn làm nét sau khi khử nhiễu, và nhẹ tay.

## unsharp_mask và laplacian khác nhau ở tần số được làm nét
tags: sharpen, laplacian, unsharp_mask, noise, amount, calibration
sources: Đo nội bộ trên skimage astronaut làm mờ nhẹ + nhiễu σ=4 (2026-10)
unsharp_mask cộng lại chi tiết ở dải tần trung bình (Gaussian sigma 2), còn laplacian dùng kernel
3×3 nên đẩy mạnh chi tiết cấp điểm ảnh, và nhiễu nằm đúng ở dải đó. Cùng amount, laplacian mạnh
hơn nhiều: amount 0.5 làm nhiễu tăng ×4.1 (unsharp_mask ×1.5), amount 1.0 tăng ×7.1 (×1.9).
Với ảnh chụp, dùng unsharp_mask; laplacian chỉ hợp ảnh sạch nhiễu như tài liệu scan, với amount nhỏ.
