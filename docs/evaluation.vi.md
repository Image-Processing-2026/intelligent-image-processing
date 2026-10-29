[English](evaluation.md) | [Tiếng Việt](evaluation.vi.md)

# Đánh giá và tái lập

Mốc doc: `02dd6f9`, ngày 2026-10-30. Báo cáo hiện có ứng với snapshot thực nghiệm riêng; không trình bày kết quả đó như xác nhận mới cho revision này.

## Ba loại bằng chứng

| Bằng chứng | Câu hỏi trả lời | Giới hạn |
|---|---|---|
| Unit/integration test | Contract, guard, route và thực thi có đúng assertion? | Test đạt không chứng minh chất lượng cảm nhận |
| Benchmark tổng hợp có tham chiếu | Phục hồi gần ảnh sạch có sẵn đến đâu? | Suy giảm nhân tạo có thể khác ảnh thật |
| Đánh giá ảnh thật | Kết quả đáp ứng yêu cầu mà không gây tác dụng phụ? | Không có ảnh sạch đích; sở thích và biến động VLM ảnh hưởng |

Đánh giá có tham chiếu dùng PSNR/SSIM với ảnh sạch căn chỉnh. Cao hơn thường là gần hơn, chưa chắc đẹp hơn. Đánh giá không tham chiếu tùy chọn pyiqa BRISQUE/NIQE khi có model, nếu không dùng điểm fallback nội bộ. Ghi backend thực tế; không hoán đổi điểm giữa backend. Không dùng ảnh suy giảm gốc hoặc baseline lọc cố định như ground truth sạch.

## Thực nghiệm hiện có

[Benchmark Module 3](module3/benchmark/README.md) dùng bốn ảnh scikit-image (astronaut, coffee, chelsea, rocket), cạnh dài 384 pixel, seed 2026 kết hợp CRC32 theo ca, suy giảm tổng hợp biết trước, PSNR/SSIM và CIEDE2000 trung bình. Chạy từ repo root:

```powershell
python scripts/benchmark_module3.py --out-dir scratch/doc-benchmark
```

[Báo cáo exposure đã lưu](module3/benchmark/exposure.md) ghi PSNR trung bình 14.61 → 44.69 dB và SSIM 0.6192 → 0.9767 cho ca làm tối exponent 2 được sửa bằng gamma 2.0. Đây là kết quả quét tham số lịch sử với suy giảm biết trước, không phải tỷ lệ thành công của agent. Không chạy lại benchmark khi viết doc. Báo cáo thiếu bản ghi đầy đủ phần cứng/revision nguồn, nên timing và so sánh khác revision chưa được kiểm chứng.

[Báo cáo A/B prompt ảnh thật](reports/2026-10-29-prompt-ab-basic.md) mô tả 17 ảnh, hai phiên bản prompt, một lượt mỗi ảnh/phiên bản. Có đối chứng và ca giới hạn. Chưa chứng minh ổn định giữa lần gọi. Ảnh lọc cố định không phải ground truth; prompt, metadata và hash snapshot thuộc protocol đó. Đọc báo cáo trước khi diễn giải rating hoặc latency.

## Protocol thực nghiệm mới

Ghi commit nguồn, phiên bản môi trường, model/backend, revision prompt, manifest/license dataset, split, số mẫu, seed, tiền xử lý, lệnh, phần cứng và vị trí output thô. Tách calibration/training khỏi holdout. So sánh cùng kích thước/ràng buộc; phân biệt cold start, nạp model và latency request.

Chấm đạt mục tiêu và tác dụng phụ riêng. Có ảnh tốt sẵn, low-key/phong cách có chủ ý, tối kèm nhiễu, ngược sáng cục bộ và cháy sáng. Báo ca thất bại/không đổi. Với chân dung, xem rò mask, texture da, halo, lệch màu và giữ nền ở cùng zoom. Không suy phần trăm độ chính xác tổng thể từ gallery nhỏ được chọn.

## Hình trong doc

```powershell
python scripts/generate_doc_figures.py
```

Ví dụ offline xác định tạo `docs/assets/gamma-mask-example.png` từ gradient sinh bằng code, không có ảnh tải mạng hoặc upload riêng tư. Chỉ minh họa gamma/blending, không kiểm chứng chẩn đoán, semantic hay chất lượng phục hồi tổng thể.

Xem [bản ghi kiểm chứng doc](documentation-verification.vi.md) cho kiểm tra đã chạy và giới hạn còn lại.
