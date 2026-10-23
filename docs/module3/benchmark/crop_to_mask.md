# Module 3 benchmark — crop_to_mask

Seed 2026, ảnh `astronaut` phóng to + nhiễu σ = 10, mask create_soft_mask (feather 15). Thời gian nhỏ nhất trong 3 lần (NLM 12MP: 1 lần). Mọi dòng đã kiểm tra kết quả giống hệt từng byte.

| độ phân giải | mask > 0 | thuật toán | toàn ảnh (ms) | cắt theo mask (ms) | tăng tốc |
|---|---|---|---|---|---|
| 1920x1080 | 5.5% | denoise nlm s=0.8 | 2748.3 | 1119.4 | ×2.5 |
| 1920x1080 | 5.5% | denoise gaussian s=2.0 | 160.4 | 17.1 | ×9.4 |
| 1920x1080 | 5.5% | denoise median s=2.0 | 172.3 | 17.7 | ×9.8 |
| 1920x1080 | 5.5% | sharpen unsharp a=1.0 | 166.1 | 16.3 | ×10.2 |
| 1920x1080 | 5.5% | gamma 2.0 | 158.4 | 14.8 | ×10.7 |
| 1920x1080 | 5.5% | temperature_shift -0.3 | 215.1 | 22.4 | ×9.6 |
| 1920x1080 | 26.1% | denoise nlm s=0.8 | 2730.5 | 1087.0 | ×2.5 |
| 1920x1080 | 26.1% | denoise gaussian s=2.0 | 152.5 | 63.9 | ×2.4 |
| 1920x1080 | 26.1% | denoise median s=2.0 | 197.7 | 64.7 | ×3.0 |
| 1920x1080 | 26.1% | sharpen unsharp a=1.0 | 186.2 | 56.6 | ×3.3 |
| 1920x1080 | 26.1% | gamma 2.0 | 174.5 | 62.8 | ×2.8 |
| 1920x1080 | 26.1% | temperature_shift -0.3 | 250.3 | 73.2 | ×3.4 |
| 4000x3000 | 5.2% | denoise nlm s=0.8 | 14860.6 | 1296.2 | ×11.5 |
| 4000x3000 | 5.2% | denoise gaussian s=2.0 | 1083.0 | 90.4 | ×12.0 |
| 4000x3000 | 5.2% | denoise median s=2.0 | 1160.2 | 107.5 | ×10.8 |
| 4000x3000 | 5.2% | sharpen unsharp a=1.0 | 1141.2 | 111.8 | ×10.2 |
| 4000x3000 | 5.2% | gamma 2.0 | 1113.1 | 94.5 | ×11.8 |
| 4000x3000 | 5.2% | temperature_shift -0.3 | 1412.4 | 118.2 | ×11.9 |
| 4000x3000 | 25.4% | denoise nlm s=0.8 | 14520.4 | 5248.0 | ×2.8 |
| 4000x3000 | 25.4% | denoise gaussian s=2.0 | 1107.2 | 344.0 | ×3.2 |
| 4000x3000 | 25.4% | denoise median s=2.0 | 1144.0 | 380.0 | ×3.0 |
| 4000x3000 | 25.4% | sharpen unsharp a=1.0 | 1147.1 | 367.8 | ×3.1 |
| 4000x3000 | 25.4% | gamma 2.0 | 923.6 | 269.1 | ×3.4 |
| 4000x3000 | 25.4% | temperature_shift -0.3 | 1137.4 | 358.2 | ×3.2 |
