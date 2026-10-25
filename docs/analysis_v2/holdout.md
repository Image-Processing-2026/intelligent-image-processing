# Bộ hold-out — hướng dẫn duyệt nhãn

19 ảnh thật mới từ Wikimedia Commons, **không** nằm trong `data/real_benchmark_manifest.json`,
`data/calibration_sources.json`, và chưa từng được dùng để chỉnh prompt (agent rev 1/2) hay
calibration. Chỉ dùng để chấm kết quả cuối của `fusion`. Không chỉnh prompt, ngưỡng hay luật trên bộ
này.

- Manifest (file cần duyệt): **`data/holdout_manifest.json`**
- Ảnh: `python data/download_real_benchmark.py --holdout` → `data/real/holdout/<nhóm>/` (gitignored,
  cạnh dài 1600 px như bộ benchmark). Đã tải sẵn trong worktree tích hợp.
- Chấm: `python scripts/benchmark_analysis.py --set holdout --config fused --reviewed-only` (lead
  chạy sau khi `fusion` xong). Không có cờ thì script **từ chối** khi còn nhãn `unreviewed`;
  `--reviewed-only` bỏ hẳn ảnh chưa duyệt, `--allow-unreviewed` chấm cả chúng (chỉ để tham khảo).

## Trạng thái duyệt

17/19 ảnh đã được Tiên duyệt. `holdout_backlit_01` và `holdout_cast_01` vẫn `unreviewed`. Bảng dưới
là nhãn nháp ban đầu; nhãn đã duyệt nằm trong manifest.

## Nhãn nháp

Lead agent đặt nhãn từ thumbnail và tên category Commons; đây là gợi ý, không phải đáp án.
Mỗi ảnh có `label_status: "unreviewed"`. Khi duyệt một ảnh:

1. Mở ảnh ở kích thước đầy đủ (`data/real/holdout/...` hoặc `source_page`).
2. Sửa `expected.defects` (loại trong `DEFECT_TYPES`, `region`, `severity` 0–3, `min_severity`),
   `expected.preserve` (trong `PRESERVE_ASPECTS`), `scene_type`, `ambiguous` (ảnh mà người duyệt
   cũng không chắc thì đặt `true`: không được chấm).
3. Đặt `label_status: "reviewed"` và `reviewed_by: "<tên>"`.

Severity: 0 không đáng kể, 1 nhẹ (sửa thì đẹp hơn), 2 rõ, 3 nặng. `region` dùng `full`, `person`,
`face`, `sky`… như bộ benchmark (`full` khớp mọi vùng khi chấm).

## Danh sách và nhãn nháp

| id | nhóm | nhãn nháp (defect@vùng:severity / preserve) | cần chú ý khi duyệt |
|---|---|---|---|
| holdout_good_01 | already_good | – | |
| holdout_good_02 | already_good | – | ánh chiều ấm: tự nhiên hay ám? |
| holdout_good_03 | already_good | – | |
| holdout_style_01 | intentional_style | preserve low_key | |
| holdout_style_02 | intentional_style | preserve low_key, warm_tone | bokeh chủ ý (soft_focus?) |
| holdout_style_03 | intentional_style | preserve high_key | |
| holdout_style_04 | intentional_style | preserve silhouette | |
| holdout_style_05 | intentional_style | preserve warm_tone | thân cây tối: thêm silhouette? |
| holdout_style_06 | intentional_style | preserve film_grain | ảnh minh họa hạt phim B&W |
| holdout_backlit_01 | backlit | backlit_subject@person:1 | mặt quay đi; có thể là chủ ý |
| holdout_backlit_02 | backlit | backlit_subject@person:1 | đám đông ngược sáng; có thể ambiguous |
| holdout_noisy_01 | noisy | noise@full:2 | NIND ISOH3, đã thu về 1600 px |
| holdout_noisy_02 | noisy | noise@full:1 | NIND ISOH2, đã thu về 1600 px |
| holdout_blur_01 | blur | blur@full:3 | mất nét toàn khung |
| holdout_blur_02 | blur | blur@cats:1 | lấy nét sau mèo; xem ở 100 % |
| holdout_cast_01 | color_cast | color_cast_cool@full:2 | tuyết lúc chạng vạng có thể xanh tự nhiên |
| holdout_cast_02 | color_cast | color_cast_cool@full:2 | có thể là green thay vì cool |
| holdout_multi_01 | multi_defect | underexposed@full:2, noise@full:1 | |
| holdout_multi_02 | multi_defect | blur@full:1, noise@full:1 | đường phố đêm; có underexposed? |

Thiếu so với yêu cầu ban đầu: nhóm backlit chỉ có 2 ảnh và cả hai đều có thể bị coi là chủ ý; nên
thay bằng ảnh chân dung ngược sáng mặt tối rõ nếu người duyệt thấy cần.
