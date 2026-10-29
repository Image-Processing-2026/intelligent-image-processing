[English](interfaces.md) | [Tiếng Việt](interfaces.vi.md)

# Hợp đồng giữa module

Mốc `02dd6f9`. [Doc schema trước](interfaces-reference.md) được giữ như snapshot tham khảo; xem schema runtime cho đầy đủ trường hiện tại.

| Ranh giới | Bất biến | Source |
|---|---|---|
| Ảnh | Mặc định RGB uint8 `(H, W, 3)`; grayscale hỗ trợ giữ shape hợp lệ | [processing base](../src/processing_engine/base.py) |
| Soft mask | float32 [0, 1], khớp kích thước; `(H, W)` hoặc singleton channel hỗ trợ | [mask utilities](../src/region_engine/mask_utils.py) |
| Phép xử lý | `apply_*(image, mask=None, **params)` trả array; wrapper chung blend | [engine](../src/processing_engine) |
| Chẩn đoán/phác đồ/lịch sử | Pydantic validate giữa stage | [state](../src/agent/state.py) |
| Phép đo | Cấu trúc measurement và calibration | [schema](../src/analyzer_evaluator/measurement/schemas.py) |
| Workspace | Model request/response theo revision | [API schema](../src/api/workspace_schemas.py) |

Đổi BGR OpenCV về RGB tại ranh giới. Blend một lần: `M*processed + (1-M)*input`. Kiểm tra hữu hạn, miền và kích thước. Xem xử lý lỗi executor: request xong chưa chứng minh mọi thao tác chạy.

Toolbox cố định denoise, gamma_correct, clahe, sharpen, color_correct. Thay đổi phải đồng bộ bounds planner, dispatch, prompt/schema, knowledge card và doc. Guard giữ phong cách/chống chỉ định ràng buộc phác đồ. Xem [kiến trúc](architecture.vi.md).
