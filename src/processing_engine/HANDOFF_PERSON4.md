# Bàn giao Module 3 → Module 4: tích hợp `white_balance` và các thay đổi hợp đồng

**Từ:** Person 3 (Processing Engine) · **Gửi:** Person 4 (Agent/Backend)
**Nhánh:** `feat/module3-processing-engine`

> ⚠️ **Trạng thái:** `apply_white_balance` đã có trong Module 3, có test, nhưng **Agent chưa gọi được**.
> Nó chỉ chạy trong pipeline sau khi Person 4 hoàn tất các bước ở Mục 2. Module 3 không sửa
> `planner.py`, `executor.py`, `state.py`, `vlm_diagnostician.py` hay playbook.

---

## 1. Vì sao cần tool mới

| Defect (đã có trong `state.py`) | Hiện tại | Sau khi tích hợp |
|---|---|---|
| `color_cast_warm` / `color_cast_cool` | `color_correct.temperature_shift`: chỉ dịch trục đỏ–lam, lượng dịch do VLM đoán | `white_balance` ước lượng nguồn sáng từ ảnh và tự tính gain, không cần đoán lượng |
| `color_cast_green` | Card `green-cast-limits`: "ngoài khả năng toolbox", `recipe: []` | `white_balance` sửa được, vì gain từng kênh bao gồm cả trục xanh lá–tím (tint) |

`color_correct` vẫn giữ nguyên để **chỉnh tông có chủ ý** (làm ấm, tăng bão hòa).
`white_balance` dùng để **khử ám màu do nguồn sáng**.

---

## 2. Các bước tích hợp (đều nằm trong phần của Person 4)

### 2.1 `src/agent/planner.py`

```python
ALLOWED_OPERATIONS = {..., "white_balance"}

PARAMETER_BOUNDS["white_balance"] = {
    "method": {"allowed": ["gray_world", "white_patch"], "default": "gray_world"},
    "percentile": {"min": 90.0, "max": 100.0, "default": 99.0},
}

# Thứ tự đề xuất: sau denoise (nhiễu làm lệch thống kê/phân vị), trước gamma/CLAHE
# (cân bằng màu trên dữ liệu gần tuyến tính nhất), và trước color_correct (chỉnh tông
# có chủ ý phải áp lên ảnh đã trung tính).
priority_map["white_balance"] = 15
```

`white_balance` không có giá trị "trung hòa" (nó luôn tự tính gain), nên **không** cần thêm
vào `_NEUTRAL_PARAMETERS`.

### 2.2 `src/agent/executor.py`, trong chuỗi `if/elif op == ...`

```python
from src.processing_engine.color import apply_white_balance

elif op == "white_balance":
    current_img = apply_white_balance(
        current_img,
        mask=mask,
        method=params.get("method", "gray_world"),
        percentile=float(params.get("percentile", 99.0)),
    )
```

### 2.3 `src/agent/state.py` (dòng ~154) và `SYSTEM_PROMPT` trong `vlm_diagnostician.py`

Thêm vào danh sách toolbox:

```
- white_balance (parameters: method in ['gray_world', 'white_patch'], percentile in [90, 100]).
  Dùng khi ảnh bị ám màu do nguồn sáng (vàng, lam, xanh lá). KHÔNG dùng khi tông màu là chủ ý
  (hoàng hôn, đèn nến, blue hour).
```

Nếu VLM dùng structured output với schema tham số theo từng operation, cần thêm schema
`{method: enum, percentile: number}` cho `white_balance`.

### 2.4 Playbook `src/agent/knowledge/cards/color.yaml` (gợi ý)

```yaml
- id: green-cast-correction           # thay cho green-cast-limits
  defects:
    - {types: [color_cast_green], region: any, min_severity: 1}
  recipe:
    - operation: white_balance
      region: $defect
      params: {method: gray_world}
  avoid:
    - "Cảnh gần như một màu (rừng, cỏ, biển): Gray World sẽ khử luôn màu chủ đạo của cảnh — dùng white_patch nếu có vật trắng, hoặc bỏ qua."
```

---

## 3. Ví dụ action

```json
{
  "region_id": "face",
  "target_prompt": "face",
  "region_type": "face",
  "detected_issue": "color_cast_green",
  "operation": "white_balance",
  "parameters": {"method": "gray_world"},
  "order": 15
}
```

```json
{
  "region_id": "full",
  "target_prompt": "full",
  "region_type": "full",
  "detected_issue": "color_cast_warm",
  "operation": "white_balance",
  "parameters": {"method": "white_patch", "percentile": 99.0},
  "order": 15
}
```

**Mask được dùng làm trọng số ước lượng:** với `region_type: "face"`, chỉ điểm ảnh trên khuôn
mặt quyết định gain. Ví dụ khuôn mặt dưới đèn vàng sẽ được khử ám vàng dựa trên chính ánh sáng
chiếu lên mặt, kể cả khi phần nền là ánh sáng ban ngày.

---

## 4. Chọn method

| Method | Giả thuyết | Công thức gain | Khi nào dùng |
|---|---|---|---|
| `gray_world` | Trung bình cảnh là xám | `g_c = mean(μ_R, μ_G, μ_B) / μ_c` | Cảnh nhiều màu, không có vật trắng rõ |
| `white_patch` | Vùng sáng nhất là vật trắng | `g_c = max(p_R, p_G, p_B) / p_c`, với `p_c` là phân vị `percentile` | Có vật trắng/xám sáng (áo, tường, giấy) |

Cả hai đều **giữ nguyên phơi sáng chung** (không đẩy ảnh sáng lên). Điểm ảnh cháy sáng
(kênh ≥ 250) bị loại khỏi phần ước lượng, và gain mỗi kênh bị giới hạn trong `[1/4, 4]`.

---

## 5. Thay đổi hành vi của các tool hiện có (cần Person 4 nắm)

1. **Lỗi đầu vào giờ raise `ProcessingInputError`** (lớp con của `ValueError`) thay vì âm thầm
   trả về ảnh gốc. Các trường hợp: `method` lạ, gamma/strength ≤ 0, NaN/inf, ảnh RGBA/float,
   mask sai shape/dtype. `execute_plan` đang bọc `try/except Exception` cho từng action nên
   pipeline không sập, chỉ bỏ qua action đó và ghi log `warning`. Hệ quả: action có
   `method` sai trước đây bị coi là "đã chạy" (thực chất không làm gì), giờ được log rõ.
2. **`temperature_shift` chuyển sang gain kênh:** `gain_R = 1 + k·t`, `gain_B = 1 − k·t` với
   `k = 20/128`. Tại xám 128, lượng dịch bằng đúng cách cộng ±20 trước đây, nên **các giá trị
   trong playbook (−0.2 … −0.5) giữ nguyên ý nghĩa**. Khác biệt: vùng tối không còn bị ám,
   còn vùng sáng dịch mạnh hơn một chút. Đây là chỉnh tông ấm/lạnh, không phải nhiệt độ màu Kelvin.
3. **Ảnh `(H, W, 1)`** giờ được trả về đúng shape `(H, W, 1)`.
4. `apply_clahe` nhận `tile_grid_size` dạng list (từ JSON). Executor hiện chưa truyền tham số này.
5. **`denoise.strength` có nghĩa vật lý:** σ_target = 12.5 · strength (σ nhiễu Gauss, thang 0–255),
   tức strength 2.0 tương ứng σ 25. Các hệ số được hiệu chỉnh bằng benchmark (`docs/module3/benchmark/`).
   **bilateral và nlm cho kết quả giống hệt từng byte so với trước**, nên số liệu trong playbook vẫn
   đúng. gaussian yếu đi khoảng một nửa ở cùng strength (trước đây quá mạnh gấp 2 lần), median đổi
   ngưỡng kernel (3×3 khi σ < 17.5, 5×5 khi σ < 35). Gợi ý cho prompt VLM: "strength ≈ σ nhiễu
   ước lượng / 12.5". Nếu Module 1 có ước lượng σ, có thể tính strength trực tiếp thay vì để VLM đoán.

---

## 6. Kiểm tra sau khi tích hợp

```bash
pytest tests/unit/test_white_balance.py tests/unit/test_processing_engine.py -q
```

Gợi ý thêm cho Person 4: một test ở `test_agent_planner.py` kiểm tra `white_balance` được giữ
lại và sắp ở order 15, cùng một test ở `test_agent_executor.py` kiểm tra dispatch đến
`apply_white_balance`.
