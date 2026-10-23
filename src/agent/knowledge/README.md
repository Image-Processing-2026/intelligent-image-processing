# Knowledge Base của Module 4

Tri thức chuyên môn giúp agent lập kế hoạch xử lý ảnh. Hệ thống dùng nó ở hai chỗ:

1. **Prompt giai đoạn Plan.** `retrieve()` chọn các card và nguyên lý phù hợp với chẩn đoán. `format_context()` đưa chúng vào prompt, với tham số đã chọn sẵn theo mức độ lỗi của chính ảnh đang xử lý.
2. **Rule engine offline**, dùng khi không có API key hoặc Gemini lỗi. `plan_actions_from_knowledge()` chạy công thức của các card có `auto_apply: true`.

KB không dùng embedding. KB chỉ có vài chục mục, và truy vấn chủ yếu là định danh có cấu trúc (loại lỗi, vùng, loại cảnh), nên hệ thống lọc theo cấu trúc rồi xếp hạng bằng BM25. Cách này chạy offline hoàn toàn và không tốn thêm lời gọi API.

## Tầng A: playbook card (`cards/*.yaml`)

```yaml
- id: dark-region                 # chữ thường, số, '-'; không trùng
  title: Vùng/chủ thể tối do chênh sáng
  priority: 60                    # cao hơn thắng khi hai card cùng (operation, vùng)
  auto_apply: true                # rule engine offline được tự áp dụng
  scenes: []                      # rỗng = mọi cảnh; có giá trị = chỉ các cảnh này
  match: any                      # any | all (mọi điều kiện defects phải khớp)
  defects:                        # điều kiện kích hoạt
    - {types: [backlit_subject, underexposed], region: region, min_severity: 2}
  unless:                         # có lỗi khớp điều kiện này thì card không áp dụng
    - {types: [underexposed], region: full, min_severity: 2}
  blocked_by: [silhouette, low_key]   # preserve trên toàn ảnh hoặc trên vùng lỗi chặn card
  metrics: {noise_level: {in: [clean, low]}}  # tùy chọn: điều kiện trên chỉ số Module 1
  recipe:
    - operation: gamma_correct    # chỉ 5 công cụ của toolbox
      region: $defect             # full | $defect (vùng của lỗi đã khớp) | tên vùng
      params: {gamma: {2: 1.4, 3: 1.6}}   # cố định hoặc theo severity
      feather_radius: 30
  avoid: ["..."]                  # chống chỉ định, đưa nguyên văn vào prompt
  rationale: "..."
  tags: [...]                     # từ khóa cho BM25 (nên có cả tiếng Anh và tiếng Việt)
  sources: ["..."]
```

- `region` trong điều kiện có thể là `full`, `region` (mọi vùng khác full), `any`, hoặc một tên vùng cụ thể.
- Với tham số theo severity, mức không có khóa dùng khóa thấp hơn gần nhất. Nếu mức thấp hơn mọi khóa thì dùng khóa nhỏ nhất.
- Card nạp lỗi nếu dùng operation ngoài toolbox, tham số sai tên hoặc vượt `PARAMETER_BOUNDS`, hay dùng loại lỗi, loại cảnh hoặc preserve không có trong từ vựng (`state.py`). Thông báo lỗi chỉ rõ file và id card.

**Khi nào được đặt `auto_apply: true`:** chỉ khi đã đo trên ảnh thật rằng công thức không làm giảm chất lượng mà không cần nhìn ngữ cảnh. Đo đạc cho thấy làm nét và chỉnh màu tự động làm giảm chất lượng (đồ ăn và hoàng hôn ấm là chủ ý), nên card của hai nhóm này chỉ tư vấn cho VLM. Test `test_auto_apply_cards_avoid_measured_regressions` giữ ràng buộc này.

## Tầng B: nguyên lý (`principles/*.md`)

Mỗi mục `## Tiêu đề` là một đoạn nguyên lý. Dòng `tags:` (phân cách bằng dấu phẩy) và dòng `sources:` (phân cách bằng dấu chấm phẩy) là siêu dữ liệu. id của đoạn là `<tên file>/<slug của tiêu đề>`.

Ưu tiên tri thức **đã đo trên chính toolbox này**, ví dụ mục "Hiệu chỉnh cường độ khử nhiễu". Tham số sách vở thường không khớp với cách Module 3 cài đặt.

## Kiểm tra sau khi sửa KB

```bash
pytest tests/unit/test_agent_knowledge.py -v
```
