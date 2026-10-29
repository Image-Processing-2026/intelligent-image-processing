# AI Image Doctor — web app

Giao diện phiên hiện chia ba phase **Phân tích → Xử lý → Kết quả** (xanh dương / tím / teal).
Flow log bên cạnh cho xem đầu vào, lý do và đầu ra đã ghi nhận; chọn vòng trong phase Xử lý để
đối chiếu đúng ảnh và phác đồ. Log dùng dữ liệu đã lưu sau khi request hoàn tất, chưa có tiến độ trực tiếp.
Chi tiết bàn giao và kiểm tra: [Studio theo phase](../docs/frontend/studio-2026-10-27.md).

Giao diện web (React + Vite + TypeScript + Tailwind + shadcn/ui) để thử và đánh giá pipeline. App gọi API
mới `/api/v1/auth/*` và `/api/v1/workspace/*` của FastAPI (`src/api/`). Gradio (`frontend/app.py`) vẫn giữ
nguyên. Thiết kế và các quyết định nằm trong `docs/frontend/` (context, decisions, spec, plan).

Yêu cầu: Node 20+ (đã kiểm với Node 23.6), Python 3.10+ với `requirements.txt` của repo.

## Chạy khi phát triển (2 terminal, từ repo root)

```powershell
uvicorn src.api.main:app --reload --port 8000
```

```powershell
npm --prefix web ci
npm --prefix web run dev
```

Mở http://localhost:5173. Vite chuyển `/api` sang `IID_BACKEND_URL` (mặc định `http://127.0.0.1:8000`)
và không đặt timeout, vì một lần chạy pipeline là một request đồng bộ.

## Chạy production-local (1 server)

```powershell
npm --prefix web ci
npm --prefix web run build
uvicorn src.api.main:app --port 8000
```

Mở http://localhost:8000. FastAPI phục vụ `web/dist` (đổi bằng `WEB_DIST_DIR`), deep link như `/s/<id>`
trả về `index.html`, còn `/api/...` không tồn tại vẫn trả 404 JSON. Không có `web/dist` thì server vẫn
chạy, chỉ không có giao diện web.

Muốn chạy không có Gemini (rule-based), để trống `GEMINI_API_KEY` trong môi trường của lệnh uvicorn.
Khóa chỉ đọc ở server, không bao giờ gửi xuống trình duyệt. Có khóa không có nghĩa là Gemini đã chạy:
nguồn thật của từng chẩn đoán/phác đồ hiện trên từng kết quả.

## Biến môi trường (backend)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GEMINI_API_KEY` | trống | Có → gọi Gemini; trống → rule-based |
| `APP_DATA_DIR` | `data/app` | SQLite tài khoản/phiên + ảnh của từng phiên |
| `CASE_MEMORY_PATH` | `data/memory/cases.sqlite` | Bộ nhớ ca bệnh; chuỗi rỗng → tắt |
| `PIPELINE_CONCURRENCY` | `1` (1–4) | Số pipeline chạy cùng lúc; request khác chờ |
| `MAX_UPLOAD_MB` | `20` | Dung lượng ảnh tối đa |
| `MAX_IMAGE_PIXELS` | `40000000` | Số điểm ảnh tối đa |
| `MIN_IMAGE_SIDE` | `16` | Cạnh ngắn tối thiểu |
| `AUTH_TOKEN_TTL_DAYS` | `30` | Hạn token đăng nhập |
| `COOKIE_SECURE` | `false` | Bật khi chạy sau HTTPS |
| `ALLOW_REGISTRATION` | `true` | Tắt để chặn đăng ký mới |
| `CORS_ORIGINS` | trống | Chỉ đặt khi frontend chạy ở origin khác; không dùng `*` |
| `WEB_DIST_DIR` | `web/dist` | Thư mục bản build |

Phía frontend chỉ có `IID_BACKEND_URL` (proxy của `npm run dev`).

## Dữ liệu nằm ở đâu

- `data/app/app.sqlite`: người dùng (mật khẩu băm argon2), token đăng nhập (chỉ lưu hash), phiên xử lý,
  hội thoại, lần chạy, phiên bản, phác đồ, đánh giá.
- `data/app/users/<uid>/sessions/<sid>/`: ảnh gốc, thumbnail, kết quả từng vòng, ảnh xem trước các phiên
  bản, ảnh render/tinh chỉnh. Xóa phiên trong app thì thư mục này bị xóa.
- `data/memory/cases.sqlite`: bộ nhớ ca bệnh (số liệu, lựa chọn, góp ý; không lưu ảnh), tách theo người dùng.
- Ảnh mẫu đọc từ `data/real/` (`coffee_underexposed.jpg`, `astro_noisy.jpg`, `cat_warm_cast.jpg`); các file
  này không nằm trong git, thiếu file nào thì mẫu đó không hiện.

Cả `data/app/` và `web/dist/` đều đã được gitignore. Muốn làm lại từ đầu thì dừng server rồi xóa `data/app/`.

## Kiểm tra

```powershell
npm --prefix web run typecheck
npm --prefix web test -- --run
npm --prefix web run build
```

Test backend của API mới: `pytest tests/unit/test_api_*.py` (offline, không cần khóa).

`web/.npmrc` đặt `legacy-peer-deps=true` để tránh lỗi "edgesOut" của npm khi cài vitest; bỏ khi npm sửa lỗi.

## Thời gian chạy đo được

Máy phát triển (CPU, rule-based, không khóa), 3 vòng / 3 phiên bản, bản production-local:

| Ảnh mẫu | Kích thước | Chẩn đoán | Chạy xử lý | Render phiên bản | Tinh chỉnh |
|---|---|---|---|---|---|
| Thiếu sáng | 600×400 | ~3,0 s (lần đầu sau khởi động) | 1,9 s | 0,1 s | 0,5 s |
| Nhiễu | 512×512 | — | 2,0 s | — | — |

Khởi động server mất vài chục giây vì phải import pipeline. Có Gemini thì mỗi vòng thêm thời gian gọi
mạng; số trên không áp dụng cho trường hợp đó.

## Hạn chế đã biết

- **Phiên hỏi ý LangGraph nằm trong RAM.** Restart server hoặc có hơn 32 phiên đang chờ trả lời thì phiên
  cũ mất. App báo "Phiên hỏi ý đã hết hạn" và cho hỏi ý lại; dữ liệu đã lưu không mất.
- **Chạy đồng bộ, không có tiến độ.** Một lần chạy là một request; UI chỉ hiện spinner, không có phần trăm và
  không hủy được. Theo thiết kế, đóng tab giữa chừng thì server vẫn chạy xong và lưu kết quả (chưa kiểm
  thực tế).
- **Chất lượng pipeline chưa được đánh giá.** "Đạt ngưỡng đánh giá của hệ thống" và "Điểm đánh giá ước lượng"
  là tiêu chí nội bộ, không phải độ chính xác.
- **Route cũ không có đăng nhập.** `/api/v1/diagnose`, `/process`, `/render`, `/refine`, `/sessions`, `/memory`
  (Gradio và script dùng) vẫn mở và nhận `user_id` từ client. Không đưa server ra mạng ngoài.
- **Xoay EXIF khác route cũ.** API mới xoay ảnh theo EXIF khi upload; route cũ thì không, nên cùng một ảnh
  có thể cho kích thước khác nhau giữa hai đường.
- **Góp ý về da** như "da hơi vàng" được luật hiểu thành điều chỉnh vùng khuôn mặt. Ảnh không có khuôn mặt
  thì có thể không thấy thay đổi.
- Ảnh được tải lại mỗi lần mở phiên (`Cache-Control: no-cache`, chưa có 304), để ảnh đã xóa hoặc của
  người khác trên cùng trình duyệt không hiện lại từ cache.
- Bundle JS khoảng 580 kB (gzip 177 kB) trong một file; với app chạy local thì chấp nhận được.

Luồng hội thoại hiện tại dùng `/chat` để làm rõ và cập nhật phương án, `/execute` để chạy khi người dùng bấm “Xử lý ảnh”/“Xử lý lại”. UI hiển thị nguồn ảnh và giữ lịch sử kết quả. Tham khảo `docs/plan_conversational_processing.md` và contract trong `docs/interfaces.md`.
