import { Loader2 } from "lucide-react";

// Pipeline chạy đồng bộ, không có dữ liệu tiến độ: chỉ báo đang chờ, không phần trăm (D3).
export const WAITING_TEXT = "Đang xử lý, thời gian phụ thuộc ảnh và số vòng";

export function WaitingNotice({ text = WAITING_TEXT }: { text?: string }) {
  return (
    <div
      className="flex items-center gap-2 rounded-md border bg-muted/50 p-3 text-sm"
      role="status"
      aria-live="polite"
    >
      <Loader2 className="size-4 shrink-0 animate-spin text-primary" aria-hidden />
      <span>{text}</span>
    </div>
  );
}
