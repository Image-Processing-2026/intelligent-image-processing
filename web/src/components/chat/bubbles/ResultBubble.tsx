import { MessageSquarePlus, Microscope } from "lucide-react";
import type { Rating, SessionDetail, Variant } from "@/api/types";
import { ResultCard } from "@/components/session/ResultCard";
import { Button } from "@/components/ui/button";
interface ResultBubbleProps {
  detail: SessionDetail;
  onRender: (variant: Variant) => void;
  renderingId: string | null;
  busy: boolean;
  /** Lưu đánh giá xong: ghi vào cache của phiên. */
  onRated: (rating: Rating) => void;
  onFocusInput: () => void;
  onOpenExecution: () => void;
  /** Khu lịch sử phiên bản chèn vào giữa phần so sánh và phần chi tiết. */
  children?: React.ReactNode;
}


export function ResultBubble({detail, onRender, renderingId, busy, onFocusInput, onOpenExecution, children}: ResultBubbleProps) {
  if (!detail.process) return null;
  return <div className="grid gap-3">
    <ResultCard detail={detail} process={detail.process} onRender={onRender} renderingId={renderingId} busy={busy}>{children}</ResultCard>
    <div className="flex gap-2"><Button variant="outline" onClick={onFocusInput} disabled={busy}><MessageSquarePlus />Trao đổi về kết quả</Button><Button variant="ghost" onClick={onOpenExecution}><Microscope />Xem thông số thực thi</Button></div>
  </div>;
}
