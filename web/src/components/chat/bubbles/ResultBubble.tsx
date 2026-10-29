import { useMutation } from "@tanstack/react-query";
import { Check, Loader2, MessageSquarePlus, Microscope } from "lucide-react";
import { toast } from "sonner";

import { errorMessage } from "@/api/client";
import type { Rating, SessionDetail, Variant } from "@/api/types";
import { saveRating } from "@/api/workspace";
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

/** Bong bóng kết quả: CompareSlider + badge kết cục + chấp nhận / góp ý / xem thông số thực thi. */
export function ResultBubble({
  detail,
  onRender,
  renderingId,
  busy,
  onRated,
  onFocusInput,
  onOpenExecution,
  children,
}: ResultBubbleProps) {
  const process = detail.process;
  const accept = useMutation({
    mutationFn: () => {
      if (!process) throw new Error("Chưa có kết quả để chấp nhận.");
      return saveRating(detail.id, {
        run_id: process.run_id,
        treatment_id: detail.active_treatment_id,
        rating: "good",
        note: "",
      });
    },
    onSuccess: (saved) => {
      onRated(saved);
      toast.success("Đã lưu đánh giá.");
    },
    onError: (error) => toast.error(errorMessage(error)),
  });
  if (!process) return null;
  const accepted = detail.ratings.some(
    (r) =>
      r.run_id === process.run_id &&
      r.treatment_id === detail.active_treatment_id &&
      r.rating === "good",
  );
  const locked = busy || detail.status !== "done";
  return (
    <div className="grid gap-3">
      <ResultCard
        detail={detail}
        process={process}
        onRender={onRender}
        renderingId={renderingId}
        busy={locked}
      >
        {children}
      </ResultCard>
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          onClick={() => accept.mutate()}
          disabled={locked || accept.isPending || accepted}
        >
          {accept.isPending ? (
            <Loader2 className="animate-spin" aria-hidden />
          ) : (
            <Check aria-hidden />
          )}
          {accepted ? "Đã chấp nhận" : "Chấp nhận kết quả"}
        </Button>
        <Button type="button" variant="outline" onClick={onFocusInput} disabled={locked}>
          <MessageSquarePlus aria-hidden />
          Góp ý tinh chỉnh
        </Button>
        <Button type="button" variant="ghost" onClick={onOpenExecution}>
          <Microscope aria-hidden />
          Xem thông số thực thi
        </Button>
      </div>
    </div>
  );
}
