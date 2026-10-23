import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, Save, ThumbsDown, ThumbsUp } from "lucide-react";
import { useId, useState } from "react";
import { toast } from "sonner";

import { errorMessage } from "@/api/client";
import type { Rating, SessionDetail } from "@/api/types";
import { saveRating, sessionKey } from "@/api/workspace";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatDateTime } from "@/lib/labels";

export const RATING_NOTE_MAX = 2000;

/**
 * Đánh giá của người dùng cho phiên bản đang xem (phục vụ đánh giá chất lượng hệ thống).
 * Mỗi (run, phiên bản) có một đánh giá; lưu lại là cập nhật.
 */
export function RatingBox({
  sessionId,
  runId,
  treatmentId,
  treatmentLabel,
  existing,
}: {
  sessionId: string;
  runId: string;
  treatmentId: string | null;
  treatmentLabel: string;
  existing: Rating | null;
}) {
  const queryClient = useQueryClient();
  const id = useId();
  const [rating, setRating] = useState<"good" | "bad" | null>(existing?.rating ?? null);
  const [note, setNote] = useState(existing?.note ?? "");
  const mutation = useMutation({
    mutationFn: () =>
      saveRating(sessionId, { run_id: runId, treatment_id: treatmentId, rating: rating!, note }),
    onSuccess: (saved) => {
      queryClient.setQueryData<SessionDetail>(sessionKey(sessionId), (old) =>
        old
          ? { ...old, ratings: [...old.ratings.filter((r) => r.id !== saved.id), saved] }
          : old,
      );
      toast.success("Đã lưu đánh giá.");
    },
    onError: (error) => toast.error(errorMessage(error)),
  });
  const dirty = rating !== (existing?.rating ?? null) || note !== (existing?.note ?? "");

  return (
    <section className="grid gap-2" aria-labelledby={`${id}-heading`}>
      <h3 id={`${id}-heading`} className="font-medium">
        Đánh giá của bạn
      </h3>
      <p className="text-sm text-muted-foreground">
        Cho phiên bản đang xem ({treatmentLabel}). Dùng để đánh giá chất lượng hệ thống về sau.
      </p>
      <form
        className="grid gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (rating && !mutation.isPending) mutation.mutate();
        }}
      >
        <ToggleGroup
          type="single"
          variant="outline"
          value={rating ?? ""}
          onValueChange={(value) => setRating(value ? (value as "good" | "bad") : null)}
          aria-label="Mức đánh giá"
          className="w-fit"
        >
          <ToggleGroupItem value="good" className="gap-1.5 px-3">
            <ThumbsUp aria-hidden />
            Tốt
          </ToggleGroupItem>
          <ToggleGroupItem value="bad" className="gap-1.5 px-3">
            <ThumbsDown aria-hidden />
            Chưa tốt
          </ToggleGroupItem>
        </ToggleGroup>
        <label htmlFor={id} className="sr-only">
          Ghi chú đánh giá
        </label>
        <Textarea
          id={id}
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Ghi chú (tùy chọn): điều gì tốt, điều gì chưa ổn"
          maxLength={RATING_NOTE_MAX}
          rows={2}
        />
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" size="sm" disabled={!rating || !dirty || mutation.isPending}>
            {mutation.isPending ? (
              <Loader2 className="animate-spin" aria-hidden />
            ) : (
              <Save aria-hidden />
            )}
            Lưu đánh giá
          </Button>
          {existing && !dirty ? (
            <span className="text-xs text-muted-foreground" role="status">
              Đã lưu lúc {formatDateTime(existing.updated_at)}
            </span>
          ) : dirty && existing ? (
            <span className="text-xs text-muted-foreground">Có thay đổi chưa lưu</span>
          ) : null}
        </div>
      </form>
    </section>
  );
}
