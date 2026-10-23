import { Loader2, Send } from "lucide-react";
import { useId, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export const FEEDBACK_MAX = 500;

/** Ô góp ý tiếng Việt; góp ý áp vào phiên bản phác đồ đang xem và tạo phiên bản mới. */
export function RefineBox({
  activeLabel,
  onSubmit,
  pending,
  disabled,
}: {
  activeLabel: string;
  onSubmit: (feedback: string) => Promise<unknown>;
  pending: boolean;
  disabled: boolean;
}) {
  const [text, setText] = useState("");
  const id = useId();
  const trimmed = text.trim();
  return (
    <section className="grid gap-2" aria-labelledby={`${id}-heading`}>
      <h3 id={`${id}-heading`} className="font-medium">
        Tinh chỉnh theo góp ý
      </h3>
      <form
        className="grid gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (!trimmed || disabled) return;
          // Chỉ xóa ô khi góp ý đã được áp dụng; lỗi thì giữ để thử lại
          onSubmit(trimmed).then(
            () => setText(""),
            () => undefined,
          );
        }}
      >
        <label htmlFor={id} className="text-sm text-muted-foreground">
          Góp ý sẽ áp vào phiên bản đang xem ({activeLabel}) và tạo một phiên bản mới.
        </label>
        <Textarea
          id={id}
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Ví dụ: da hơi vàng, trời gắt quá"
          maxLength={FEEDBACK_MAX}
          rows={2}
          disabled={disabled}
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">
            {text.length}/{FEEDBACK_MAX} ký tự
          </span>
          <Button type="submit" size="sm" disabled={disabled || !trimmed}>
            {pending ? <Loader2 className="animate-spin" aria-hidden /> : <Send aria-hidden />}
            Gửi góp ý
          </Button>
        </div>
      </form>
    </section>
  );
}
