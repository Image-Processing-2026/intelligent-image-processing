import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Loader2, Pencil, X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { errorMessage } from "@/api/client";
import { SESSIONS_KEY, renameSession, sessionKey } from "@/api/workspace";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function EditableTitle({ sessionId, title }: { sessionId: string; title: string }) {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: (next: string) => renameSession(sessionId, next),
    onSuccess: () => {
      setEditing(false);
      setError(null);
      void queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
    },
    onError: (err) => {
      setError(errorMessage(err));
      toast.error(errorMessage(err));
    },
  });

  const save = () => {
    const next = draft.trim();
    if (!next || next.length > 120) {
      setError("Tên phiên cần 1–120 ký tự.");
      return;
    }
    if (next === title) {
      setEditing(false);
      return;
    }
    mutation.mutate(next);
  };

  if (!editing) {
    return (
      <div className="flex min-w-0 items-center gap-1">
        <h1 className="truncate text-xl font-semibold" title={title}>
          {title}
        </h1>
        <Button
          variant="ghost"
          size="icon"
          className="size-8 shrink-0"
          aria-label="Đổi tên phiên"
          onClick={() => {
            setDraft(title);
            setError(null);
            setEditing(true);
          }}
        >
          <Pencil aria-hidden />
        </Button>
      </div>
    );
  }

  return (
    <form
      className="grid min-w-0 flex-1 gap-1"
      onSubmit={(event) => {
        event.preventDefault();
        save();
      }}
    >
      <div className="flex items-center gap-1">
        <Input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") setEditing(false);
          }}
          aria-label="Tên phiên"
          aria-invalid={error ? true : undefined}
          maxLength={200}
          autoFocus
          disabled={mutation.isPending}
          className="h-9 min-w-0 max-w-md"
        />
        <Button type="submit" size="icon" className="size-9" aria-label="Lưu tên" disabled={mutation.isPending}>
          {mutation.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Check aria-hidden />}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="size-9"
          aria-label="Hủy đổi tên"
          onClick={() => setEditing(false)}
          disabled={mutation.isPending}
        >
          <X aria-hidden />
        </Button>
      </div>
      {error ? <p className="text-sm text-destructive">{error}</p> : null}
    </form>
  );
}
