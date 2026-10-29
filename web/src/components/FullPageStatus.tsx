import { Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";

export function FullPageSpinner({ label = "Đang tải…" }: { label?: string }) {
  return (
    <div className="flex min-h-svh items-center justify-center gap-2 text-sm text-muted-foreground" role="status">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </div>
  );
}

export function FullPageError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-3 p-4 text-center">
      <p className="max-w-sm text-sm" role="alert">
        {message}
      </p>
      <Button variant="outline" onClick={onRetry}>
        Thử lại
      </Button>
    </div>
  );
}
