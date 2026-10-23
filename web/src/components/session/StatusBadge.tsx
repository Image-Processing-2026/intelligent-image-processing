import type { PlanSource, SessionStatus } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { DECISION_LABELS, SOURCE_LABELS, STATUS_LABELS, label } from "@/lib/labels";
import { cn } from "@/lib/utils";

const STATUS_STYLES: Record<SessionStatus, string> = {
  uploaded: "border-border text-muted-foreground",
  awaiting_answers: "border-primary/40 text-primary",
  running: "border-warning/50 text-warning",
  done: "border-success/50 text-success",
  failed: "border-destructive/50 text-destructive",
};

export function StatusBadge({ status, className }: { status: SessionStatus; className?: string }) {
  return (
    <Badge variant="outline" className={cn("font-normal", STATUS_STYLES[status], className)}>
      {STATUS_LABELS[status] ?? status}
    </Badge>
  );
}

/** Nguồn THẬT của chẩn đoán/phác đồ, lấy từ field `source` của backend (không suy từ API key). */
export function SourceBadge({ source, prefix }: { source: PlanSource | string; prefix: string }) {
  const isFallback = source === "vlm_fallback";
  return (
    <Badge
      variant="outline"
      className={cn("font-normal", isFallback && "border-warning/50 text-warning")}
    >
      {prefix}: {label(SOURCE_LABELS, source)}
    </Badge>
  );
}

const DECISION_STYLES: Record<string, string> = {
  SHIP: "border-success/50 text-success",
  STOP_BEST_EFFORT: "border-warning/50 text-warning",
  RE_PROCESS: "border-primary/40 text-primary",
};

export function DecisionBadge({ decision }: { decision: string }) {
  return (
    <Badge variant="outline" className={cn("font-normal", DECISION_STYLES[decision])}>
      {label(DECISION_LABELS, decision)}
    </Badge>
  );
}
