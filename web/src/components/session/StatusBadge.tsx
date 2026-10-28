import { CircleCheck, CircleHelp, Minus, TriangleAlert, Undo2, type LucideIcon } from "lucide-react";

import type { Outcome, OutcomeStatus, PlanSource, SessionStatus } from "@/api/types";
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

/** Kết cục trung thực của ảnh được trả: luôn có cả chữ lẫn biểu tượng, không chỉ màu. */
const OUTCOME_VIEW: Record<OutcomeStatus, { text: string; icon: LucideIcon; style: string }> = {
  ACCEPTED: { text: "Đã duyệt", icon: CircleCheck, style: "border-success/50 text-success" },
  UNVERIFIED: { text: "Chưa xác minh", icon: CircleHelp, style: "border-warning/50 text-warning" },
  NOT_MET: { text: "Chưa đạt", icon: TriangleAlert, style: "border-warning/50 text-warning" },
  KEPT_ORIGINAL: {
    text: "Giữ ảnh gốc",
    icon: Undo2,
    style: "border-border text-muted-foreground",
  },
  NO_CHANGE: { text: "Không cần chỉnh", icon: Minus, style: "border-border text-muted-foreground" },
};

export function OutcomeBadge({ outcome, className }: { outcome: Outcome; className?: string }) {
  const view = OUTCOME_VIEW[outcome.status];
  if (!view) return null;
  const Icon = view.icon;
  return (
    <Badge
      variant="outline"
      title={outcome.reason || undefined}
      className={cn("gap-1 font-normal", view.style, className)}
    >
      <Icon aria-hidden className="size-3.5" />
      {view.text}
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
