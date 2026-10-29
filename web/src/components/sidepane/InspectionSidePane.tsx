import { X } from "lucide-react";
import { useEffect, useState } from "react";

import type { SessionDetail } from "@/api/types";
import { latestDiagnosis } from "@/components/studio/flow-data";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import { ExecutionTab, MasksTab, MetricsTab, ReasoningTab } from "./tabs";

export type InspectionTab = "masks" | "metrics" | "reasoning" | "execution";

const TABS: { id: InspectionTab; label: string }[] = [
  { id: "masks", label: "Mặt nạ" },
  { id: "metrics", label: "Số đo" },
  { id: "reasoning", label: "Lý do" },
  { id: "execution", label: "Thực thi" },
];

/** true khi màn hình đủ rộng để đặt bảng soi cạnh khung chat (≥ 1024px); không có matchMedia → true. */
export function useIsDesktop(): boolean {
  const query = "(min-width: 1024px)";
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" && window.matchMedia ? window.matchMedia(query).matches : true,
  );
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const media = window.matchMedia(query);
    const onChange = () => setMatches(media.matches);
    onChange();
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return matches;
}

interface InspectionSidePaneProps {
  detail: SessionDetail;
  tab: InspectionTab;
  onTabChange: (tab: InspectionTab) => void;
  onClose: () => void;
  className?: string;
}

/** Bảng soi kỹ thuật: chỉ ánh xạ dữ liệu thật của SessionDetail, không sinh số liệu thêm. */
export function InspectionSidePane({
  detail,
  tab,
  onTabChange,
  onClose,
  className,
}: InspectionSidePaneProps) {
  const diagnosis = latestDiagnosis(detail);
  return (
    <section
      aria-label="Bảng soi kỹ thuật"
      className={cn("flex min-h-0 flex-col bg-background", className)}
    >
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <h2 className="text-sm font-semibold">Chi tiết kỹ thuật</h2>
        <Button
          type="button"
          size="icon-sm"
          variant="ghost"
          className="ml-auto"
          aria-label="Đóng bảng chi tiết"
          onClick={onClose}
        >
          <X aria-hidden />
        </Button>
      </div>
      <div role="tablist" aria-label="Mục chi tiết" className="flex gap-1 border-b px-2 py-1.5">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={`inspect-tab-${item.id}`}
            aria-selected={tab === item.id}
            aria-controls="inspect-panel"
            onClick={() => onTabChange(item.id)}
            className={cn(
              "rounded-md px-2.5 py-1 text-xs font-medium text-muted-foreground hover:bg-accent",
              tab === item.id && "bg-accent text-accent-foreground",
            )}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id="inspect-panel"
        aria-labelledby={`inspect-tab-${tab}`}
        className="min-h-0 flex-1 overflow-y-auto p-3 text-sm"
      >
        {tab === "masks" ? <MasksTab detail={detail} diagnosis={diagnosis} /> : null}
        {tab === "metrics" ? <MetricsTab diagnosis={diagnosis} /> : null}
        {tab === "reasoning" ? <ReasoningTab detail={detail} diagnosis={diagnosis} /> : null}
        {tab === "execution" ? <ExecutionTab detail={detail} /> : null}
      </div>
    </section>
  );
}
