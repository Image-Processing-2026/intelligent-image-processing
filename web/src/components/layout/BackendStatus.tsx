import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { apiFetch, onConnectivity } from "@/api/client";
import { SYSTEM_KEY, useSystemInfo } from "@/api/workspace";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface Health {
  status: string;
}

/** Chế độ chạy theo cấu hình; nhắc rằng có khóa không có nghĩa là Gemini đã chạy. */
export function modeLabel(keyConfigured: boolean | undefined): string {
  if (keyConfigured === undefined) return "Chế độ: chưa rõ";
  return keyConfigured
    ? "Chế độ: có khóa Gemini (nguồn thật xem ở từng kết quả)"
    : "Chế độ: rule-based (không có khóa Gemini)";
}

/**
 * Chỉ báo backend + chế độ chạy. Không poll (D3): kiểm tra /health khi tải trang, cập nhật theo
 * kết quả của các request khác (mất kết nối ↔ kết nối lại), và có nút thử lại.
 */
export function BackendStatus() {
  const queryClient = useQueryClient();
  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => apiFetch<Health>("/health", { skipAuthRedirect: true }),
    retry: false,
    staleTime: Infinity,
  });
  const system = useSystemInfo();
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => onConnectivity(setOnline), []);

  const reachable = online ?? (health.isPending ? null : health.data?.status === "ok");
  const label =
    reachable === null ? "Đang kiểm tra" : reachable ? "Hoạt động" : "Mất kết nối";
  const mode = modeLabel(system.data?.gemini_key_configured);

  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className="inline-flex items-center gap-1.5 rounded-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
            role="status"
            tabIndex={0}
          >
            <span
              aria-hidden
              className={cn(
                "size-2 rounded-full",
                reachable === null
                  ? "bg-muted-foreground"
                  : reachable
                    ? "bg-success"
                    : "bg-destructive",
              )}
            />
            <span className="sr-only sm:not-sr-only">Backend: {label}</span>
            <span className="sr-only">. {mode}</span>
          </span>
        </TooltipTrigger>
        <TooltipContent>{mode}</TooltipContent>
      </Tooltip>
      {system.data && reachable ? (
        <span className="hidden lg:inline">
          · {system.data.gemini_key_configured ? "Có khóa Gemini" : "Rule-based"}
        </span>
      ) : null}
      {reachable === false ? (
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={() => {
            void health.refetch();
            void queryClient.invalidateQueries({ queryKey: SYSTEM_KEY });
          }}
        >
          <RefreshCw aria-hidden />
          Thử lại
        </Button>
      ) : null}
    </span>
  );
}
