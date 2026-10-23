import { ImageIcon, Plus } from "lucide-react";
import { Link, NavLink } from "react-router-dom";

import { errorMessage } from "@/api/client";
import { useSessions } from "@/api/workspace";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDateTime, formatRelative } from "@/lib/labels";
import { cn } from "@/lib/utils";

import { StatusBadge } from "./StatusBadge";

export function SessionSidebar({ onNavigate }: { onNavigate?: () => void }) {
  const sessions = useSessions();
  return (
    <nav aria-label="Phiên xử lý" className="flex h-full min-h-0 flex-col">
      <div className="p-3">
        <Button asChild className="w-full justify-start">
          <Link to="/" onClick={onNavigate}>
            <Plus aria-hidden />
            Phiên mới
          </Link>
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {sessions.isPending ? (
          <div className="grid gap-2 p-1" aria-label="Đang tải danh sách phiên">
            {[0, 1, 2].map((key) => (
              <Skeleton key={key} className="h-14 w-full" />
            ))}
          </div>
        ) : sessions.isError ? (
          <div className="grid gap-2 p-2 text-sm">
            <p role="alert">{errorMessage(sessions.error)}</p>
            <Button variant="outline" size="sm" onClick={() => sessions.refetch()}>
              Thử lại
            </Button>
          </div>
        ) : sessions.data.length === 0 ? (
          <p className="p-2 text-sm text-muted-foreground">Chưa có phiên nào.</p>
        ) : (
          <ul className="grid gap-1">
            {sessions.data.map((item) => (
              <li key={item.id}>
                <NavLink
                  to={`/s/${item.id}`}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-3 rounded-md p-2 text-sm hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring",
                      isActive && "bg-accent text-accent-foreground",
                    )
                  }
                >
                  {item.thumbnail_url ? (
                    <img
                      src={item.thumbnail_url}
                      alt=""
                      className="size-10 shrink-0 rounded object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <span className="flex size-10 shrink-0 items-center justify-center rounded bg-muted">
                      <ImageIcon className="size-4 text-muted-foreground" aria-hidden />
                    </span>
                  )}
                  <span className="grid min-w-0 flex-1 gap-1">
                    <span className="truncate font-medium">{item.title}</span>
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <StatusBadge status={item.status} className="px-1.5 py-0 text-[11px]" />
                      <time dateTime={item.updated_at} title={formatDateTime(item.updated_at)}>
                        {formatRelative(item.updated_at)}
                      </time>
                    </span>
                  </span>
                </NavLink>
              </li>
            ))}
          </ul>
        )}
      </div>
    </nav>
  );
}
