import { Navigate, Outlet, useLocation, useSearchParams } from "react-router-dom";

import { safeNext, useMe } from "@/api/auth";
import { errorMessage } from "@/api/client";
import { FullPageError, FullPageSpinner } from "@/components/FullPageStatus";

/** Chặn route cần đăng nhập: chưa đăng nhập → /login?next=…; lỗi mạng → báo lỗi + thử lại. */
export function RequireAuth() {
  const me = useMe();
  const location = useLocation();
  if (me.isPending) return <FullPageSpinner />;
  if (me.isError) return <FullPageError message={errorMessage(me.error)} onRetry={() => me.refetch()} />;
  if (!me.data) {
    const next = `${location.pathname}${location.search}`;
    const query = next === "/" ? "" : `?next=${encodeURIComponent(next)}`;
    return <Navigate to={`/login${query}`} replace />;
  }
  return <Outlet />;
}

/**
 * Trang đăng nhập/đăng ký: đã đăng nhập (kể cả ngay sau khi đăng nhập thành công) thì chuyển
 * tới ?next= nếu là đường dẫn nội bộ, ngược lại về trang chính.
 */
export function RedirectIfAuthed() {
  const me = useMe();
  const [params] = useSearchParams();
  if (me.isPending) return <FullPageSpinner />;
  if (me.data) return <Navigate to={safeNext(params.get("next"))} replace />;
  return <Outlet />;
}
