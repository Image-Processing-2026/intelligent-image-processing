import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

import { ME_QUERY_KEY } from "@/api/auth";
import { setUnauthorizedHandler } from "@/api/client";

/** 401 ở bất kỳ request nào: xóa dữ liệu đã cache rồi chuyển về trang đăng nhập. */
export function UnauthorizedRedirect() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  useEffect(() => {
    setUnauthorizedHandler((error) => {
      queryClient.clear();
      queryClient.setQueryData(ME_QUERY_KEY, null);
      toast.error(error.message, { id: "unauthenticated" });
      const next = `${window.location.pathname}${window.location.search}`;
      const query = next === "/" || next.startsWith("/login") ? "" : `?next=${encodeURIComponent(next)}`;
      navigate(`/login${query}`, { replace: true });
    });
    return () => setUnauthorizedHandler(null);
  }, [navigate, queryClient]);
  return null;
}
