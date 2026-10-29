import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { AppRoutes, createQueryClient } from "@/App";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/lib/theme";

export interface MockRoute {
  method?: string;
  path: string;
  status?: number;
  body?: unknown;
  /** Trả về Promise chưa xong để kiểm tra trạng thái đang gửi. */
  pending?: boolean;
}

/** Giả lập fetch theo bảng route; request không khớp → 404 envelope. */
export function mockApi(routes: MockRoute[]) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const method = (init?.method ?? "GET").toUpperCase();
    const route = routes.find(
      (item) => (item.method ?? "GET").toUpperCase() === method && url === `/api/v1${item.path}`,
    );
    if (!route) {
      return new Response(JSON.stringify({ error: { code: "not_found", message: "nf" } }), {
        status: 404,
      });
    }
    if (route.pending) return new Promise<Response>(() => undefined);
    const status = route.status ?? 200;
    return new Response(status === 204 ? null : JSON.stringify(route.body ?? {}), { status });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

export const UNAUTHENTICATED = {
  status: 401,
  body: {
    error: {
      code: "unauthenticated",
      message: "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.",
    },
  },
};

export const ALICE = {
  id: "a".repeat(32),
  username: "alice",
  settings: { max_iterations: 3, num_variants: 3 },
  created_at: "2026-10-22T00:00:00+00:00",
};

export function renderApp(path: string) {
  const queryClient = createQueryClient();
  return render(
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <MemoryRouter initialEntries={[path]}>
            <AppRoutes />
          </MemoryRouter>
          <Toaster />
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>,
  );
}
