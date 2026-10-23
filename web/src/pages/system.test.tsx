import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SUMMARY } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const SYSTEM = {
  backend: "ok",
  gemini_key_configured: false,
  gemini_model: "gemini-test",
  memory_enabled: true,
  pipeline_concurrency: 1,
  registration_open: true,
  max_upload_mb: 20,
};
const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
  { path: "/workspace/samples", body: [] },
  { path: "/workspace/system", body: SYSTEM },
];

describe("backend indicator", () => {
  it("shows the backend state and the configured mode", async () => {
    mockApi(BASE);
    renderApp("/");
    const status = (await screen.findByText("Backend: Hoạt động")).closest("[role=status]");
    await waitFor(() =>
      expect(status).toHaveTextContent("Chế độ: rule-based (không có khóa Gemini)"),
    );
  });

  it("switches to disconnected after a failed request and offers a retry", async () => {
    const fetchMock = mockApi(BASE);
    const user = userEvent.setup();
    renderApp("/settings");
    await screen.findByText("Backend: Hoạt động");
    // Backend tắt: mọi request sau đều lỗi mạng
    fetchMock.mockImplementation(async () => Promise.reject(new TypeError("Failed to fetch")));
    const iterations = screen.getByRole("radiogroup", { name: "Số vòng tối đa" });
    await user.click(within(iterations).getByRole("radio", { name: "2" }));
    await user.click(screen.getByRole("button", { name: "Lưu mặc định" }));
    expect(await screen.findByText("Backend: Mất kết nối")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Thử lại" })).toBeInTheDocument();
  });

  it("says a configured key is not proof that Gemini ran", async () => {
    mockApi([
      ...BASE.filter((r) => r.path !== "/workspace/system"),
      { path: "/workspace/system", body: { ...SYSTEM, gemini_key_configured: true } },
    ]);
    renderApp("/settings");
    expect(
      await screen.findByText("Chế độ: có khóa Gemini (nguồn thật xem ở từng kết quả)"),
    ).toBeInTheDocument();
  });
});

describe("memory page", () => {
  it("shows the user's own memory stats", async () => {
    mockApi([
      ...BASE,
      {
        path: "/workspace/memory",
        body: {
          enabled: true,
          cases: 4,
          choices: { natural: 3, vivid: 1 },
          feedback: { cooler: 2 },
          preferred_style: "natural",
        },
      },
    ]);
    renderApp("/memory");
    expect(await screen.findByText("Số ca đã lưu")).toBeInTheDocument();
    const overview = screen.getByRole("heading", { name: "Tổng quan" }).closest("[data-slot=card]");
    expect(overview).toHaveTextContent("4");
    expect(overview).toHaveTextContent("Tự nhiên");
    expect(screen.getByText("Lạnh hơn (bớt vàng)")).toBeInTheDocument();
    expect(screen.getByText("3 lần")).toBeInTheDocument();
  });

  it("explains when memory is disabled", async () => {
    mockApi([...BASE, { path: "/workspace/memory", body: { enabled: false, cases: 0 } }]);
    renderApp("/memory");
    expect(await screen.findByText("Bộ nhớ ca bệnh đang tắt trên máy chủ.")).toBeInTheDocument();
  });
});

describe("settings page", () => {
  it("saves default iterations and variants on the server", async () => {
    const fetchMock = mockApi([
      ...BASE,
      {
        method: "PATCH",
        path: "/auth/me/settings",
        body: { ...ALICE, settings: { max_iterations: 2, num_variants: 1 } },
      },
    ]);
    const user = userEvent.setup();
    renderApp("/settings");
    const save = await screen.findByRole("button", { name: "Lưu mặc định" });
    expect(save).toBeDisabled();
    const iterations = screen.getByRole("radiogroup", { name: "Số vòng tối đa" });
    await user.click(within(iterations).getByRole("radio", { name: "2" }));
    const variants = screen.getByRole("radiogroup", { name: "Số phiên bản" });
    await user.click(within(variants).getByRole("radio", { name: "1" }));
    await user.click(save);
    await waitFor(() => {
      const patch = fetchMock.mock.calls.find(([, init]) => init?.method === "PATCH");
      expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ max_iterations: 2, num_variants: 1 });
    });
  });

  it("switches the theme and stores it in this browser", async () => {
    mockApi(BASE);
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const user = userEvent.setup();
    renderApp("/settings");
    const theme = await screen.findByRole("radiogroup", { name: "Chủ đề màu" });
    await user.click(within(theme).getByRole("radio", { name: "Tối" }));
    expect(document.documentElement).toHaveClass("dark");
    expect(setItem).toHaveBeenCalledWith("iid-theme", "dark");
    await user.click(within(theme).getByRole("radio", { name: "Sáng" }));
    expect(document.documentElement).not.toHaveClass("dark");
  });
});
