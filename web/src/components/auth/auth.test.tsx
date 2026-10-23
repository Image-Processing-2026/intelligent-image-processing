import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { safeNext } from "@/api/auth";
import { ALICE, UNAUTHENTICATED, mockApi, renderApp } from "@/test/utils";

const HEALTH = { path: "/health", body: { status: "ok" } };

describe("route guard", () => {
  it("redirects to login when not signed in", async () => {
    mockApi([{ path: "/auth/me", ...UNAUTHENTICATED }, HEALTH]);
    renderApp("/s/abc");
    expect(await screen.findByRole("heading", { name: "Đăng nhập" })).toBeInTheDocument();
  });

  it("shows the app shell with the username when signed in", async () => {
    mockApi([{ path: "/auth/me", body: ALICE }, HEALTH]);
    renderApp("/");
    expect(await screen.findByRole("button", { name: /alice/ })).toBeInTheDocument();
  });

  it("shows a retry message on network failure instead of redirecting", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))),
    );
    renderApp("/");
    expect(await screen.findByRole("alert")).toHaveTextContent("Không kết nối được máy chủ");
    expect(screen.getByRole("button", { name: "Thử lại" })).toBeInTheDocument();
  });
});

describe("login form", () => {
  it("shows the server error and keeps the form", async () => {
    mockApi([
      { path: "/auth/me", ...UNAUTHENTICATED },
      {
        method: "POST",
        path: "/auth/login",
        status: 401,
        body: {
          error: {
            code: "invalid_credentials",
            message: "Tên đăng nhập hoặc mật khẩu không đúng.",
          },
        },
      },
    ]);
    const user = userEvent.setup();
    renderApp("/login");
    await user.type(await screen.findByLabelText("Tên đăng nhập"), "alice");
    await user.type(screen.getByLabelText("Mật khẩu"), "wrong-password");
    await user.click(screen.getByRole("button", { name: "Đăng nhập" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tên đăng nhập hoặc mật khẩu không đúng.",
    );
    expect(screen.getByLabelText("Tên đăng nhập")).toHaveValue("alice");
    expect(screen.getByRole("button", { name: "Đăng nhập" })).toBeEnabled();
  });

  it("disables the submit button while the request is pending", async () => {
    mockApi([
      { path: "/auth/me", ...UNAUTHENTICATED },
      { method: "POST", path: "/auth/login", pending: true },
    ]);
    const user = userEvent.setup();
    renderApp("/login");
    await user.type(await screen.findByLabelText("Tên đăng nhập"), "alice");
    await user.type(screen.getByLabelText("Mật khẩu"), "password123");
    await user.click(screen.getByRole("button", { name: "Đăng nhập" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Đăng nhập" })).toBeDisabled());
  });

  it("navigates to next after a successful login", async () => {
    mockApi([
      { path: "/auth/me", ...UNAUTHENTICATED },
      { method: "POST", path: "/auth/login", body: ALICE },
      HEALTH,
    ]);
    const user = userEvent.setup();
    renderApp("/login?next=%2Fmemory");
    await user.type(await screen.findByLabelText("Tên đăng nhập"), "alice");
    await user.type(screen.getByLabelText("Mật khẩu"), "password123");
    await user.click(screen.getByRole("button", { name: "Đăng nhập" }));
    expect(await screen.findByRole("heading", { name: "Bộ nhớ của tôi" })).toBeInTheDocument();
  });
});

describe("register form", () => {
  it("validates locally before calling the server", async () => {
    const fetchMock = mockApi([{ path: "/auth/me", ...UNAUTHENTICATED }]);
    const user = userEvent.setup();
    renderApp("/register");
    await user.type(await screen.findByLabelText("Tên đăng nhập"), "bob");
    await user.type(screen.getByLabelText("Mật khẩu"), "password123");
    await user.type(screen.getByLabelText("Xác nhận mật khẩu"), "password124");
    await user.click(screen.getByRole("button", { name: "Tạo tài khoản" }));
    expect(await screen.findByText("Mật khẩu xác nhận không khớp.")).toBeInTheDocument();
    expect(screen.getByLabelText("Xác nhận mật khẩu")).toHaveFocus();
    const calledRegister = fetchMock.mock.calls.some(([url]) =>
      String(url).includes("/auth/register"),
    );
    expect(calledRegister).toBe(false);
  });

  it("shows a duplicate username error from the server", async () => {
    mockApi([
      { path: "/auth/me", ...UNAUTHENTICATED },
      {
        method: "POST",
        path: "/auth/register",
        status: 409,
        body: { error: { code: "username_taken", message: "Tên đăng nhập đã được dùng." } },
      },
    ]);
    const user = userEvent.setup();
    renderApp("/register");
    await user.type(await screen.findByLabelText("Tên đăng nhập"), "alice");
    await user.type(screen.getByLabelText("Mật khẩu"), "password123");
    await user.type(screen.getByLabelText("Xác nhận mật khẩu"), "password123");
    await user.click(screen.getByRole("button", { name: "Tạo tài khoản" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Tên đăng nhập đã được dùng.");
  });
});

describe("safeNext", () => {
  it("only allows internal paths", () => {
    expect(safeNext("/s/abc")).toBe("/s/abc");
    expect(safeNext("//evil.test")).toBe("/");
    expect(safeNext("https://evil.test")).toBe("/");
    expect(safeNext("/\\evil.test")).toBe("/");
    expect(safeNext(null)).toBe("/");
  });
});
