import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DETAIL, SESSION_ID, SUMMARY } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/samples", body: [] },
];

describe("new session page", () => {
  it("keeps the start button disabled until an image is chosen", async () => {
    mockApi([...BASE, { path: "/workspace/sessions", body: [] }]);
    renderApp("/");
    const button = await screen.findByRole("button", { name: "Bắt đầu chẩn đoán" });
    expect(button).toBeDisabled();
    expect(await screen.findByText("Chưa có phiên nào.")).toBeInTheDocument();
  });

  it("rejects unsupported files on the client", async () => {
    mockApi([...BASE, { path: "/workspace/sessions", body: [] }]);
    const user = userEvent.setup({ applyAccept: false });
    renderApp("/");
    const input = await screen.findByLabelText("Kéo thả ảnh vào đây");
    await user.upload(input, new File(["GIF89a"], "anim.gif", { type: "image/gif" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP.",
    );
    expect(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" })).toBeDisabled();
  });

  it("uploads, then opens the session with its diagnosis", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: "/workspace/sessions", body: [] },
      { method: "POST", path: "/workspace/sessions", status: 201, body: DETAIL },
      { path: `/workspace/sessions/${SESSION_ID}`, body: DETAIL },
    ]);
    const user = userEvent.setup();
    renderApp("/");
    const input = await screen.findByLabelText("Kéo thả ảnh vào đây");
    await user.upload(input, new File(["png"], "coffee.png", { type: "image/png" }));
    await user.click(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" }));

    expect(await screen.findByRole("heading", { name: "Chẩn đoán ban đầu" })).toBeInTheDocument();
    expect(screen.getByText("Chẩn đoán: Luật (rule_based)")).toBeInTheDocument();
    expect(screen.getAllByText("Thiếu sáng").length).toBeGreaterThan(0);
    expect(screen.getByText("Mức 2: Vừa")).toBeInTheDocument();

    const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
    const body = post?.[1]?.body as FormData;
    expect(body.get("file")).toBeInstanceOf(File);
    expect(body.get("max_iterations")).toBe("3");
    expect(body.has("num_variants")).toBe(false);
  });

  it("shows the server error and lets the user retry", async () => {
    mockApi([
      ...BASE,
      { path: "/workspace/sessions", body: [] },
      {
        method: "POST",
        path: "/workspace/sessions",
        status: 415,
        body: {
          error: { code: "unsupported_format", message: "Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP." },
        },
      },
    ]);
    const user = userEvent.setup();
    renderApp("/");
    const input = await screen.findByLabelText("Kéo thả ảnh vào đây");
    await user.upload(input, new File(["x"], "fake.png", { type: "image/png" }));
    await user.click(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" }));
    expect(await screen.findByText("Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" })).toBeEnabled();
  });
});

describe("session page", () => {
  it("lists sessions in the sidebar with status", async () => {
    mockApi([
      ...BASE,
      { path: "/workspace/sessions", body: [SUMMARY] },
      { path: `/workspace/sessions/${SESSION_ID}`, body: DETAIL },
    ]);
    renderApp(`/s/${SESSION_ID}`);
    const nav = await screen.findByRole("navigation", { name: "Phiên xử lý" });
    expect(await within(nav).findByText("Ảnh mẫu: Thiếu sáng")).toBeInTheDocument();
    expect(within(nav).getByText("Chờ trả lời")).toBeInTheDocument();
  });

  it("shows a not-found state for other users' sessions", async () => {
    mockApi([...BASE, { path: "/workspace/sessions", body: [] }]);
    renderApp(`/s/${SESSION_ID}`);
    expect(
      await screen.findByRole("heading", { name: "Không tìm thấy phiên xử lý" }),
    ).toBeInTheDocument();
  });

  it("deletes after confirmation", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: "/workspace/sessions", body: [SUMMARY] },
      { path: `/workspace/sessions/${SESSION_ID}`, body: DETAIL },
      { method: "DELETE", path: `/workspace/sessions/${SESSION_ID}`, status: 204 },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Xóa" }));
    const dialog = await screen.findByRole("alertdialog");
    expect(dialog).toHaveTextContent("Không thể hoàn tác.");
    await user.click(within(dialog).getByRole("button", { name: "Xóa phiên" }));
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(true),
    );
    expect(await screen.findByRole("heading", { name: "Phiên mới" })).toBeInTheDocument();
  });

  it("renames the session inline", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: "/workspace/sessions", body: [SUMMARY] },
      { path: `/workspace/sessions/${SESSION_ID}`, body: DETAIL },
      {
        method: "PATCH",
        path: `/workspace/sessions/${SESSION_ID}`,
        body: { ...SUMMARY, title: "Cà phê" },
      },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Đổi tên phiên" }));
    const input = screen.getByRole("textbox", { name: "Tên phiên" });
    await user.clear(input);
    await user.type(input, "Cà phê{Enter}");
    await waitFor(() => {
      const patch = fetchMock.mock.calls.find(([, init]) => init?.method === "PATCH");
      expect(patch?.[1]?.body).toBe('{"title":"Cà phê"}');
    });
  });
});
