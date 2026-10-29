import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DONE, SESSION_ID, SUMMARY } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
];
const SESSION_PATH = `/workspace/sessions/${SESSION_ID}`;

describe("rating is replaced by conversation", () => {
  it("does not show the rating form for a result", async () => {
    mockApi([...BASE, {path: SESSION_PATH, body: DONE}]);
    renderApp(`/s/${SESSION_ID}`);
    await screen.findByRole("button", {name: "Xử lý lại"});
    expect(screen.queryByRole("region", {name: "Đánh giá của bạn"})).toBeNull();
  });
});

describe("export", () => {
  it("links to the session JSON export", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DONE }]);
    renderApp(`/s/${SESSION_ID}`);
    const link = await screen.findByRole("link", { name: "Xuất JSON phiên" });
    expect(link).toHaveAttribute("href", `/api/v1/workspace/sessions/${SESSION_ID}/export`);
    expect(link).toHaveAttribute("download", `session-${SESSION_ID}.json`);
  });
});
