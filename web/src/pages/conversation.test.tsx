import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it } from "vitest";
import { DETAIL, SESSION_ID, SUMMARY } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

it("keeps chat separate from executing the displayed proposal", async () => {
  const path = `/workspace/sessions/${SESSION_ID}`;
  const detail = {...DETAIL, conversation: {revision: 1, status: "ready", reply: "Điều chỉnh ánh sáng", source_mode: "rebuild", source_reason: "Dùng ảnh gốc", actions: []}};
  const fetch = mockApi([
    {path: "/auth/me", body: ALICE}, {path: "/health", body: {status: "ok"}},
    {path: "/workspace/sessions", body: [SUMMARY]}, {path, body: detail},
    {method: "POST", path: `${path}/execute`, body: detail},
  ]);
  renderApp(`/s/${SESSION_ID}`);
  expect(await screen.findByRole("region", {name: "Phương án sẽ áp dụng"})).toBeInTheDocument();
  expect(screen.queryByRole("region", {name: "Đánh giá của bạn"})).not.toBeInTheDocument();
  expect(screen.queryByText("Theo đề xuất của AI")).not.toBeInTheDocument();
  expect(fetch.mock.calls.some(([url]) => String(url).endsWith("/execute"))).toBe(false);
  await userEvent.setup().click(screen.getByRole("button", {name: "Xử lý ảnh"}));
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).endsWith("/execute"))).toBe(true));
});
