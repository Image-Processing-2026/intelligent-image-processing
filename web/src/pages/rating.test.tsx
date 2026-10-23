import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DONE, PIPELINE_TREATMENT, RUN_ID, SESSION_ID, SUMMARY } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
];
const SESSION_PATH = `/workspace/sessions/${SESSION_ID}`;

const SAVED = {
  id: "r".repeat(32),
  run_id: RUN_ID,
  treatment_id: PIPELINE_TREATMENT.id,
  rating: "bad" as const,
  note: "còn tối",
  created_at: "2026-10-22T11:00:00+00:00",
  updated_at: "2026-10-22T11:00:00+00:00",
};

describe("rating", () => {
  it("saves a rating with a note for the active version", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DONE },
      { method: "PUT", path: `${SESSION_PATH}/ratings`, body: SAVED },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const section = await screen.findByRole("region", { name: "Đánh giá của bạn" });
    const save = within(section).getByRole("button", { name: "Lưu đánh giá" });
    expect(save).toBeDisabled(); // chưa chọn mức
    await user.click(within(section).getByRole("radio", { name: "Chưa tốt" }));
    await user.type(within(section).getByLabelText("Ghi chú đánh giá"), "còn tối");
    await user.click(save);

    await waitFor(() => {
      const put = fetchMock.mock.calls.find(([, init]) => init?.method === "PUT");
      expect(JSON.parse(String(put?.[1]?.body))).toEqual({
        run_id: RUN_ID,
        treatment_id: PIPELINE_TREATMENT.id,
        rating: "bad",
        note: "còn tối",
      });
    });
    expect(await within(section).findByText(/Đã lưu lúc/)).toBeInTheDocument();
  });

  it("shows an existing rating after reopening", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: { ...DONE, ratings: [SAVED] } }]);
    renderApp(`/s/${SESSION_ID}`);
    const section = await screen.findByRole("region", { name: "Đánh giá của bạn" });
    expect(within(section).getByRole("radio", { name: "Chưa tốt" })).toBeChecked();
    expect(within(section).getByLabelText("Ghi chú đánh giá")).toHaveValue("còn tối");
    expect(within(section).getByRole("button", { name: "Lưu đánh giá" })).toBeDisabled();
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
