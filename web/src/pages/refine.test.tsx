import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  DONE,
  PIPELINE_TREATMENT,
  REFINED,
  REFINE_TREATMENT,
  SESSION_ID,
  SUMMARY,
} from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
];
const FEEDBACK_PLACEHOLDER = /Nhập góp ý tinh chỉnh cho kết quả/;
const SESSION_PATH = `/workspace/sessions/${SESSION_ID}`;

describe("refine by feedback", () => {
  it("sends feedback for the active version and shows the new version", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DONE },
      { method: "POST", path: `${SESSION_PATH}/refine`, body: REFINED },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const box = await screen.findByPlaceholderText(FEEDBACK_PLACEHOLDER);
    await user.type(box, "da hơi vàng");
    await user.click(screen.getByRole("button", { name: "Gửi góp ý tinh chỉnh" }));

    expect(await screen.findByText(/Đã tạo phiên bản/)).toHaveTextContent("v2 từ v1");
    const post = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/refine"));
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      feedback: "da hơi vàng",
      treatment_id: PIPELINE_TREATMENT.id,
    });
    expect(screen.getByText("Giảm ám vàng toàn ảnh")).toBeInTheDocument();
    expect(screen.getByText("Góp ý của bạn")).toBeInTheDocument();
    expect(box).toHaveValue("");

    const history = screen.getByRole("region", { name: "Lịch sử phiên bản phác đồ" });
    const current = within(history).getByText("Đang dùng").closest("li");
    expect(current).toHaveTextContent("v2 · Tinh chỉnh theo góp ý");
    expect(current).toHaveTextContent("Lạnh hơn (bớt vàng) (mức 1)");
  });

  it("keeps the feedback text when refining fails", async () => {
    mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DONE },
      {
        method: "POST",
        path: `${SESSION_PATH}/refine`,
        status: 500,
        body: { error: { code: "pipeline_failed", message: "Xử lý ảnh không thành công." } },
      },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const box = await screen.findByPlaceholderText(FEEDBACK_PLACEHOLDER);
    await user.type(box, "sáng hơn");
    await user.click(screen.getByRole("button", { name: "Gửi góp ý tinh chỉnh" }));
    expect((await screen.findAllByText("Xử lý ảnh không thành công.")).length).toBeGreaterThan(0);
    expect(box).toHaveValue("sáng hơn");
  });

  it("goes back to an earlier version without re-running the pipeline", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: REFINED },
      {
        method: "POST",
        path: `${SESSION_PATH}/treatments/${PIPELINE_TREATMENT.id}/activate`,
        body: { ...REFINED, active_treatment_id: PIPELINE_TREATMENT.id },
      },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const history = await screen.findByRole("region", { name: "Lịch sử phiên bản phác đồ" });
    await user.click(within(history).getByRole("button", { name: "Dùng bản này" }));
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/activate"))).toBe(true),
    );
    expect(
      await within(history).findByRole("button", { name: "Dùng bản này" }),
    ).toBeInTheDocument();
    const current = within(history).getByText("Đang dùng").closest("li");
    expect(current).toHaveTextContent("v1 · Kết quả pipeline");
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/refine"))).toBe(false);
  });

  it("says plainly when the feedback was not understood", async () => {
    const notUnderstood = {
      ...REFINED,
      messages: REFINED.messages.map((m) =>
        m.kind === "refine_result"
          ? { ...m, content: { ...m.content, source: "none", notes: [], adjustments: [] } }
          : m,
      ),
      treatments: [PIPELINE_TREATMENT, { ...REFINE_TREATMENT, refine_source: "none" }],
    };
    mockApi([...BASE, { path: SESSION_PATH, body: notUnderstood }]);
    renderApp(`/s/${SESSION_ID}`);
    expect(
      (await screen.findAllByText(/không hiểu góp ý — giữ nguyên phác đồ/)).length,
    ).toBeGreaterThan(0);
  });
});
