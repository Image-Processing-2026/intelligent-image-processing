import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DONE, RENDERED, SESSION_ID, SUMMARY, VARIANTS } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
];
const SESSION_PATH = `/workspace/sessions/${SESSION_ID}`;

describe("result view", () => {
  it("compares original and result with a keyboard-operable slider", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DONE }]);
    renderApp(`/s/${SESSION_ID}`);
    const slider = await screen.findByRole("slider", { name: "Vị trí so sánh" });
    expect(slider).toHaveValue("50");
    slider.focus();
    fireEvent.change(slider, { target: { value: "70" } });
    expect(slider).toHaveValue("70");
    const after = screen.getAllByAltText("v1 · Kết quả pipeline")[0];
    expect(after).toHaveStyle({ clipPath: "inset(0 0 0 70%)" });
    expect(screen.getByText("Đánh giá của kết quả pipeline (v1)")).toBeInTheDocument();
  });

  it("shows ranked variants with recommendation reason and preview label", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DONE }]);
    renderApp(`/s/${SESSION_ID}`);
    const section = await screen.findByRole("region", { name: "Các phiên bản" });
    const items = Array.from(section.querySelectorAll<HTMLElement>(":scope > ul > li"));
    expect(items).toHaveLength(2);
    expect(within(items[0]).getByText("Hạng 1")).toBeInTheDocument();
    expect(within(items[0]).getByText("Đề xuất")).toBeInTheDocument();
    expect(within(items[0]).getByText("Lý do đề xuất: theo ý định của bạn")).toBeInTheDocument();
    expect(within(items[1]).queryByText("Đề xuất")).toBeNull();
    expect(within(section).getAllByText("Xem trước, cạnh dài ≤ 1024px")).toHaveLength(2);
    expect(within(section).getByText("Xếp hạng theo điểm ước lượng")).toBeInTheDocument();
  });

  it("renders the chosen variant and offers a full-resolution download", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DONE },
      {
        method: "POST",
        path: `${SESSION_PATH}/variants/${VARIANTS[0].id}/render`,
        body: RENDERED,
      },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const section = await screen.findByRole("region", { name: "Các phiên bản" });
    await user.click(within(section).getAllByRole("button", { name: "Chọn và render" })[0]);

    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/render"))).toBe(true),
    );
    expect(
      (await screen.findAllByText("v2 · Phiên bản đã chọn (Tự nhiên)", { selector: "span" }))
        .length,
    ).toBeGreaterThan(0);
    const link = within(section).getByRole("link", { name: /Tải về 640×480/ });
    expect(link).toHaveAttribute("href", expect.stringContaining("?download=1"));
    expect(within(section).getByText("Đã chọn")).toBeInTheDocument();
    expect(screen.getByText(/ở độ phân giải gốc 640×480/)).toBeInTheDocument();
  });

  it("shows the iteration timeline with plan, reasoning and knowledge ids", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DONE }]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: /02 Xử lý/ }));
    await user.click(screen.getByText("Chi tiết các vòng & chỉ số"));
    const timeline = screen.getByRole("region", { name: "Các vòng xử lý" });
    await user.click(within(timeline).getByRole("button", { name: /Vòng 1/ }));
    expect(within(timeline).getByText("Chỉnh gamma @ Toàn ảnh (gamma=1.4)")).toBeInTheDocument();
    expect(within(timeline).getByText("Lý do: Ảnh tối, tăng gamma.")).toBeInTheDocument();
    expect(within(timeline).getByText("underexposed_global")).toBeInTheDocument();
    expect(within(timeline).getByText("92,1")).toBeInTheDocument(); // chỉ số sau
  });

  it("keeps technical JSON collapsed until opened", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DONE }]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const toggle = await screen.findByRole("button", { name: /Nâng cao: dữ liệu kỹ thuật/ });
    expect(screen.queryByText(/"final_decision"/)).toBeNull();
    await user.click(toggle);
    expect(await screen.findByText(/"final_decision": "SHIP"/)).toBeInTheDocument();
  });

  it("opens the zoom dialog with zoom levels", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DONE }]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Phóng to" }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("radio", { name: "200%" }));
    expect(within(dialog).getByRole("img")).toHaveAttribute("width", "1280");
  });
});
