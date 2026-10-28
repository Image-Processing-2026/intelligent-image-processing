import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { DETAIL, DONE, SESSION_ID, SUMMARY } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

function prepare(detail = DONE) {
  mockApi([
    { path: "/auth/me", body: ALICE }, { path: "/health", body: { status: "ok" } },
    { path: "/workspace/sessions", body: [SUMMARY] },
    { path: `/workspace/sessions/${SESSION_ID}`, body: detail },
  ]);
  renderApp(`/s/${SESSION_ID}`);
}

describe("phase studio", () => {
  it("opens a completed session in results and keeps analysis available separately", async () => {
    prepare();
    const user = userEvent.setup();
    expect(await screen.findByRole("heading", { name: "Kết quả xử lý" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Chẩn đoán ban đầu" })).toBeNull();
    await user.click(screen.getByRole("button", { name: /01 Phân tích/ }));
    expect(screen.getByRole("heading", { name: "Chẩn đoán ban đầu" })).toBeInTheDocument();
    expect(screen.getByText("→ Tự nhiên (chỉnh nhẹ tay)")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Kết quả xử lý" })).toBeNull();
  });

  it("locks future phases until processing has started", async () => {
    prepare(DETAIL);
    expect(await screen.findByRole("button", { name: /02 Xử lý/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /03 Kết quả/ })).toBeDisabled();
    const log = screen.getByRole("complementary", { name: "Flow log" });
    expect(within(log).getByText("Ảnh trong nhà thiếu sáng")).toBeInTheDocument();
  });

  it("inspects a saved processing step using the real plan reason and output image", async () => {
    prepare();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /02 Xử lý/ }));
    const inspector = screen.getByRole("region", { name: "Chi tiết bước: Lập phác đồ" });
    expect(within(inspector).getByText("Ảnh tối, tăng gamma.")).toBeInTheDocument();
    expect(within(inspector).getByText(/gamma=1.4/)).toBeInTheDocument();
    const log = screen.getByRole("complementary", { name: "Flow log" });
    await user.click(within(log).getByRole("button", { name: /03 Xử lý ảnh/ }));
    expect(within(log).getByRole("img", { name: "Đầu ra: Xử lý ảnh" })).toBeInTheDocument();
    expect(within(log).getByText(/chưa có log thực thi riêng từng thao tác/)).toBeInTheDocument();
  });

  it("switches both the preview and inspector to the selected recorded round", async () => {
    prepare({ ...DONE, process: { ...DONE.process!, history: [DONE.process!.history[0], {
      ...DONE.process!.history[0], iteration: 2,
      plan: { ...DONE.process!.history[0].plan, reasoning: "Lý do riêng của vòng hai" },
    }] } });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /02 Xử lý/ }));
    const log = screen.getByRole("complementary", { name: "Flow log" });
    await user.click(within(log).getByRole("button", { name: /^Vòng 2$/ }));
    expect(within(log).getByText("Lý do riêng của vòng hai")).toBeInTheDocument();
    const panel = screen.getByRole("region", { name: "Phác đồ và thực thi" });
    expect(within(panel).getByRole("heading", { name: "Phác đồ & thực thi" }).parentElement).toHaveTextContent("Vòng 2");
  });
});
