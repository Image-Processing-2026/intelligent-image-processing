import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DETAIL, DONE, PROCESS, SESSION_ID, SUMMARY } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
];
const SESSION_PATH = `/workspace/sessions/${SESSION_ID}`;

describe("answering questions", () => {
  it("sends the chosen answers and notes, waits honestly, then shows the result", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      { method: "POST", path: `${SESSION_PATH}/answers`, body: DONE },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);

    const balanced = await screen.findByRole("radio", { name: /Cân bằng/ });
    expect(balanced).toBeChecked(); // mặc định của câu hỏi
    await user.click(screen.getByRole("radio", { name: /Tự nhiên/ }));
    await user.type(screen.getByLabelText("Ghi chú thêm (tùy chọn)"), "giữ ấm");
    await user.click(screen.getByRole("button", { name: "Chạy xử lý" }));

    expect(await screen.findByRole("heading", { name: "Kết quả xử lý" })).toBeInTheDocument();
    const post = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/answers"));
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      answers: { style: "natural" },
      notes: "giữ ấm",
    });
    expect(screen.getAllByText("Đạt ngưỡng đánh giá của hệ thống").length).toBeGreaterThan(0);
    expect(screen.getByText("Điểm đánh giá ước lượng")).toBeInTheDocument();
    expect(screen.getAllByText("Phác đồ: Luật (rule_based)")).toHaveLength(2);
    expect(screen.getByText("→ Tự nhiên (chỉnh nhẹ tay)")).toBeInTheDocument();
    expect(screen.queryByText(/%/)).toBeNull(); // điểm không hiển thị như phần trăm
    expect(screen.queryByText(/chính xác/i)).toBeNull();
  });

  it("locks the form and shows the waiting text while the pipeline runs", async () => {
    mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      { method: "POST", path: `${SESSION_PATH}/answers`, pending: true },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Chạy xử lý" }));
    expect(
      await screen.findByText("Đang xử lý, thời gian phụ thuộc ảnh và số vòng"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Chạy xử lý" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Xóa" })).toBeDisabled();
  });

  it("offers to clarify again when the question thread expired", async () => {
    const expired = {
      ...DETAIL,
      messages: [
        ...DETAIL.messages,
        {
          id: 9,
          role: "system" as const,
          kind: "notice" as const,
          created_at: "2026-10-22T10:05:00+00:00",
          content: { code: "thread_expired", text: "Phiên hỏi ý đã hết hạn. Hãy hỏi ý lại." },
        },
      ],
    };
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: expired },
      { method: "POST", path: `${SESSION_PATH}/clarify`, body: DETAIL },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Hỏi ý lại" }));
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/clarify"))).toBe(true),
    );
  });

  it("re-reads the session after a thread_expired error", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      {
        method: "POST",
        path: `${SESSION_PATH}/answers`,
        status: 409,
        body: { error: { code: "thread_expired", message: "Phiên hỏi ý đã hết hạn." } },
      },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Chạy xử lý" }));
    await waitFor(() => {
      const reads = fetchMock.mock.calls.filter(
        ([url, init]) => String(url).endsWith(SESSION_PATH) && !init?.method,
      );
      expect(reads.length).toBeGreaterThanOrEqual(2);
    });
    expect(screen.getByRole("button", { name: "Chạy xử lý" })).toBeEnabled();
  });
});

describe("result card", () => {
  it("flags the vlm_fallback source", async () => {
    const fallback = {
      ...DONE,
      process: { ...PROCESS, sources: { diagnosis: ["vlm_fallback"], plan: ["vlm_fallback"] } },
    };
    mockApi([...BASE, { path: SESSION_PATH, body: fallback }]);
    renderApp(`/s/${SESSION_ID}`);
    expect(await screen.findByText("Gọi Gemini lỗi, đã dùng luật dự phòng.")).toBeInTheDocument();
  });

  it("shows PSNR/SSIM against the ground truth for synthetic runs", async () => {
    const synthetic = {
      ...DONE,
      process: {
        ...PROCESS,
        is_synthetic: true,
        final_decision: "STOP_BEST_EFFORT",
        final_evaluation: { psnr: 27.5, ssim: 0.91, mse: 12 },
      },
    };
    mockApi([...BASE, { path: SESSION_PATH, body: synthetic }]);
    renderApp(`/s/${SESSION_ID}`);
    expect(await screen.findByText("PSNR (so với ảnh chuẩn)")).toBeInTheDocument();
    expect(screen.getByText("Đã dừng — xem kết quả và lịch sử")).toBeInTheDocument();
    expect(screen.queryByText("Điểm đánh giá ước lượng")).toBeNull();
  });

  it("does not offer answering again once done", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DONE }]);
    renderApp(`/s/${SESSION_ID}`);
    await screen.findByRole("heading", { name: "Kết quả xử lý" });
    expect(screen.queryByRole("button", { name: "Chạy xử lý" })).toBeNull();
    expect(screen.getByText("Đã trả lời — xem câu trả lời bên dưới.")).toBeInTheDocument();
    // Card câu hỏi đã trả lời hiện đúng lựa chọn đã gửi (không phải mặc định)
    expect(screen.getByRole("radio", { name: /Tự nhiên/ })).toBeChecked();
  });
});
