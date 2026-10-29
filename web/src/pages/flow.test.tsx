import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  ASKING,
  DETAIL,
  DONE,
  PIPELINE_TREATMENT,
  PROCESS,
  SESSION_ID,
  SUMMARY,
  WITH_EDIT,
} from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
];
const NOTE_PLACEHOLDER = /Nhập phản hồi hoặc yêu cầu chỉnh sửa chẩn đoán/;
const SESSION_PATH = `/workspace/sessions/${SESSION_ID}`;

const NO_VARIANTS = {
  ...DONE,
  options: { ...DONE.options, num_variants: 1 },
  variants: [],
  treatments: [PIPELINE_TREATMENT],
  process: { ...PROCESS, recommended_variant: null, recommended_reason: null },
};

describe("answering questions", () => {
  it("starts processing without any question and shows a single result", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      { method: "POST", path: `${SESSION_PATH}/answers`, body: NO_VARIANTS },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);

    expect(screen.queryByRole("radio")).toBeNull(); // không còn câu hỏi phong cách
    await user.click(await screen.findByRole("button", { name: "Bắt đầu xử lý" }));

    expect(await screen.findByRole("heading", { name: "Kết quả xử lý" })).toBeInTheDocument();
    const post = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/answers"));
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ answers: {}, notes: "" });
    expect(screen.queryByRole("heading", { name: "Các phiên bản" })).toBeNull();
    expect(screen.getAllByText("Đạt ngưỡng đánh giá của hệ thống").length).toBeGreaterThan(0);
    expect(screen.getByText("Điểm đánh giá ước lượng")).toBeInTheDocument();
    expect(screen.queryByText(/%/)).toBeNull(); // điểm không hiển thị như phần trăm
    expect(screen.queryByText(/chính xác/i)).toBeNull();
  });

  it("still asks an ambiguity question and sends the chosen answer", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: ASKING },
      { method: "POST", path: `${SESSION_PATH}/answers`, body: NO_VARIANTS },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);

    expect(await screen.findByRole("radio", { name: /Giữ tông ấm/ })).toBeChecked();
    await user.click(screen.getByRole("radio", { name: /Sửa về trung tính/ }));
    await user.click(screen.getByRole("button", { name: "Bắt đầu xử lý" }));
    await screen.findByRole("heading", { name: "Kết quả xử lý" });
    const post = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/answers"));
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ answers: { tone: "fix" }, notes: "" });
  });

  it("adds an analysis note as a chip, removes it, then starts processing", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      { method: "POST", path: `${SESSION_PATH}/analysis-notes`, body: WITH_EDIT },
      { method: "DELETE", path: `${SESSION_PATH}/analysis-edits/0`, body: DETAIL },
      { method: "POST", path: `${SESSION_PATH}/answers`, body: NO_VARIANTS },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);

    await user.type(await screen.findByPlaceholderText(NOTE_PLACEHOLDER), "mặt còn tối");
    await user.click(screen.getByRole("button", { name: "Gửi bổ sung phân tích" }));

    const chip = "Thêm: Thiếu sáng @ Khuôn mặt (mức 2)";
    expect(await screen.findByRole("button", { name: `Xóa chỉnh sửa: ${chip}` })).toBeInTheDocument();
    const note = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/analysis-notes"));
    expect(JSON.parse(String(note?.[1]?.body))).toEqual({ message: "mặt còn tối" });
    expect(screen.getByPlaceholderText(NOTE_PLACEHOLDER)).toHaveValue(""); // đã gửi xong thì xóa ô nhập
    expect(screen.getByText("Bạn bổ sung")).toBeInTheDocument(); // lỗi do người dùng thêm

    await user.click(screen.getByRole("button", { name: `Xóa chỉnh sửa: ${chip}` }));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: `Xóa chỉnh sửa: ${chip}` })).toBeNull(),
    );

    await user.click(screen.getByRole("button", { name: "Bắt đầu xử lý" }));
    expect(await screen.findByRole("heading", { name: "Kết quả xử lý" })).toBeInTheDocument();
  });

  it("keeps the typed note and shows the error when the note fails", async () => {
    mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      {
        method: "POST",
        path: `${SESSION_PATH}/analysis-notes`,
        status: 409,
        body: { error: { code: "invalid_state", message: "Phiên không còn chờ trả lời." } },
      },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.type(await screen.findByPlaceholderText(NOTE_PLACEHOLDER), "nền cháy là cố ý");
    await user.click(screen.getByRole("button", { name: "Gửi bổ sung phân tích" }));
    expect(await screen.findByText("Phiên không còn chờ trả lời.")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(NOTE_PLACEHOLDER)).toHaveValue("nền cháy là cố ý");
  });

  it("shows regions with low confidence in the inspection side pane", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: WITH_EDIT }]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Xem chi tiết chẩn đoán" }));
    const pane = await screen.findByRole("region", { name: "Bảng soi kỹ thuật" });
    expect(within(pane).getByRole("img", { name: "Ảnh gốc với vùng được tô màu" })).toBeInTheDocument();
    expect(within(pane).getByText("Tin cậy thấp")).toBeInTheDocument();
    await user.click(within(pane).getByRole("tab", { name: "Số đo" }));
    expect(within(pane).getByRole("tab", { name: "Số đo" })).toHaveAttribute("aria-selected", "true");
    await user.click(within(pane).getByRole("button", { name: "Đóng bảng chi tiết" }));
    expect(screen.queryByRole("region", { name: "Bảng soi kỹ thuật" })).toBeNull();
  });

  it("locks the form and shows the waiting text while the pipeline runs", async () => {
    mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      { method: "POST", path: `${SESSION_PATH}/answers`, pending: true },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Bắt đầu xử lý" }));
    expect(
      await screen.findByText("Đang xử lý, thời gian phụ thuộc ảnh và số vòng"),
    ).toBeInTheDocument();
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
    await user.click(await screen.findByRole("button", { name: "Bắt đầu xử lý" }));
    await waitFor(() => {
      const reads = fetchMock.mock.calls.filter(
        ([url, init]) => String(url).endsWith(SESSION_PATH) && !init?.method,
      );
      expect(reads.length).toBeGreaterThanOrEqual(2);
    });
    expect(screen.getByRole("button", { name: "Bắt đầu xử lý" })).toBeEnabled();
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
    expect(screen.getAllByText("Đã dừng — xem kết quả và lịch sử").length).toBeGreaterThan(0);
    expect(screen.queryByText("Điểm đánh giá ước lượng")).toBeNull();
  });
});
