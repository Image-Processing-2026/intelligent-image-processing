import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  DETAIL,
  DONE,
  PIPELINE_TREATMENT,
  PROCESS,
  SESSION_ID,
  SUMMARY,
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

describe("conversation and explicit execution", () => {
  it("executes the displayed revision only after clicking", async () => {
    const fetchMock = mockApi([...BASE, {path: SESSION_PATH, body: DETAIL}, {method: "POST", path: `${SESSION_PATH}/execute`, body: NO_VARIANTS}]);
    renderApp(`/s/${SESSION_ID}`);
    await userEvent.setup().click(await screen.findByRole("button", {name: "Xử lý ảnh"}));
    expect(await screen.findByRole("heading", {name: "Kết quả xử lý"})).toBeInTheDocument();
    const post = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/execute"));
    expect(JSON.parse(String(post?.[1]?.body)).revision).toBe(1);
  });
  it("keeps an unsent request from being omitted by execution", async () => {
    mockApi([...BASE, {path: SESSION_PATH, body: DETAIL}]);
    renderApp(`/s/${SESSION_ID}`);
    const input = await screen.findByPlaceholderText(NOTE_PLACEHOLDER);
    await userEvent.setup().type(input, "sáng hơn");
    expect(screen.getByRole("button", {name: "Xử lý ảnh"})).toBeDisabled();
  });
  it("keeps the typed request after a failed chat", async () => {
    mockApi([...BASE, {path: SESSION_PATH, body: DETAIL}, {method: "POST", path: `${SESSION_PATH}/chat`, status: 500, body: {error: {code: "pipeline_failed", message: "Chưa cập nhật được phương án"}}}]);
    renderApp(`/s/${SESSION_ID}`);
    const input = await screen.findByPlaceholderText(NOTE_PLACEHOLDER);
    const user = userEvent.setup();
    await user.type(input, "sáng hơn");
    await user.click(screen.getByRole("button", {name: "Gửi bổ sung phân tích"}));
    await screen.findAllByText("Chưa cập nhật được phương án");
    expect(input).toHaveValue("sáng hơn");
  });
  it("locks execution while a chat is updating", async () => {
    mockApi([...BASE, {path: SESSION_PATH, body: DETAIL}, {method: "POST", path: `${SESSION_PATH}/chat`, pending: true}]);
    renderApp(`/s/${SESSION_ID}`);
    const input = await screen.findByPlaceholderText(NOTE_PLACEHOLDER);
    const user = userEvent.setup();
    await user.type(input, "sáng hơn");
    await user.click(screen.getByRole("button", {name: "Gửi bổ sung phân tích"}));
    expect(screen.getByRole("button", {name: "Xử lý ảnh"})).toBeDisabled();
    expect(input).toBeDisabled();
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
