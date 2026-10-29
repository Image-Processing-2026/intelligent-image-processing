import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, onTestFinished } from "vitest";
import { DETAIL, DONE, ITER_ARTIFACT, SESSION_ID, SUMMARY, WITH_EDIT } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";
import { buildFlowSteps } from "./flow-data";
import { retainedEvaluation } from "./assessment";
import { RoundAssessment } from "./RoundAssessment";
import type { SessionDetail } from "@/api/types";

const NOTE_PLACEHOLDER = /Nhập phản hồi hoặc yêu cầu chỉnh sửa chẩn đoán/;
const PATH = `/workspace/sessions/${SESSION_ID}`;
const base = (detail: SessionDetail) => [
  { path: "/auth/me", body: ALICE }, { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] }, { path: PATH, body: detail },
];

const tracked = {
  ...DONE.process!.history[0],
  defect_status: [{ type: "backlit_subject", region: "person", before: 3, after: 1, status: "improved" as const, confidence: .9 }],
  side_effects: [{ kind: "dark_noise" as const, region: "person", severity: 2, value: 2.5, before: 2, after: 5, evidence: "Measured noise increased" }],
};

describe("latest backend compatibility", () => {
  it("distinguishes regional measurements of a rejected attempt from the retained result", () => {
    render(<RoundAssessment round={{ ...tracked, rolled_back: true }} />);
    expect(screen.getByText(/Đánh giá ảnh thử đã bị loại bỏ/)).toHaveTextContent("không phải đánh giá của ảnh kết quả đang giữ");
  });
  it("does not present the discarded attempt score as the retained output score", () => {
    const process = { ...DONE.process!, rolled_back: true,
      final_evaluation: { estimated_quality_score: 95 }, history: [{ ...tracked, rolled_back: true }] };
    expect(retainedEvaluation(process)).toBeNull();
    const withRetainedRound = { ...process, history: [{ ...tracked, eval_score: { estimated_quality_score: 71 } }, { ...tracked, rolled_back: true, iteration: 2 }] };
    expect(retainedEvaluation(withRetainedRound)).toEqual({ estimated_quality_score: 71 });
    const summary = buildFlowSteps({ ...DONE, process }, "result")[0];
    expect(summary.output.some((f) => f.label === "Điểm ước lượng")).toBe(false);
  });
  it("includes recorded regional defect changes and side effects in decision inputs", () => {
    const detail = { ...DONE, process: { ...DONE.process!, history: [tracked] } };
    const decision = buildFlowSteps(detail, "processing").find((s) => s.id === "decide")!;
    expect(decision.input.some((f) => f.value.includes("3 → 1") && f.value.includes("Cải thiện"))).toBe(true);
    expect(decision.input.some((f) => f.value.includes("Measured noise increased"))).toBe(true);
  });

  it("exposes stored measurement and fusion evidence instead of reporting missing data", () => {
    const detail = { ...DETAIL, diagnosis: {
      ...DETAIL.messages[0].content.diagnosis as NonNullable<SessionDetail["diagnosis"]>,
      measurement: { width: 640, height: 480, tile_size: 64, exposure: { mean_l: { value: 12, confidence: .8 } } },
      region_stats: { person: { region: "person", area_fraction: .1, values: { "exposure.mean_l": { value: 9, confidence: .9 } }, mask_source: "grabcut", mask_confidence: .7 } },
      fusion_notes: ["Measured exposure confirms the hypothesis"],
    } };
    const step = buildFlowSteps(detail, "analysis").find((s) => s.id === "evidence")!;
    expect(step.rationale).toContain("Measured exposure confirms the hypothesis");
    expect(step.output.some((f) => f.label.includes("person") && f.value.includes("9"))).toBe(true);
  });

  it("keeps the input and edit chips locked while an analysis note is pending", async () => {
    const fetchMock = mockApi([...base(WITH_EDIT), { method: "POST", path: `${PATH}/chat`, pending: true }]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const input = await screen.findByPlaceholderText(NOTE_PLACEHOLDER);
    await user.type(input, "mặt còn tối");
    await user.click(screen.getByRole("button", { name: "Gửi bổ sung phân tích" }));
    expect(screen.getByRole("button", { name: /Xóa chỉnh sửa:/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Xử lý ảnh" })).toBeDisabled();
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/chat"))).toHaveLength(1);
  });

  it("reloads the overlay when a saved analysis changes but its artifact URL stays the same", async () => {
    mockApi([...base(WITH_EDIT), { method: "POST", path: `${PATH}/chat`, body: { ...WITH_EDIT, updated_at: "2026-10-27T09:00:00.123456+00:00" } }]);
    // Màn hình rộng: bảng soi nằm cạnh khung chat, không phải drawer modal
    const original = window.matchMedia;
    window.matchMedia = ((query: string) => ({ ...original(query), matches: true })) as typeof window.matchMedia;
    onTestFinished(() => { window.matchMedia = original; });
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Xem chi tiết chẩn đoán" }));
    const image = await screen.findByRole("img", { name: "Ảnh gốc với vùng được tô màu" });
    const before = image.getAttribute("src");
    await user.type(screen.getByPlaceholderText(NOTE_PLACEHOLDER), "mặt còn tối");
    await user.click(screen.getByRole("button", { name: "Gửi bổ sung phân tích" }));
    await waitFor(() => expect(image.getAttribute("src")).not.toBe(before));
  });

  it("labels rejected attempts in the execution tab of the side pane", async () => {
    const detail = { ...DONE, artifacts: [...DONE.artifacts, { ...ITER_ARTIFACT, id: "attempt2", url: "/attempt2.png" }],
      process: { ...DONE.process!, history: [ { ...tracked, rolled_back: true }, { ...tracked, iteration: 2 } ],
        artifacts: { ...DONE.process!.artifacts, intermediate: [ITER_ARTIFACT.id, "attempt2"] } } };
    mockApi(base(detail));
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Xem thông số thực thi" }));
    const pane = await screen.findByRole("region", { name: "Bảng soi kỹ thuật" });
    expect(within(pane).getByText(/Vòng này đã bị rollback/)).toBeInTheDocument();
    expect(within(pane).getAllByText(/Mức 3 → 1/).length).toBeGreaterThan(0);
  });
});
