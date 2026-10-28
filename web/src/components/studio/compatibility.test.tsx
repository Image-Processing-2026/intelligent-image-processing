import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { DETAIL, DONE, ITER_ARTIFACT, SESSION_ID, SUMMARY, WITH_EDIT } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";
import { buildFlowSteps } from "./flow-data";
import { retainedEvaluation } from "./assessment";
import { RoundAssessment } from "./RoundAssessment";
import type { SessionDetail } from "@/api/types";

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

  it("keeps pending analysis mutations locked after switching phases", async () => {
    const fetchMock = mockApi([...base(WITH_EDIT), { method: "POST", path: `${PATH}/analysis-notes`, pending: true }]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.type(await screen.findByLabelText("Tin nhắn bổ sung"), "mặt còn tối");
    await user.click(screen.getByRole("button", { name: "Gửi bổ sung" }));
    await user.click(screen.getByRole("button", { name: "02 Xử lý" }));
    await user.click(screen.getByRole("button", { name: "01 Phân tích" }));
    expect(screen.getByLabelText("Tin nhắn bổ sung")).toBeDisabled();
    expect(screen.getByRole("button", { name: /Xóa chỉnh sửa:/ })).toBeDisabled();
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/analysis-notes"))).toHaveLength(1);
  });

  it("reloads the overlay when a saved analysis changes but its artifact URL stays the same", async () => {
    mockApi([...base(WITH_EDIT), { method: "POST", path: `${PATH}/analysis-notes`, body: { ...WITH_EDIT, updated_at: "2026-10-27T09:00:00.123456+00:00" } }]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    const image = await screen.findByRole("img", { name: "Ảnh gốc với vùng được tô màu" });
    const before = image.getAttribute("src");
    await user.type(screen.getByLabelText("Tin nhắn bổ sung"), "mặt còn tối");
    await user.click(screen.getByRole("button", { name: "Gửi bổ sung" }));
    await waitFor(() => expect(image.getAttribute("src")).not.toBe(before));
  });

  it("restores the region overlay when selecting a region after showing the original", async () => {
    mockApi(base(WITH_EDIT));
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Hiện ảnh gốc" }));
    await user.click(screen.getAllByRole("button", { name: "Xem vùng" })[0]);
    expect(screen.getByRole("img", { name: "Ảnh gốc với vùng được tô màu" })).toBeInTheDocument();
  });

  it("labels rejected attempts and uses the retained image as the next round input", async () => {
    const detail = { ...DONE, artifacts: [...DONE.artifacts, { ...ITER_ARTIFACT, id: "attempt2", url: "/attempt2.png" }],
      process: { ...DONE.process!, history: [ { ...tracked, rolled_back: true }, { ...tracked, iteration: 2 } ],
        artifacts: { ...DONE.process!.artifacts, intermediate: [ITER_ARTIFACT.id, "attempt2"] } } };
    mockApi(base(detail));
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "02 Xử lý" }));
    expect(screen.getByText(/Ảnh thử của vòng 1 đã bị loại bỏ/)).toBeInTheDocument();
    await user.click(within(screen.getByRole("complementary", { name: "Flow log" })).getByRole("button", { name: "Vòng 2" }));
    expect(screen.getByRole("img", { name: "Đầu vào vòng 2" })).toHaveAttribute("src", DONE.original!.url);
  });
});
