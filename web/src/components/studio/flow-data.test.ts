import { describe, expect, it } from "vitest";
import { DETAIL, DONE, ITER_ARTIFACT } from "@/test/fixtures";
import { buildFlowSteps, latestDiagnosis } from "./flow-data";

describe("saved flow log", () => {
  it("recovers the initial diagnosis from messages in older sessions", () => {
    expect(latestDiagnosis(DETAIL)?.summary).toBe("Ảnh trong nhà thiếu sáng");
    const steps = buildFlowSteps(DETAIL, "analysis");
    expect(steps.find((s) => s.id === "diagnosis")?.output[0].value).toContain("mean 40");
    expect(steps.find((s) => s.id === "evidence")?.output).toEqual([]);
    expect(steps.find((s) => s.id === "evidence")?.rationale).toBeNull();
  });

  it("uses the backend plan rationale, source and parameters without inventing tool success", () => {
    const steps = buildFlowSteps(DONE, "processing");
    expect(steps.find((s) => s.id === "plan")?.rationale).toBe("Ảnh tối, tăng gamma.");
    expect(steps.find((s) => s.id === "plan")?.actor).toBe("Luật (rule_based)");
    expect(steps.find((s) => s.id === "plan")?.output[0].value).toContain("gamma=1.4");
    expect(steps.find((s) => s.id === "execute")?.rationale).toBeNull();
    expect(steps.find((s) => s.id === "execute")?.image).toEqual(ITER_ARTIFACT);
    expect(steps.find((s) => s.id === "evaluate")?.output).toContainEqual({ label: "Độ sáng trung bình", value: "92,1" });
  });

  it("does not create rounds or pending steps that were never recorded", () => {
    expect(buildFlowSteps(DETAIL, "processing")).toEqual([]);
    expect(buildFlowSteps(DONE, "processing", 20)).toEqual([]);
    const noImage = { ...DONE, process: { ...DONE.process!, artifacts: { final: "missing", intermediate: [] } } };
    expect(buildFlowSteps(noImage, "processing").find((s) => s.id === "execute")?.image).toBeNull();
  });

  it("keeps final decisions and estimated metrics separate from user ratings", () => {
    const step = buildFlowSteps(DONE, "result")[0];
    expect(step.output).toContainEqual({ label: "Điểm ước lượng", value: "71,25" });
    expect(step.output[0].value).toBe("Đạt ngưỡng đánh giá của hệ thống");
  });
});
