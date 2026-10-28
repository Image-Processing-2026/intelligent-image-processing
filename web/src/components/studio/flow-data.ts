import type { Artifact, DiagnosisEdit, DiagnosisReport, SessionDetail } from "@/api/types";
import type { StudioPhase } from "@/components/studio/PhaseNav";
import { describeAction } from "@/lib/actions";
import { describeEdit } from "@/lib/edits";
import { DECISION_LABELS, DEFECT_LABELS, SOURCE_LABELS, formatNumber, label, regionLabel } from "@/lib/labels";
import { assessmentFields, measurementFields, retainedEvaluation } from "./assessment";

export interface FlowField { label: string; value: string }
export interface FlowStep {
  id: string;
  title: string;
  actor: string;
  input: FlowField[];
  rationale: string | null;
  output: FlowField[];
  image?: Artifact | null;
}

export function latestDiagnosis(detail: SessionDetail): DiagnosisReport | null {
  const messages = detail.messages.filter((m) => m.kind === "diagnosis");
  return detail.draft_diagnosis ?? detail.diagnosis ??
    (messages[messages.length - 1]?.content.diagnosis as DiagnosisReport | null) ?? null;
}

function defects(diagnosis: DiagnosisReport | null): FlowField[] {
  return diagnosis?.defects.map((d) => ({
    label: `${label(DEFECT_LABELS, d.type)} · ${regionLabel(d.region)}`,
    value: `Mức ${d.severity}${d.evidence ? ` · ${d.evidence}` : ""}`,
  })) ?? [];
}

function metrics(values: Record<string, unknown>): FlowField[] {
  const names: Record<string, string> = {
    brightness_mean: "Độ sáng trung bình", contrast_std: "Tương phản",
    noise_variance: "Phương sai nhiễu", sharpness_laplacian_var: "Độ nét (Laplacian)",
    estimated_quality_score: "Điểm ước lượng", psnr: "PSNR (dB)", ssim: "SSIM",
    brisque_score: "BRISQUE", niqe_score: "NIQE",
  };
  return Object.entries(names).flatMap(([key, name]) => typeof values[key] === "number"
    ? [{ label: name, value: formatNumber(values[key] as number, key === "ssim" ? 4 : 2) }] : []);
}

/** Ánh xạ bản ghi thật sang log; không suy ra việc thực thi công cụ chỉ từ phác đồ. */
export function buildFlowSteps(detail: SessionDetail, phase: StudioPhase, roundIndex = 0): FlowStep[] {
  const diagnosis = latestDiagnosis(detail);
  const original = detail.original;
  const imageInput = original ? [{ label: "Ảnh gốc", value: `${original.width} × ${original.height} px · RGB` }] : [];
  const answers = detail.messages.filter((m) => m.kind === "answers").at(-1);
  const answerFields = ((answers?.content.labels ?? []) as { question: string; answer: string }[])
    .map((a) => ({ label: a.question, value: a.answer }));
  const edits = (answers?.content.edits ?? detail.draft_edits ?? []) as DiagnosisEdit[];
  const intent = [...answerFields,
    ...edits.map((edit) => ({ label: "Bổ sung phân tích", value: describeEdit(edit) })),
    ...(answers?.content.notes ? [{ label: "Ghi chú", value: String(answers.content.notes) }] : []),
  ];
  if (phase === "analysis") return [
    { id: "input", title: "Tiếp nhận ảnh", actor: "Ảnh đầu vào", input: imageInput, rationale: null,
      output: [{ label: "Ảnh sử dụng", value: "Ảnh gốc đã lưu trong phiên" }], image: original },
    { id: "diagnosis", title: "Chẩn đoán ảnh", actor: diagnosis ? label(SOURCE_LABELS, diagnosis.source) : "Chưa ghi nhận",
      input: imageInput, rationale: diagnosis?.summary || null, output: defects(diagnosis) },
    { id: "evidence", title: "Bằng chứng & vùng ảnh", actor: "Số đo đã lưu", input: defects(diagnosis),
      rationale: diagnosis?.fusion_notes?.join("\n") || null, output: [...Object.values(diagnosis?.region_metrics ?? {}).flatMap((r) => [
        { label: `Độ sáng · ${regionLabel(r.region)}`, value: formatNumber(r.brightness_mean, 1) },
        { label: `Backend vùng · ${regionLabel(r.region)}`, value: r.backend },
      ]), ...measurementFields(diagnosis)] },
    { id: "intent", title: "Xác nhận ý định", actor: "Người dùng", input: defects(diagnosis),
      rationale: null, output: intent },
  ];
  const process = detail.process;
  if (!process) return [];
  const artifacts = new Map(detail.artifacts.map((a) => [a.id, a]));
  if (phase === "result") return [
    { id: "summary", title: "Tổng kết xử lý", actor: "Điều phối", input: [{ label: "Số vòng đã chạy", value: String(process.total_iterations) }],
      rationale: null, output: [{ label: "Quyết định cuối", value: label(DECISION_LABELS, process.final_decision) },
        ...metrics(retainedEvaluation(process) ?? {}),
        ...(process.rolled_back ? [{ label: "Rollback", value: "Ảnh thử cuối đã bị loại. Điểm của ảnh thử không được dùng làm điểm ảnh đang giữ." }] : [])] },
    { id: "variants", title: "Các phiên bản", actor: "Phác đồ & phong cách", input: process.treatment.map((a, i) => ({ label: `Bước ${i + 1}`, value: describeAction(a) })),
      rationale: null, output: detail.variants.map((v) => ({ label: v.label, value: `${v.recommended ? "Đề xuất · " : ""}Điểm ước lượng: ${formatNumber(v.quality_score, 1)}` })) },
    { id: "versions", title: "Lịch sử tinh chỉnh", actor: "Người dùng & hệ thống", input: [{ label: "Số phiên bản đã lưu", value: String(detail.treatments.length) }],
      rationale: null, output: detail.treatments.map((t) => ({ label: `v${t.version}${t.id === detail.active_treatment_id ? " · Đang dùng" : ""}`, value: t.feedback_text || (t.source === "pipeline" ? "Kết quả pipeline" : "Phiên bản đã render") })) },
  ];
  const round = process.history[roundIndex];
  if (!round) return [];
  const actions = round.plan.actions.map((a, i) => ({ label: `Bước ${i + 1}`, value: describeAction(a) }));
  const outputImage = artifacts.get(process.artifacts.intermediate[roundIndex]) ?? null;
  return [
    { id: "measure", title: "Phân tích đầu vòng", actor: round.diagnosis ? label(SOURCE_LABELS, round.diagnosis.source) : "Analyzer",
      input: metrics(round.metrics_before), rationale: round.diagnosis?.summary || null,
      output: [...defects(round.diagnosis), ...(round.diagnosis?.diff ? assessmentFields({ ...round, defect_status: round.diagnosis.diff.defects, side_effects: round.diagnosis.diff.side_effects }) : [])] },
    { id: "plan", title: "Lập phác đồ", actor: label(SOURCE_LABELS, round.plan.source),
      input: [...defects(round.diagnosis), ...intent], rationale: round.plan.reasoning || null,
      output: [...actions, ...round.plan.knowledge.map((k) => ({ label: "Thẻ tri thức", value: k }))] },
    { id: "execute", title: "Xử lý ảnh", actor: "Engine · Thuật toán cổ điển", input: actions,
      rationale: null, output: outputImage ? [{ label: round.rolled_back ? "Ảnh thử đã bị loại" : "Ảnh sau vòng", value: `${outputImage.width} × ${outputImage.height} px` }]
        : [{ label: "Ảnh trung gian", value: "Không có ảnh được lưu cho vòng này" }], image: outputImage },
    { id: "evaluate", title: "Đánh giá thay đổi", actor: "Evaluator", input: metrics(round.metrics_before),
      rationale: null, output: [...assessmentFields(round), ...metrics(round.metrics_after), ...metrics(round.eval_score)] },
    { id: "decide", title: "Quyết định vòng tiếp", actor: "Điều phối", input: [...assessmentFields(round), ...metrics(round.eval_score)],
      rationale: null, output: [{ label: "Quyết định", value: label(DECISION_LABELS, round.decision) },
        ...(round.rolled_back ? [{ label: "Rollback", value: "Đã quay lại kết quả trước" }] : [])] },
  ];
}
