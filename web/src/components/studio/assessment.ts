import type { DiagnosisReport, HistoryItem, ProcessResult, SessionDetail } from "@/api/types";
import { DEFECT_LABELS, formatNumber, formatPercent, label, regionLabel } from "@/lib/labels";

export const DEFECT_STATUS_LABELS: Record<string, string> = {
  resolved: "Đã khắc phục", improved: "Cải thiện", unchanged: "Không đổi", worse: "Nặng hơn",
};
export const SIDE_EFFECT_LABELS: Record<string, string> = {
  halo: "Viền halo", dark_noise: "Nhiễu vùng tối tăng", banding: "Phân dải màu",
  channel_clipping: "Cháy kênh màu", oversaturation: "Màu quá bão hòa",
};
export const MEASUREMENT_LABELS: Record<string, string> = {
  "exposure.mean_l": "Độ sáng L*", "exposure.clip_high": "Tỉ lệ cháy sáng",
  "exposure.clip_low": "Tỉ lệ bệt đen", "noise.sigma": "Ước lượng nhiễu σ",
};

/** Các trường bằng chứng do backend ghi, không tái thực hiện logic decide ở trình duyệt. */
export function assessmentFields(round: HistoryItem) {
  return [
    ...(round.defect_status ?? []).map((d) => ({
      label: `${label(DEFECT_LABELS, d.type)} · ${regionLabel(d.region)} (so với ảnh gốc)`,
      value: `Mức ${d.before} → ${d.after} · ${label(DEFECT_STATUS_LABELS, d.status)}`,
    })),
    ...(round.side_effects ?? []).map((e) => ({
      label: `Tác dụng phụ: ${label(SIDE_EFFECT_LABELS, e.kind)} · ${regionLabel(e.region)}`,
      value: `Mức ${e.severity}${e.before != null && e.after != null ? ` · ${formatNumber(e.before)} → ${formatNumber(e.after)}` : ""}${e.evidence ? ` · ${e.evidence}` : ""}`,
    })),
  ];
}

export function measurementFields(diagnosis: DiagnosisReport | null) {
  if (!diagnosis) return [];
  return [
    ...Object.entries(diagnosis.region_stats ?? {}).flatMap(([name, stats]) => [
      { label: `Mặt nạ · ${regionLabel(name)}`, value: `${stats.mask_source} · tin cậy ${formatPercent(stats.mask_confidence, 0)}` },
      ...Object.entries(stats.values).map(([key, estimate]) => ({
        label: `${MEASUREMENT_LABELS[key] ?? key} · ${regionLabel(name)}`,
        value: `${formatNumber(estimate.value, 3)} · tin cậy ${formatPercent(estimate.confidence, 0)}`,
      })),
    ]),
    ...(["exposure", "noise", "sharpness", "color", "compression", "haze"] as const).flatMap((group) =>
      Object.entries(diagnosis.measurement?.[group] ?? {}).map(([key, estimate]) => ({
        label: `${MEASUREMENT_LABELS[`${group}.${key}`] ?? `${group}.${key}`} · Toàn ảnh`,
        value: `${formatNumber(estimate.value, 3)} · tin cậy ${formatPercent(estimate.confidence, 0)}`,
      }))),
    ...(diagnosis.measurement?.notes ?? []).map((note) => ({ label: "Ghi chú phép đo", value: note })),
  ];
}

/** Ảnh đầu vòng là kết quả được giữ lại gần nhất; bỏ qua các ảnh thử đã rollback. */
export function roundImages(detail: SessionDetail, index: number) {
  const process = detail.process;
  const artifacts = new Map(detail.artifacts.map((a) => [a.id, a]));
  const after = artifacts.get(process?.artifacts.intermediate[index] ?? "") ?? null;
  for (let previous = index - 1; previous >= 0; previous--) {
    if (!process?.history[previous]?.rolled_back) {
      return { before: artifacts.get(process?.artifacts.intermediate[previous] ?? "") ?? null, after };
    }
  }
  return { before: detail.original, after };
}

/** final_evaluation vẫn mô tả ảnh thử khi rollback; chỉ dùng đánh giá của ảnh được giữ. */
export function retainedEvaluation(process: ProcessResult): Record<string, unknown> | null {
  if (!process.rolled_back) return process.final_evaluation;
  for (let index = process.history.length - 2; index >= 0; index--) {
    if (!process.history[index].rolled_back) return process.history[index].eval_score;
  }
  return null;
}
