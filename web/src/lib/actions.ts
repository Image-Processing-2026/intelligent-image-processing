// Mô tả phác đồ cho người đọc: tên thao tác tiếng Việt + vùng + tham số (giá trị thật, không làm tròn ý nghĩa).
import type { RegionAction, Treatment, Variant } from "@/api/types";

import { label, regionLabel } from "./labels";

export const OPERATION_LABELS: Record<string, string> = {
  denoise: "Khử nhiễu",
  gamma_correct: "Chỉnh gamma",
  clahe: "Cân bằng tương phản (CLAHE)",
  sharpen: "Làm nét",
  color_correct: "Cân bằng màu",
};

export function operationLabel(operation: string): string {
  return OPERATION_LABELS[operation] ?? operation;
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (Array.isArray(value)) return `[${value.map(formatValue).join(", ")}]`;
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function describeParameters(parameters: Record<string, unknown>): string {
  return Object.entries(parameters ?? {})
    .map(([key, value]) => `${key}=${formatValue(value)}`)
    .join(", ");
}

/** Ví dụ: "Chỉnh gamma @ Toàn ảnh (gamma=1.4)". */
export function describeAction(action: RegionAction): string {
  const params = describeParameters(action.parameters);
  const region = regionLabel(action.target_prompt || action.region_id || "full");
  return `${operationLabel(action.operation)} @ ${region}${params ? ` (${params})` : ""}`;
}

export const RECOMMENDED_REASON_LABELS: Record<string, string> = {
  intent: "theo ý định của bạn",
  preference: "theo lựa chọn trước đây của bạn",
  critic: "Gemini nhận xét",
  score: "theo điểm ước lượng",
};

export const RANKING_SOURCE_LABELS: Record<string, string> = {
  critic: "Xếp hạng bởi Gemini (critic)",
  score: "Xếp hạng theo điểm ước lượng",
};

export const TREATMENT_SOURCE_LABELS: Record<string, string> = {
  pipeline: "Kết quả pipeline",
  variant: "Phiên bản đã chọn",
  refine: "Tinh chỉnh theo góp ý",
};

export const REFINE_SOURCE_LABELS: Record<string, string> = {
  rules: "hiểu bằng luật",
  vlm: "hiểu bằng Gemini",
  none: "không hiểu góp ý — giữ nguyên phác đồ",
};

export const ADJUSTMENT_LABELS: Record<string, string> = {
  brighter: "Sáng hơn",
  darker: "Tối hơn",
  more_contrast: "Tăng tương phản",
  less_contrast: "Giảm tương phản",
  warmer: "Ấm hơn",
  cooler: "Lạnh hơn (bớt vàng)",
  more_saturated: "Màu rực hơn",
  less_saturated: "Màu nhạt hơn",
  sharper: "Nét hơn",
  softer: "Mềm hơn",
  less_noise: "Bớt nhiễu",
  less_denoise: "Bớt làm mịn",
};

/** Nhãn ngắn của một phiên bản phác đồ, ví dụ "v2 · Phiên bản đã chọn (Tự nhiên)". */
export function treatmentLabel(treatment: Treatment, variants: Variant[]): string {
  const variant = variants.find((item) => item.id === treatment.variant_id);
  const base = `v${treatment.version} · ${label(TREATMENT_SOURCE_LABELS, treatment.source)}`;
  return variant ? `${base} (${variant.label})` : base;
}
