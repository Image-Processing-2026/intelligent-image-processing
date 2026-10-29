// Nhãn tiếng Việt cho mã của backend. Mã lạ được hiển thị nguyên văn (không đoán nghĩa).
import type { PlanSource, SessionStatus } from "@/api/types";

export const STATUS_LABELS: Record<SessionStatus, string> = {
  uploaded: "Đã tải lên",
  awaiting_answers: "Chờ trả lời",
  running: "Đang xử lý",
  done: "Hoàn tất",
  failed: "Lỗi",
};

// Trung thực theo spec §5.5: không gọi SHIP là "tối ưu" hay "chính xác"
export const DECISION_LABELS: Record<string, string> = {
  SHIP: "Đạt ngưỡng đánh giá của hệ thống",
  STOP_BEST_EFFORT: "Đã dừng — xem kết quả và lịch sử",
  RE_PROCESS: "Tiếp tục lặp",
};

export const SOURCE_LABELS: Record<PlanSource, string> = {
  vlm: "Gemini",
  rule_based: "Luật (rule_based)",
  vlm_fallback: "Luật dự phòng (vlm_fallback)",
};

export const SCENE_LABELS: Record<string, string> = {
  portrait: "Chân dung",
  group: "Ảnh nhóm",
  landscape: "Phong cảnh",
  cityscape: "Đô thị",
  night: "Ban đêm",
  indoor: "Trong nhà",
  food: "Món ăn",
  product: "Sản phẩm",
  document: "Tài liệu",
  animal: "Động vật",
  macro: "Cận cảnh (macro)",
  other: "Khác",
};

export const DEFECT_LABELS: Record<string, string> = {
  underexposed: "Thiếu sáng",
  overexposed: "Dư sáng",
  backlit_subject: "Chủ thể ngược sáng",
  low_contrast: "Tương phản thấp",
  noise: "Nhiễu",
  blur: "Mờ",
  color_cast_warm: "Ám vàng/ấm",
  color_cast_cool: "Ám xanh/lạnh",
  color_cast_green: "Ám xanh lá",
  oversaturated: "Màu quá rực",
  undersaturated: "Màu nhạt",
  motion_blur: "Nhòe do chuyển động",
  jpeg_artifacts: "Lỗi nén JPEG",
  haze: "Mờ do sương",
  highlight_clipping: "Mất chi tiết vùng cháy sáng",
};

export const PRESERVE_LABELS: Record<string, string> = {
  warm_tone: "Tông ấm",
  cool_tone: "Tông lạnh",
  low_key: "Không khí tối (low-key)",
  high_key: "Không khí sáng (high-key)",
  silhouette: "Bóng đổ (silhouette)",
  film_grain: "Hạt film",
  soft_focus: "Nét mềm",
  muted_colors: "Màu trầm",
  vivid_colors: "Màu rực",
};

export const ORIGIN_LABELS: Record<string, string> = {
  vlm: "Gemini",
  rule: "Luật từ chỉ số",
  measured: "Đo theo vùng",
  user: "Người dùng yêu cầu",
};

export const SEVERITY_LABELS = ["Không đáng kể", "Nhẹ", "Vừa", "Nặng"];

export const BRIGHTNESS_LEVEL_LABELS: Record<string, string> = {
  underexposed: "Tối",
  normal: "Bình thường",
  overexposed: "Sáng quá",
};

export function label(map: Record<string, string>, code: string | null | undefined): string {
  if (!code) return "—";
  return map[code] ?? code;
}

const REGION_LABELS: Record<string, string> = {
  full: "Toàn ảnh",
  face: "Khuôn mặt",
  sky: "Bầu trời",
  ground: "Mặt đất",
  center: "Trung tâm",
};

export function regionLabel(region: string): string {
  return REGION_LABELS[region] ?? region;
}

const DATE_FORMAT = new Intl.DateTimeFormat("vi-VN", { dateStyle: "short", timeStyle: "short" });

export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : DATE_FORMAT.format(date);
}

const RELATIVE = new Intl.RelativeTimeFormat("vi", { numeric: "auto" });

export function formatRelative(iso: string, now: number = Date.now()): string {
  const date = new Date(iso).getTime();
  if (Number.isNaN(date)) return iso;
  const seconds = Math.round((date - now) / 1000);
  const abs = Math.abs(seconds);
  if (abs < 60) return "vừa xong";
  if (abs < 3600) return RELATIVE.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return RELATIVE.format(Math.round(seconds / 3600), "hour");
  if (abs < 86400 * 7) return RELATIVE.format(Math.round(seconds / 86400), "day");
  return formatDateTime(iso);
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("vi-VN", { maximumFractionDigits: digits });
}

export function formatPercent(ratio: number | null | undefined, digits = 1): string {
  if (ratio === null || ratio === undefined || Number.isNaN(ratio)) return "—";
  return `${(ratio * 100).toLocaleString("vi-VN", { maximumFractionDigits: digits })}%`;
}
