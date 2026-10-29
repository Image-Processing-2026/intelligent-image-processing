import type { DiagnosisEdit } from "@/api/types";

import { DEFECT_LABELS, PRESERVE_LABELS, label, regionLabel } from "./labels";

/** Mô tả ngắn một chỉnh sửa của người dùng, dùng cho chip và ghi chú trong hội thoại. */
export function describeEdit(edit: DiagnosisEdit): string {
  const where = regionLabel(edit.region);
  if (edit.op === "add_defect") {
    return `Thêm: ${label(DEFECT_LABELS, edit.type)} @ ${where} (mức ${edit.severity})`;
  }
  if (edit.op === "remove_defect") {
    return `Bỏ: ${label(DEFECT_LABELS, edit.type)} @ ${where}`;
  }
  return `Giữ: ${label(PRESERVE_LABELS, edit.aspect)} @ ${where}`;
}
