import { BarChart3, X } from "lucide-react";

import type { DiagnosisEdit, DiagnosisReport, Question } from "@/api/types";
import { SeverityBadge } from "@/components/session/DiagnosisCard";
import { type AnswerDraft, selectedValue } from "@/components/session/QuestionsCard";
import { SourceBadge } from "@/components/session/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { describeEdit } from "@/lib/edits";
import { DEFECT_LABELS, PRESERVE_LABELS, label, regionLabel } from "@/lib/labels";

/** Gợi ý nhanh: bấm là gửi như một tin nhắn bổ sung phân tích. */
export const QUICK_SUGGESTIONS = [
  "Làm sáng khuôn mặt",
  "Giữ nguyên tông màu ấm",
  "Không làm nền cháy sáng",
] as const;

interface DiagnosisBubbleProps {
  diagnosis: DiagnosisReport | null;
  /** Chẩn đoán hiện hành đang chờ trả lời: có câu hỏi, chip chỉnh sửa, gợi ý và nút xử lý. */
  live: boolean;
  questions: Question[];
  draft: AnswerDraft;
  onDraftChange: (draft: AnswerDraft) => void;
  edits: DiagnosisEdit[];
  /** Đã có yêu cầu riêng của người dùng (chip chỉnh sửa hoặc ghi chú) để xử lý "cách khác". */
  customReady: boolean;
  onRemoveEdit: (index: number) => void;
  onQuickNote: (text: string) => void;
  onStart: () => void;
  onOpenDetails: () => void;
  pending: boolean;
}

/** Bong bóng chẩn đoán: tóm tắt tự nhiên + câu hỏi làm rõ + chip chỉnh sửa + hành động. */
export function DiagnosisBubble({
  diagnosis,
  live,
  questions,
  draft,
  onDraftChange,
  edits,
  customReady,
  onRemoveEdit,
  onQuickNote,
  onOpenDetails,
  pending,
}: DiagnosisBubbleProps) {
  // Chọn "cách khác" nhưng chưa mô tả gì: chưa có gì để xử lý
  const needsCustom =
    !customReady &&
    questions.some((q) => q.id === "approach" && selectedValue(q, draft) === "custom");
  const defects = [...(diagnosis?.defects ?? [])].sort((a, b) => b.severity - a.severity);
  return (
    <div className="grid gap-3 rounded-lg border bg-card px-4 py-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold">Chẩn đoán ảnh</h2>
        {diagnosis ? <SourceBadge prefix="Chẩn đoán" source={diagnosis.source} /> : null}
      </div>

      {diagnosis ? (
        <>
          <p>{diagnosis.summary || "Đã phân tích xong ảnh."}</p>
          {defects.length === 0 ? (
            <p className="text-muted-foreground">Không phát hiện lỗi cần xử lý.</p>
          ) : (
            <ul className="grid gap-1.5" aria-label="Lỗi phát hiện">
              {defects.map((defect, index) => (
                <li
                  key={`${defect.type}-${defect.region}-${index}`}
                  className="flex flex-wrap items-center gap-2"
                >
                  <span className="font-medium">{label(DEFECT_LABELS, defect.type)}</span>
                  <span className="text-muted-foreground">@ {regionLabel(defect.region)}</span>
                  <SeverityBadge severity={defect.severity} />
                  {defect.origin === "user" ? (<Badge className="font-normal">Bạn bổ sung</Badge>) : null}
                </li>
              ))}
            </ul>
          )}
          {diagnosis.preserve.length > 0 ? (
            <p className="text-muted-foreground">
              Sẽ giữ nguyên:{" "}
              {diagnosis.preserve
                .map(
                  (item) => `${label(PRESERVE_LABELS, item.aspect)} @ ${regionLabel(item.region)}`,
                )
                .join("; ")}
            </p>
          ) : null}
        </>
      ) : (
        <p className="text-muted-foreground">Không có dữ liệu chẩn đoán.</p>
      )}

      {live && edits.length > 0 ? (
        <ul className="flex flex-wrap gap-1.5" aria-label="Chỉnh sửa của bạn">
          {edits.map((edit, index) => {
            const text = describeEdit(edit);
            return (
              <li
                key={`${text}-${index}`}
                className="flex items-center gap-1 rounded-full border bg-accent py-0.5 pr-1 pl-2.5 text-xs"
              >
                <span>{text}</span>
                <button
                  type="button"
                  aria-label={`Xóa chỉnh sửa: ${text}`}
                  disabled={pending}
                  onClick={() => onRemoveEdit(index)}
                  className="rounded-full p-0.5 hover:bg-background disabled:opacity-50"
                >
                  <X className="size-3" aria-hidden />
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}

      {live && questions.length > 0 ? (
        <div className="grid gap-4">
          {questions.map((question) => {
            const labelId = `q-${question.id}`;
            return (
              <div key={question.id} className="grid gap-2">
                <p id={labelId} className="font-medium">
                  {question.text}
                </p>
                <RadioGroup
                  aria-labelledby={labelId}
                  value={selectedValue(question, draft)}
                  onValueChange={(value) =>
                    onDraftChange({ ...draft, answers: { ...draft.answers, [question.id]: value } })
                  }
                  disabled={pending}
                  className="gap-2"
                >
                  {question.choices.map((choice) => {
                    const id = `q-${question.id}-${choice.value}`;
                    return (
                      <div key={choice.value} className="flex items-center gap-2">
                        <RadioGroupItem id={id} value={choice.value} />
                        <Label htmlFor={id} className="font-normal">
                          {choice.label}
                          {choice.value === question.default ? (
                            <span className="text-muted-foreground"> (mặc định)</span>
                          ) : null}
                        </Label>
                      </div>
                    );
                  })}
                </RadioGroup>
              </div>
            );
          })}
        </div>
      ) : null}

      {live ? (
        <>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Gợi ý nhanh">
            {QUICK_SUGGESTIONS.map((suggestion) => (
              <Button
                key={suggestion}
                type="button"
                size="sm"
                variant="outline"
                className="h-7 rounded-full px-3 text-xs font-normal"
                disabled={pending}
                onClick={() => onQuickNote(suggestion)}
              >
                {suggestion}
              </Button>
            ))}
          </div>
          {needsCustom ? (
            <p className="text-xs text-muted-foreground" role="status">
              Hãy mô tả cách bạn muốn xử lý ở khung chat bên dưới (hoặc bấm một gợi ý nhanh).
            </p>
          ) : null}
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="outline" onClick={onOpenDetails}>
              <BarChart3 aria-hidden />
              Xem chi tiết chẩn đoán
            </Button>
          </div>
        </>
      ) : diagnosis ? (
        <div>
          <Button type="button" size="sm" variant="ghost" onClick={onOpenDetails}>
            <BarChart3 aria-hidden />
            Xem chi tiết chẩn đoán
          </Button>
        </div>
      ) : null}
    </div>
  );
}
