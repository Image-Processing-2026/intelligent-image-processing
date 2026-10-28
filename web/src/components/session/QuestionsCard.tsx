import { Loader2, Play } from "lucide-react";

import type { Question } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

export interface AnswerDraft {
  answers: Record<string, string>;
}

/** Lựa chọn đang có hiệu lực của một câu hỏi: bản nháp nếu hợp lệ, ngược lại mặc định. */
export function selectedValue(question: Question, draft: AnswerDraft): string {
  const value = draft.answers[question.id];
  return question.choices.some((choice) => choice.value === value) ? value : question.default;
}

interface QuestionsCardProps {
  questions: Question[];
  /** Chỉ bộ câu hỏi hiện hành của phiên đang chờ trả lời mới tương tác được. */
  active: boolean;
  draft: AnswerDraft;
  onDraftChange: (draft: AnswerDraft) => void;
  onSubmit: () => void;
  pending: boolean;
  /** Câu trả lời đã gửi cho bộ câu hỏi này (từ message 'answers'), nếu có. */
  answeredWith?: Record<string, string>;
}

export function QuestionsCard({
  questions,
  active,
  draft,
  onDraftChange,
  onSubmit,
  pending,
  answeredWith,
}: QuestionsCardProps) {
  const disabled = !active || pending;
  // Không có câu hỏi nào (chẩn đoán không mơ hồ): chỉ còn nút bắt đầu xử lý
  if (questions.length === 0 && !active) return null;
  // Bộ câu hỏi đã trả lời hiển thị đúng lựa chọn đã gửi, không phải bản nháp hiện tại
  const shown: AnswerDraft = answeredWith ? { answers: answeredWith } : draft;
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <h2 className="text-base">
            {questions.length > 0 ? "Câu hỏi về ý định của bạn" : "Sẵn sàng xử lý"}
          </h2>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form
          className="grid gap-5 text-sm"
          onSubmit={(event) => {
            event.preventDefault();
            if (!disabled) onSubmit();
          }}
        >
          {questions.map((question) => {
            const labelId = `q-${question.id}`;
            return (
              <div key={question.id} className="grid gap-2">
                <p id={labelId} className="font-medium">
                  {question.text}
                </p>
                <RadioGroup
                  aria-labelledby={labelId}
                  value={selectedValue(question, shown)}
                  onValueChange={(value) =>
                    onDraftChange({ ...draft, answers: { ...draft.answers, [question.id]: value } })
                  }
                  disabled={disabled}
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

          {active ? (
            <div>
              <Button type="submit" disabled={disabled}>
                {pending ? (
                  <Loader2 className="animate-spin" aria-hidden />
                ) : (
                  <Play aria-hidden />
                )}
                Bắt đầu xử lý
              </Button>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              {answeredWith
                ? "Đã trả lời — xem câu trả lời bên dưới."
                : "Bộ câu hỏi này không còn dùng được."}
            </p>
          )}
        </form>
      </CardContent>
    </Card>
  );
}
