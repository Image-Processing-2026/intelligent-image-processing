import { Loader2, Send, X } from "lucide-react";
import { useState } from "react";

import { errorMessage } from "@/api/client";
import type { DiagnosisEdit } from "@/api/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { describeEdit } from "@/lib/edits";

export const ANALYSIS_NOTE_MAX = 1000;

interface AnalysisChatProps {
  edits: DiagnosisEdit[];
  /** Gửi một tin nhắn; reject khi lỗi để khung chat giữ nguyên nội dung đang gõ. */
  onSend: (message: string) => Promise<unknown>;
  onDeleteEdit: (index: number) => void;
  disabled: boolean;
}

/** Khung chat "Bổ sung phân tích": thêm/bỏ lỗi, đánh dấu điều cần giữ trước khi xử lý. */
export function AnalysisChat({ edits, onSend, onDeleteEdit, disabled }: AnalysisChatProps) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const message = text.trim();
  const locked = disabled || sending;

  const submit = async () => {
    if (!message || locked) return;
    setSending(true);
    setError(null);
    try {
      await onSend(message);
      setText("");
    } catch (caught) {
      // Giữ nguyên nội dung đang gõ để người dùng gửi lại
      setError(errorMessage(caught));
    } finally {
      setSending(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <h2 className="text-base">Bổ sung phân tích</h2>
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3 text-sm">
        <p className="text-muted-foreground">
          Hệ thống có thể bỏ sót hoặc nhận nhầm. Nói rõ vùng và vấn đề để chỉnh bản chẩn đoán trước
          khi xử lý.
        </p>

        {edits.length > 0 ? (
          <ul className="flex flex-wrap gap-2" aria-label="Các chỉnh sửa của bạn">
            {edits.map((edit, index) => {
              const summary = describeEdit(edit);
              return (
                <li key={`${index}-${summary}`}>
                  <Badge variant="outline" className="gap-1 py-1 pr-1 font-normal">
                    {summary}
                    <button
                      type="button"
                      onClick={() => onDeleteEdit(index)}
                      disabled={locked}
                      aria-label={`Xóa chỉnh sửa: ${summary}`}
                      className="rounded-full p-0.5 hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50"
                    >
                      <X className="size-3" aria-hidden />
                    </button>
                  </Badge>
                </li>
              );
            })}
          </ul>
        ) : null}

        <form
          className="grid gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <Label htmlFor="analysis-note">Tin nhắn bổ sung</Label>
          <Textarea
            id="analysis-note"
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength={ANALYSIS_NOTE_MAX}
            placeholder="Ví dụ: mặt còn tối, nền cháy là cố ý, giữ tông ấm"
            disabled={locked}
            rows={2}
          />
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs text-muted-foreground">
              {text.length}/{ANALYSIS_NOTE_MAX} ký tự
            </p>
            <Button type="submit" size="sm" variant="outline" disabled={locked || !message}>
              {sending ? (
                <Loader2 className="animate-spin" aria-hidden />
              ) : (
                <Send aria-hidden />
              )}
              Gửi bổ sung
            </Button>
          </div>
        </form>

        {error ? (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}
      </CardContent>
    </Card>
  );
}
