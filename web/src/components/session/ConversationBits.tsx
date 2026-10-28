import { Download, ImageIcon, Info, Loader2, RefreshCw } from "lucide-react";

import type { Adjustment, Artifact, DiagnosisEdit } from "@/api/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ADJUSTMENT_LABELS, REFINE_SOURCE_LABELS } from "@/lib/actions";
import { describeEdit } from "@/lib/edits";
import { formatNumber, label } from "@/lib/labels";

/** Thông báo hệ thống trong hội thoại (thread hết hạn, lỗi lần chạy), kèm hành động nếu có. */
export function NoticeCard({
  text,
  actionLabel,
  onAction,
  pending,
}: {
  text: string;
  actionLabel?: string;
  onAction?: () => void;
  pending?: boolean;
}) {
  return (
    <Alert>
      <Info aria-hidden />
      <AlertDescription className="grid gap-2">
        <span>{text}</span>
        {actionLabel && onAction ? (
          <div>
            <Button size="sm" variant="outline" onClick={onAction} disabled={pending}>
              {pending ? (
                <Loader2 className="animate-spin" aria-hidden />
              ) : (
                <RefreshCw aria-hidden />
              )}
              {actionLabel}
            </Button>
          </div>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}

/** Lượt của người dùng (câu trả lời, góp ý) hiển thị như bong bóng bên phải. */
export function UserBubble({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-lg bg-accent px-4 py-3 text-sm text-accent-foreground">
        <p className="mb-1 text-xs font-medium opacity-80">{title}</p>
        {children}
      </div>
    </div>
  );
}

interface AnswerLabel {
  question: string;
  answer: string;
}

export function AnswersBubble({ labels, notes }: { labels: AnswerLabel[]; notes: string }) {
  return (
    <UserBubble title="Câu trả lời của bạn">
      <ul className="grid gap-1">
        {labels.map((item) => (
          <li key={item.question}>
            <span className="opacity-80">{item.question}</span>
            <br />
            <span className="font-medium">→ {item.answer}</span>
          </li>
        ))}
      </ul>
      {notes ? <p className="mt-2">Ghi chú: {notes}</p> : null}
    </UserBubble>
  );
}

/** Phản hồi của hệ thống cho một tin nhắn bổ sung phân tích: edit đã hiểu hoặc lời hỏi lại. */
export function AnalysisUpdateNote({
  edits,
  unparsed,
}: {
  edits: DiagnosisEdit[];
  unparsed: boolean;
}) {
  return (
    <div className="grid gap-1 rounded-lg border px-4 py-3 text-sm">
      {unparsed ? (
        <p>
          Chưa hiểu, bạn mô tả cụ thể vùng và vấn đề? Tin nhắn vẫn được gửi kèm làm ghi chú khi xử
          lý.
        </p>
      ) : (
        <>
          <p className="text-xs text-muted-foreground">Đã cập nhật bản chẩn đoán nháp:</p>
          <ul className="grid gap-0.5">
            {edits.map((edit, index) => (
              <li key={index}>{describeEdit(edit)}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

/** Ghi nhận một lần render phiên bản ở độ phân giải gốc, kèm nút tải về. */
export function RenderResultNote({
  label,
  width,
  height,
  downloadUrl,
}: {
  label: string;
  width: number;
  height: number;
  downloadUrl: string | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3 text-sm">
      <ImageIcon className="size-4 text-primary" aria-hidden />
      <span>
        Đã render phiên bản <span className="font-medium">{label}</span> ở độ phân giải gốc{" "}
        {width}×{height}.
      </span>
      {downloadUrl ? (
        <Button asChild size="sm" variant="outline" className="ml-auto">
          <a href={downloadUrl} download>
            <Download aria-hidden />
            Tải về
          </a>
        </Button>
      ) : null}
    </div>
  );
}

/** Kết quả một lần tinh chỉnh: phiên bản mới, điều chỉnh đã làm, nguồn hiểu góp ý. */
export function RefineResultNote({
  version,
  baseVersion,
  notes,
  adjustments,
  source,
  qualityScore,
  artifact,
}: {
  version: number;
  baseVersion: number | null;
  notes: string[];
  adjustments: Adjustment[];
  source: string;
  qualityScore: number | null;
  artifact: Artifact | null;
}) {
  const understood = source !== "none";
  return (
    <div className="grid gap-2 rounded-lg border px-4 py-3 text-sm">
      <p>
        Đã tạo phiên bản <span className="font-medium">v{version}</span>
        {baseVersion ? ` từ v${baseVersion}` : ""} ·{" "}
        <span className={understood ? "text-muted-foreground" : "text-warning"}>
          {label(REFINE_SOURCE_LABELS, source)}
        </span>
      </p>
      {notes.length > 0 ? (
        <div>
          <p className="text-xs text-muted-foreground">Đã điều chỉnh:</p>
          <ul className="list-inside list-disc">
            {notes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </div>
      ) : adjustments.length > 0 ? (
        <p>
          Đã điều chỉnh: {adjustments.map((a) => label(ADJUSTMENT_LABELS, a.kind)).join(", ")}
        </p>
      ) : null}
      <div className="flex flex-wrap items-center gap-3">
        {qualityScore !== null ? (
          <span className="text-xs text-muted-foreground">
            Điểm đánh giá ước lượng: {formatNumber(qualityScore, 1)}
          </span>
        ) : null}
        {artifact ? (
          <Button asChild size="sm" variant="outline" className="ml-auto">
            <a href={artifact.download_url} download>
              <Download aria-hidden />
              Tải v{version}
            </a>
          </Button>
        ) : null}
      </div>
    </div>
  );
}
