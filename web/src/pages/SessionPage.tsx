import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, FileJson } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { toast } from "sonner";

import { ApiError, errorMessage } from "@/api/client";
import type {
  Adjustment,
  DiagnosisReport,
  Message,
  Question,
  SessionDetail,
  Treatment,
  Variant,
} from "@/api/types";
import {
  SESSIONS_KEY,
  answerQuestions,
  clarifyAgain,
  activateTreatment,
  exportUrl,
  refineTreatment,
  renderVariant,
  sessionKey,
  useSession,
} from "@/api/workspace";
import { WaitingNotice } from "@/components/WaitingNotice";
import {
  AnswersBubble,
  NoticeCard,
  RefineResultNote,
  RenderResultNote,
  UserBubble,
} from "@/components/session/ConversationBits";
import { RatingBox } from "@/components/result/RatingBox";
import { RefineBox } from "@/components/result/RefineBox";
import { TreatmentHistory } from "@/components/result/TreatmentHistory";
import { treatmentLabel } from "@/lib/actions";
import { DeleteSessionButton } from "@/components/session/DeleteSessionButton";
import { DiagnosisCard } from "@/components/session/DiagnosisCard";
import { EditableTitle } from "@/components/session/EditableTitle";
import { type AnswerDraft, QuestionsCard, selectedValue } from "@/components/session/QuestionsCard";
import { ResultCard } from "@/components/session/ResultCard";
import { StatusBadge } from "@/components/session/StatusBadge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDateTime } from "@/lib/labels";

// Loại message đã có giao diện ở lát hiện tại (feedback/refine/render thêm ở S4–S5)
const RENDERED_KINDS = new Set<Message["kind"]>([
  "diagnosis",
  "questions",
  "notice",
  "answers",
  "result",
  "render_result",
  "feedback",
  "refine_result",
]);

const EMPTY_DRAFT: AnswerDraft = { answers: {}, notes: "" };

function lastIndexOf(messages: Message[], kind: Message["kind"]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].kind === kind) return index;
  }
  return -1;
}

function SessionView({ detail }: { detail: SessionDetail }) {
  const queryClient = useQueryClient();
  // Bản nháp câu trả lời giữ ở cấp trang: hỏi ý lại vẫn giữ lựa chọn nếu id câu hỏi trùng
  const [draft, setDraft] = useState<AnswerDraft>(EMPTY_DRAFT);

  const applyDetail = (next: SessionDetail) => {
    queryClient.setQueryData(sessionKey(next.id), next);
    void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
  };
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: sessionKey(detail.id) });
    void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
  };
  const onFlowError = (error: Error) => {
    toast.error(errorMessage(error));
    // Backend đã ghi notice / trạng thái mới (thread hết hạn, lỗi pipeline): tải lại phiên
    if (error instanceof ApiError && error.status !== 401) refresh();
  };

  const answer = useMutation({
    mutationFn: (questions: Question[]) => {
      const answers = Object.fromEntries(questions.map((q) => [q.id, selectedValue(q, draft)]));
      return answerQuestions(detail.id, answers, draft.notes.trim());
    },
    onSuccess: applyDetail,
    onError: onFlowError,
  });
  const clarify = useMutation({
    mutationFn: () => clarifyAgain(detail.id),
    onSuccess: applyDetail,
    onError: onFlowError,
  });

  const render = useMutation({
    mutationFn: (variant: Variant) => renderVariant(detail.id, variant.id),
    onSuccess: (next) => {
      applyDetail(next);
      toast.success("Đã render phiên bản ở độ phân giải gốc.");
    },
    onError: onFlowError,
  });

  const refineMutation = useMutation({
    mutationFn: (feedback: string) =>
      refineTreatment(detail.id, feedback, detail.active_treatment_id),
    onSuccess: applyDetail,
    onError: onFlowError,
  });
  const activate = useMutation({
    mutationFn: (treatment: Treatment) => activateTreatment(detail.id, treatment.id),
    onSuccess: (next) => {
      applyDetail(next);
      toast.success("Đã chuyển phiên bản đang dùng.");
    },
    onError: onFlowError,
  });

  const busy =
    answer.isPending ||
    clarify.isPending ||
    render.isPending ||
    refineMutation.isPending ||
    activate.isPending;
  const runningElsewhere = detail.status === "running" && !busy;
  const messages = detail.messages.filter((message) => RENDERED_KINDS.has(message.kind));
  const currentQuestions = lastIndexOf(messages, "questions");
  const lastNotice = lastIndexOf(messages, "notice");
  const canClarify =
    !detail.process && (detail.status === "awaiting_answers" || detail.status === "failed");
  const artifactById = new Map(detail.artifacts.map((artifact) => [artifact.id, artifact]));
  const activeTreatment = detail.treatments.find((t) => t.id === detail.active_treatment_id);
  const activeLabel = activeTreatment
    ? treatmentLabel(activeTreatment, detail.variants)
    : "kết quả pipeline";

  const renderMessage = (message: Message, index: number) => {
    const content = message.content;
    switch (message.kind) {
      case "diagnosis":
        return (
          <DiagnosisCard
            diagnosis={(content.diagnosis as DiagnosisReport | null) ?? null}
            original={detail.original}
          />
        );
      case "questions": {
        const questions = (content.questions as Question[]) ?? [];
        const next = messages[index + 1];
        const answeredWith =
          next?.kind === "answers" ? (next.content.answers as Record<string, string>) : undefined;
        return (
          <QuestionsCard
            questions={questions}
            active={index === currentQuestions && detail.status === "awaiting_answers"}
            draft={draft}
            onDraftChange={setDraft}
            onSubmit={() => answer.mutate(questions)}
            pending={busy}
            answeredWith={answeredWith}
          />
        );
      }
      case "answers":
        return (
          <AnswersBubble
            labels={(content.labels as { question: string; answer: string }[]) ?? []}
            notes={String(content.notes ?? "")}
          />
        );
      case "result":
        return detail.process && detail.process.run_id === content.run_id ? (
          <ResultCard
            detail={detail}
            process={detail.process}
            onRender={(variant) => render.mutate(variant)}
            renderingId={render.isPending ? (render.variables?.id ?? null) : null}
            busy={busy || detail.status !== "done"}
          >
            <RatingBox
              // Đổi phiên bản đang xem → form đánh giá đọc lại đánh giá của phiên bản đó
              key={detail.active_treatment_id ?? "none"}
              sessionId={detail.id}
              runId={detail.process.run_id}
              treatmentId={detail.active_treatment_id}
              treatmentLabel={activeLabel}
              existing={
                detail.ratings.find(
                  (r) =>
                    r.run_id === detail.process?.run_id &&
                    r.treatment_id === detail.active_treatment_id,
                ) ?? null
              }
            />
            <RefineBox
              activeLabel={activeLabel}
              onSubmit={(feedback) => refineMutation.mutateAsync(feedback)}
              pending={refineMutation.isPending}
              disabled={busy || detail.status !== "done"}
            />
            <TreatmentHistory
              treatments={detail.treatments}
              variants={detail.variants}
              activeId={detail.active_treatment_id}
              artifactById={artifactById}
              onActivate={(treatment) => activate.mutate(treatment)}
              disabled={busy || detail.status !== "done"}
            />
          </ResultCard>
        ) : null;
      case "feedback":
        return (
          <UserBubble title="Góp ý của bạn">
            <p>{String(content.text ?? "")}</p>
          </UserBubble>
        );
      case "refine_result":
        return (
          <RefineResultNote
            version={Number(content.version)}
            baseVersion={content.base_version ? Number(content.base_version) : null}
            notes={(content.notes as string[]) ?? []}
            adjustments={(content.adjustments as Adjustment[]) ?? []}
            source={String(content.source ?? "none")}
            qualityScore={typeof content.quality_score === "number" ? content.quality_score : null}
            artifact={artifactById.get(String(content.artifact_id)) ?? null}
          />
        );
      case "render_result": {
        const artifact = artifactById.get(String(content.artifact_id));
        return (
          <RenderResultNote
            label={String(content.label ?? "")}
            width={Number(content.width)}
            height={Number(content.height)}
            downloadUrl={artifact?.download_url ?? null}
          />
        );
      }
      case "notice": {
        const withAction = index === lastNotice && canClarify;
        return (
          <NoticeCard
            text={String(content.text ?? "")}
            actionLabel={withAction ? "Hỏi ý lại" : undefined}
            onAction={withAction ? () => clarify.mutate() : undefined}
            pending={busy}
          />
        );
      }
      default:
        return null;
    }
  };

  return (
    <div className="mx-auto grid max-w-4xl gap-4 p-4 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid min-w-0 flex-1 gap-1">
          <EditableTitle key={detail.title} sessionId={detail.id} title={detail.title} />
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <StatusBadge status={busy ? "running" : detail.status} />
            <span>Tạo lúc {formatDateTime(detail.created_at)}</span>
            <span>
              · {detail.options.max_iterations} vòng tối đa · {detail.options.num_variants} phiên
              bản
              {detail.options.has_ground_truth ? " · có ảnh chuẩn" : ""}
            </span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline" size="sm">
            <a href={exportUrl(detail.id)} download={`session-${detail.id}.json`}>
              <FileJson aria-hidden />
              Xuất JSON phiên
            </a>
          </Button>
          <DeleteSessionButton
            sessionId={detail.id}
            disabled={busy || detail.status === "running"}
          />
        </div>
      </div>

      {detail.error && !busy ? (
        <Alert variant="destructive">
          <AlertCircle aria-hidden />
          <AlertDescription>{detail.error.message}</AlertDescription>
        </Alert>
      ) : null}

      <ol className="grid gap-4" aria-label="Hội thoại">
        {messages.map((message, index) => {
          const view = renderMessage(message, index);
          return view ? <li key={message.id}>{view}</li> : null;
        })}
      </ol>

      {busy ? <WaitingNotice /> : null}
      {runningElsewhere ? (
        // Không poll (D3): người dùng tự tải lại khi muốn xem kết quả
        <NoticeCard
          text="Phiên đang được xử lý ở một yêu cầu khác. Tải lại sau ít phút để xem kết quả."
          actionLabel="Tải lại"
          onAction={refresh}
        />
      ) : null}
    </div>
  );
}

export default function SessionPage() {
  const { sid } = useParams();
  const session = useSession(sid);

  if (session.isPending) {
    return (
      <div className="mx-auto grid max-w-4xl gap-4 p-4 sm:p-6" aria-label="Đang tải phiên">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-72 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  if (session.isError) {
    const notFound = session.error instanceof ApiError && session.error.status === 404;
    return (
      <div className="mx-auto grid max-w-xl gap-3 p-6 text-center">
        <h1 className="text-lg font-semibold">
          {notFound ? "Không tìm thấy phiên xử lý" : "Không tải được phiên xử lý"}
        </h1>
        <p className="text-sm text-muted-foreground" role="alert">
          {notFound
            ? "Phiên không tồn tại hoặc bạn không có quyền xem."
            : errorMessage(session.error)}
        </p>
        <div className="flex justify-center gap-2">
          {!notFound ? (
            <Button variant="outline" onClick={() => session.refetch()}>
              Thử lại
            </Button>
          ) : null}
          <Button asChild>
            <Link to="/">Về Phiên mới</Link>
          </Button>
        </div>
      </div>
    );
  }

  // key theo id: đổi phiên thì bản nháp câu trả lời không mang sang phiên khác
  return <SessionView key={session.data.id} detail={session.data} />;
}
