import { useEffect, useRef } from "react";

import type {
  Adjustment,
  DiagnosisEdit,
  DiagnosisReport,
  Message,
  Question,
  Rating,
  SessionDetail,
  Treatment,
  Variant,
} from "@/api/types";
import { CompareSlider } from "@/components/result/CompareSlider";
import { TreatmentHistory } from "@/components/result/TreatmentHistory";
import {
  AnalysisUpdateNote,
  AnswersBubble,
  NoticeCard,
  RefineResultNote,
  RenderResultNote,
  UserBubble,
} from "@/components/session/ConversationBits";
import type { AnswerDraft } from "@/components/session/QuestionsCard";
import { latestDiagnosis } from "@/components/studio/flow-data";
import type { InspectionTab } from "@/components/sidepane/InspectionSidePane";

import { DiagnosisBubble } from "./bubbles/DiagnosisBubble";
import { FallbackBubble } from "./bubbles/FallbackBubble";
import { ResultBubble } from "./bubbles/ResultBubble";
import { UserPromptBubble } from "./bubbles/UserPromptBubble";

export interface ChatActions {
  draft: AnswerDraft;
  onDraftChange: (draft: AnswerDraft) => void;
  /** Bắt đầu xử lý với bộ câu hỏi hiện hành. */
  onStart: (questions: Question[]) => void;
  onQuickNote: (text: string) => void;
  onRemoveEdit: (index: number) => void;
  onClarify: () => void;
  onRender: (variant: Variant) => void;
  renderingId: string | null;
  onActivate: (treatment: Treatment) => void;
  onRated: (rating: Rating) => void;
  onFocusInput: () => void;
  onOpenPane: (tab: InspectionTab) => void;
  busy: boolean;
}

function lastIndexOf(messages: Message[], kind: Message["kind"]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].kind === kind) return index;
  }
  return -1;
}

/**
 * Danh sách tin nhắn của phiên: mỗi `message.kind` một bong bóng riêng, kind lạ rơi về
 * FallbackBubble. Tự cuộn xuống cuối khi có tin nhắn mới.
 */
export function ChatContainer({
  detail,
  actions,
  children,
}: {
  detail: SessionDetail;
  actions: ChatActions;
  /** Nội dung chèn sau tin nhắn cuối (thông báo đang chờ...). */
  children?: React.ReactNode;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const messages = detail.messages;
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages.length, detail.status, detail.treatments.length]);

  const currentQuestions = lastIndexOf(messages, "questions");
  const currentDiagnosis = lastIndexOf(messages, "diagnosis");
  const lastNotice = lastIndexOf(messages, "notice");
  const awaiting = detail.status === "awaiting_answers";
  const questionList =
    currentQuestions >= 0 ? ((messages[currentQuestions].content.questions as Question[]) ?? []) : [];
  const canClarify = !detail.process && (awaiting || detail.status === "failed");
  const artifactById = new Map(detail.artifacts.map((artifact) => [artifact.id, artifact]));
  // Yêu cầu riêng của người dùng sau lần chẩn đoán hiện hành (ngoài yêu cầu ban đầu đã thành chip)
  const customReady =
    (detail.draft_edits ?? []).length > 0 ||
    messages.some((m, i) => i > currentDiagnosis && m.kind === "analysis_note");
  const hasPrompt = messages.some((message) => message.kind === "user_prompt");

  const renderMessage = (message: Message, index: number): React.ReactNode => {
    const content = message.content ?? {};
    switch (message.kind) {
      case "user_prompt":
        return <UserPromptBubble text={String(content.text ?? "")} original={detail.original} />;
      case "diagnosis": {
        const stored = (content.diagnosis as DiagnosisReport | null) ?? null;
        const live = awaiting && index === currentDiagnosis;
        return (
          <DiagnosisBubble
            diagnosis={live ? (latestDiagnosis(detail) ?? stored) : stored}
            live={live}
            questions={questionList.filter(q => q.id !== "approach")}
            draft={actions.draft}
            onDraftChange={actions.onDraftChange}
            edits={detail.draft_edits ?? []}
            customReady={customReady}
            onRemoveEdit={actions.onRemoveEdit}
            onQuickNote={actions.onQuickNote}
            onStart={() => actions.onStart(questionList)}
            onOpenDetails={() => actions.onOpenPane(live ? "masks" : "metrics")}
            pending={actions.busy}
          />
        );
      }
      // Câu hỏi được gộp hiển thị trong DiagnosisBubble
      case "questions":
        return null;
      case "analysis_note":
      case "feedback":
        return (
          <UserBubble title={message.kind === "feedback" ? "Góp ý của bạn" : "Bổ sung của bạn"}>
            <p>{String(content.text ?? "")}</p>
          </UserBubble>
        );
      case "analysis_update":
        return (
          <AnalysisUpdateNote
            edits={(content.edits as DiagnosisEdit[]) ?? []}
            unparsed={Boolean(content.unparsed)}
          />
        );
      case "answers":
        return (
          <AnswersBubble
            labels={(content.labels as { question: string; answer: string }[]) ?? []}
            notes={String(content.notes ?? "")}
          />
        );
      case "result":
        return detail.process && detail.process.run_id === content.run_id ? (
          <ResultBubble
            detail={detail}
            onRender={actions.onRender}
            renderingId={actions.renderingId}
            busy={actions.busy}
            onRated={actions.onRated}
            onFocusInput={actions.onFocusInput}
            onOpenExecution={() => actions.onOpenPane("execution")}
          >
            <TreatmentHistory
              treatments={detail.treatments}
              variants={detail.variants}
              activeId={detail.active_treatment_id}
              artifactById={artifactById}
              onActivate={actions.onActivate}
              disabled={actions.busy || detail.status !== "done"}
            />
          </ResultBubble>
        ) : null;
      case "refine_result": {
        const artifact = artifactById.get(String(content.artifact_id)) ?? null;
        return (
          <div className="grid gap-3">
            <RefineResultNote
              version={Number(content.version)}
              baseVersion={content.base_version ? Number(content.base_version) : null}
              notes={(content.notes as string[]) ?? []}
              adjustments={(content.adjustments as Adjustment[]) ?? []}
              source={String(content.source ?? "none")}
              qualityScore={
                typeof content.quality_score === "number" ? content.quality_score : null
              }
              artifact={artifact}
            />
            {detail.original && artifact ? (
              <CompareSlider
                before={detail.original}
                after={artifact}
                afterLabel={`v${Number(content.version)}`}
              />
            ) : null}
          </div>
        );
      }
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
            onAction={withAction ? actions.onClarify : undefined}
            pending={actions.busy}
          />
        );
      }
      default:
        return <FallbackBubble message={message} />;
    }
  };

  return (
    <ol className="mx-auto grid w-full max-w-3xl min-w-0 gap-4 p-4" aria-label="Hội thoại">
      {!hasPrompt && detail.original ? (
        // Phiên cũ không có yêu cầu ban đầu: vẫn mở đầu bằng ảnh gốc
        <li key="original" className="min-w-0">
          <UserPromptBubble text="" original={detail.original} />
        </li>
      ) : null}
      {messages.map((message, index) => {
        const view = renderMessage(message, index);
        return view ? (
          <li key={message.id} data-message-id={message.id} data-kind={message.kind} className="min-w-0">
            {view}
          </li>
        ) : null;
      })}
      {children ? <li className="min-w-0">{children}</li> : null}
      <div ref={endRef} aria-hidden />
    </ol>
  );
}
