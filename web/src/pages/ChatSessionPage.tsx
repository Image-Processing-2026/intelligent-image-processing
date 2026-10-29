import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, FileJson, PanelRight } from "lucide-react";
import { useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { toast } from "sonner";

import { ApiError, errorMessage } from "@/api/client";
import type { Question, Rating, SessionDetail, Treatment, Variant } from "@/api/types";
import {
  SESSIONS_KEY,
  activateTreatment,
  addAnalysisNote,
  answerQuestions,
  clarifyAgain,
  deleteAnalysisEdit,
  exportUrl,
  refineTreatment,
  renderVariant,
  sessionKey,
  useSession,
} from "@/api/workspace";
import { WaitingNotice } from "@/components/WaitingNotice";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { ChatInputBar } from "@/components/chat/ChatInputBar";
import { DeleteSessionButton } from "@/components/session/DeleteSessionButton";
import { EditableTitle } from "@/components/session/EditableTitle";
import { type AnswerDraft, selectedValue } from "@/components/session/QuestionsCard";
import { StatusBadge } from "@/components/session/StatusBadge";
import {
  InspectionSidePane,
  type InspectionTab,
  useIsDesktop,
} from "@/components/sidepane/InspectionSidePane";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";

const EMPTY_DRAFT: AnswerDraft = { answers: {} };

function ChatSessionView({ detail }: { detail: SessionDetail }) {
  const queryClient = useQueryClient();
  const desktop = useIsDesktop();
  // Bản nháp câu trả lời giữ ở cấp trang: hỏi ý lại vẫn giữ lựa chọn nếu id câu hỏi trùng
  const [draft, setDraft] = useState<AnswerDraft>(EMPTY_DRAFT);
  const [paneOpen, setPaneOpen] = useState(false);
  const [paneTab, setPaneTab] = useState<InspectionTab>("masks");
  const inputRef = useRef<HTMLDivElement>(null);

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
      // Ghi chú đi qua khung chat bổ sung phân tích (backend nối tin nhắn chưa hiểu vào notes)
      return answerQuestions(detail.id, answers, "");
    },
    onSuccess: applyDetail,
    onError: onFlowError,
  });
  const addNote = useMutation({
    mutationFn: (message: string) => addAnalysisNote(detail.id, message),
    onSuccess: applyDetail,
    onError: (error) => {
      if (error instanceof ApiError && error.status !== 401) refresh();
    },
  });
  const removeEdit = useMutation({
    mutationFn: (index: number) => deleteAnalysisEdit(detail.id, index),
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
  const refine = useMutation({
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
    addNote.isPending ||
    removeEdit.isPending ||
    clarify.isPending ||
    render.isPending ||
    refine.isPending ||
    activate.isPending;

  const openPane = (tab: InspectionTab) => {
    setPaneTab(tab);
    setPaneOpen(true);
  };
  const onRated = (saved: Rating) => {
    queryClient.setQueryData<SessionDetail>(sessionKey(detail.id), (old) =>
      old ? { ...old, ratings: [...old.ratings.filter((r) => r.id !== saved.id), saved] } : old,
    );
  };

  const pane = (
    <InspectionSidePane
      detail={detail}
      tab={paneTab}
      onTabChange={setPaneTab}
      onClose={() => setPaneOpen(false)}
      className="h-full"
    />
  );

  return (
    <div className="absolute inset-0 flex min-h-0 flex-col overflow-hidden">
      <header className="flex flex-wrap items-center gap-2 border-b px-4 py-2">
        <div className="grid min-w-0 flex-1 gap-1">
          <EditableTitle key={detail.title} sessionId={detail.id} title={detail.title} />
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <StatusBadge status={busy ? "running" : detail.status} />
            <span>
              {detail.options.max_iterations} vòng tối đa
              {detail.options.has_ground_truth ? " · có ảnh chuẩn" : ""}
            </span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            aria-pressed={paneOpen}
            onClick={() => setPaneOpen(!paneOpen)}
          >
            <PanelRight aria-hidden />
            Bảng chi tiết
          </Button>
          <Button asChild variant="outline" size="sm">
            <a href={exportUrl(detail.id)} download={`session-${detail.id}.json`}>
              <FileJson aria-hidden />
              Xuất JSON phiên
            </a>
          </Button>
          <DeleteSessionButton sessionId={detail.id} disabled={busy || detail.status === "running"} />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto">
            {detail.error && !busy ? (
              <Alert variant="destructive" className="mx-auto mt-4 max-w-3xl">
                <AlertCircle aria-hidden />
                <AlertDescription>{detail.error.message}</AlertDescription>
              </Alert>
            ) : null}
            <ChatContainer
              detail={detail}
              actions={{
                draft,
                onDraftChange: setDraft,
                onStart: (questions) => answer.mutate(questions),
                onQuickNote: (text) => addNote.mutate(text),
                onRemoveEdit: (index) => removeEdit.mutate(index),
                onClarify: () => clarify.mutate(),
                onRender: (variant) => render.mutate(variant),
                renderingId: render.isPending ? (render.variables?.id ?? null) : null,
                onActivate: (treatment) => activate.mutate(treatment),
                onRated,
                onFocusInput: () => inputRef.current?.querySelector("textarea")?.focus(),
                onOpenPane: openPane,
                busy,
              }}
            >
              {/* Phiên được tải lại tự cập nhật (poll) cho tới khi pipeline xong */}
              {busy || detail.status === "running" ? <WaitingNotice /> : null}
            </ChatContainer>
          </div>
          <div ref={inputRef} className="border-t bg-background p-3">
            <div className="mx-auto max-w-3xl">
              <ChatInputBar
                sessionId={detail.id}
                sessionStatus={busy ? "running" : detail.status}
                activeTreatmentId={detail.active_treatment_id}
                onSendNote={(message) => addNote.mutateAsync(message)}
                onRefine={(feedback) => refine.mutateAsync(feedback)}
              />
            </div>
          </div>
        </div>

        {paneOpen && desktop ? (
          <aside className="hidden min-h-0 w-[35%] max-w-xl min-w-72 border-l lg:block">{pane}</aside>
        ) : null}
      </div>

      {!desktop ? (
        <Sheet open={paneOpen} onOpenChange={setPaneOpen}>
          <SheetContent side="right" className="w-full max-w-md gap-0 p-0">
            <SheetHeader className="sr-only">
              <SheetTitle>Chi tiết kỹ thuật</SheetTitle>
              <SheetDescription>Mặt nạ, số đo, lý do và thao tác đã thực thi.</SheetDescription>
            </SheetHeader>
            {pane}
          </SheetContent>
        </Sheet>
      ) : null}
    </div>
  );
}

export default function ChatSessionPage() {
  const { sid } = useParams();
  const session = useSession(sid);

  if (session.isPending) {
    return (
      <div className="mx-auto grid max-w-3xl gap-4 p-4 sm:p-6" aria-label="Đang tải phiên">
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
  return <ChatSessionView key={session.data.id} detail={session.data} />;
}
