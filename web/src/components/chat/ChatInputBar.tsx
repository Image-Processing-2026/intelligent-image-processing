import { useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  ImagePlus,
  Loader2,
  Paperclip,
  SendHorizontal,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useMe } from "@/api/auth";
import { errorMessage } from "@/api/client";
import type { SessionDetail, SessionStatus } from "@/api/types";
import {
  type NewSessionInput,
  SESSIONS_KEY,
  addAnalysisNote,
  createSession,
  refineTreatment,
  sessionKey,
  useSamples,
} from "@/api/workspace";
import {
  ACCEPTED_TYPES,
  checkFile,
  useObjectUrl,
} from "@/components/new-session/Dropzone";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export const CHAT_INPUT_MAX_LENGTH = 1000;
const ITERATION_CHOICES = [1, 2, 3, 4, 5];

export type ChatInputMode = "new" | "analysis_note" | "refine" | "disabled";

export interface ChatInputBarProps {
  /** ID phiên làm việc hiện tại nếu đã có */
  sessionId?: string;
  /** Trạng thái phiên hiện tại */
  sessionStatus?: SessionStatus;
  /** Chế độ chỉ định cụ thể (ghi đè logic tự suy từ status) */
  mode?: ChatInputMode;
  /** Treatment ID cần refine (nếu muốn chỉ định phiên bản cụ thể) */
  activeTreatmentId?: string | null;

  /** Callback tùy biến khi tạo phiên mới */
  onNewSession?: (input: NewSessionInput) => Promise<SessionDetail | void>;
  /** Callback tùy biến khi gửi bổ sung phân tích */
  onSendNote?: (message: string) => Promise<unknown>;
  /** Callback tùy biến khi gửi góp ý tinh chỉnh */
  onRefine?: (feedback: string) => Promise<unknown>;
  onDraftTextChange?: (text: string) => void;

  /** Khóa thao tác từ bên ngoài */
  disabled?: boolean;
  /** Placeholder tùy chỉnh */
  placeholder?: string;
  /** Tự động focus ô nhập liệu */
  autoFocus?: boolean;
  /** Class CSS tùy biến cho container ngoài */
  className?: string;
}

/** Xác định chế độ hoạt động của ChatInputBar dựa trên props và trạng thái phiên */
export function resolveInputMode(
  modeProp?: ChatInputMode,
  sessionStatus?: SessionStatus,
  sessionId?: string,
): ChatInputMode {
  if (modeProp) return modeProp;
  if (!sessionId) return "new";
  if (sessionStatus === "awaiting_answers") return "analysis_note";
  if (sessionStatus === "done") return "refine";
  if (
    sessionStatus === "running" ||
    sessionStatus === "uploaded" ||
    sessionStatus === "failed"
  ) {
    return "disabled";
  }
  return "new";
}

/**
 * Thanh nhập liệu đa phương thức Conversational Image Doctor.
 * Hỗ trợ:
 * - Kéo thả ảnh & paste trực tiếp từ clipboard (Ctrl+V).
 * - Thumbnail Chip có nút ✕ gỡ ảnh.
 * - Textarea tự co giãn tối đa 1000 ký tự, gửi bằng Enter (Shift+Enter để xuống dòng).
 * - Modal Tùy chọn nâng cao (chọn ảnh mẫu, ảnh chuẩn ground truth, max_iterations).
 * - Tự động điều phối theo ngữ cảnh (Tạo phiên / Note chẩn đoán / Refine kết quả).
 */
export function ChatInputBar({
  sessionId,
  sessionStatus,
  mode: explicitMode,
  activeTreatmentId,
  onNewSession,
  onSendNote,
  onRefine,
  onDraftTextChange,
  disabled = false,
  placeholder,
  autoFocus = false,
  className,
}: ChatInputBarProps) {
  const me = useMe();
  const samples = useSamples();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const effectiveMode = resolveInputMode(explicitMode, sessionStatus, sessionId);
  const isNewMode = effectiveMode === "new";

  // Form states
  const [file, setFile] = useState<File | null>(null);
  const [sample, setSample] = useState<string | null>(null);
  const [groundTruth, setGroundTruth] = useState<File | null>(null);
  const [iterations, setIterations] = useState<number>(
    me.data?.settings.max_iterations ?? 3,
  );
  const [text, setText] = useState("");
  useEffect(() => { onDraftTextChange?.(text); }, [text, onDraftTextChange]);
  const [isDragging, setIsDragging] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // References
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const gtInputRef = useRef<HTMLInputElement>(null);
  const iterationsId = useId();

  // Object URLs for previews
  const filePreview = useObjectUrl(file);
  const gtPreview = useObjectUrl(groundTruth);

  // Auto-resize textarea height
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  // Cập nhật số vòng mặc định khi thông tin người dùng được tải về
  useEffect(() => {
    if (me.data?.settings.max_iterations) {
      setIterations(me.data.settings.max_iterations);
    }
  }, [me.data?.settings.max_iterations]);

  // Nhãn của ảnh mẫu đang chọn
  const sampleLabel = useMemo(() => {
    if (!sample || !samples.data) return null;
    return samples.data.find((item) => item.id === sample)?.label ?? sample;
  }, [sample, samples.data]);

  // Ảnh mẫu đang chọn
  const sampleItem = useMemo(() => {
    if (!sample || !samples.data) return null;
    return samples.data.find((item) => item.id === sample);
  }, [sample, samples.data]);

  // Kiểm tra tính hợp lệ và đính kèm file ảnh người dùng
  const attachImageFile = (candidate: File | undefined | null) => {
    if (!candidate) return;
    const problem = checkFile(candidate);
    if (problem) {
      setFileError(problem);
      return;
    }
    setFile(candidate);
    setSample(null);
    setFileError(null);
  };

  // Kiểm tra tính hợp lệ và đính kèm ảnh chuẩn ground truth
  const attachGroundTruth = (candidate: File | undefined | null) => {
    if (!candidate) return;
    const problem = checkFile(candidate);
    if (problem) {
      setFileError(problem);
      return;
    }
    setGroundTruth(candidate);
    setFileError(null);
  };

  // Xử lý sự kiện Paste từ clipboard (Ctrl+V)
  const handlePaste = (event: React.ClipboardEvent) => {
    // 1. Kiểm tra clipboardData.files
    const files = event.clipboardData?.files;
    if (files && files.length > 0) {
      const firstFile = files[0];
      if (firstFile.type.startsWith("image/")) {
        event.preventDefault();
        if (!isNewMode) {
          setFileError("Chỉ có thể đính kèm ảnh khi tạo phiên mới.");
          return;
        }
        attachImageFile(firstFile);
        return;
      }
    }

    // 2. Kiểm tra clipboardData.items
    const items = event.clipboardData?.items;
    if (items && items.length > 0) {
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.startsWith("image/")) {
          event.preventDefault();
          if (!isNewMode) {
            setFileError("Chỉ có thể đính kèm ảnh khi tạo phiên mới.");
            return;
          }
          const candidate = item.getAsFile();
          if (candidate) {
            attachImageFile(candidate);
          }
          return;
        }
      }
    }
    // Nếu là text thông thường, không preventDefault để textarea nhận text tự nhiên.
  };

  // Kéo thả file ảnh
  const handleDragOver = (event: React.DragEvent) => {
    if (!isNewMode || disabled || busy) return;
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    if (event.currentTarget.contains(event.relatedTarget as Node)) return;
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent) => {
    if (!isNewMode || disabled || busy) return;
    event.preventDefault();
    setIsDragging(false);
    const candidate = event.dataTransfer.files?.[0];
    if (candidate) {
      attachImageFile(candidate);
    }
  };

  // Gỡ ảnh đính kèm
  const handleRemoveFile = () => {
    setFile(null);
    setFileError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Gỡ ảnh mẫu
  const handleRemoveSample = () => {
    setSample(null);
    setFileError(null);
  };

  // Gỡ ảnh chuẩn
  const handleRemoveGroundTruth = () => {
    setGroundTruth(null);
    setFileError(null);
    if (gtInputRef.current) {
      gtInputRef.current.value = "";
    }
  };

  // Xác định điều kiện có thể submit
  const isLocked = disabled || busy || effectiveMode === "disabled";
  const trimmed = text.trim();
  const canSubmit = useMemo(() => {
    if (isLocked) return false;
    if (isNewMode) {
      return Boolean(file || sample);
    }
    return trimmed.length > 0;
  }, [isLocked, isNewMode, file, sample, trimmed]);

  // Hành vi gửi
  const handleSubmit = async () => {
    if (isLocked) return;

    if (isNewMode) {
      if (!file && !sample) {
        setFileError("Vui lòng chọn hoặc dán ảnh để bắt đầu.");
        return;
      }

      setBusy(true);
      setSubmitError(null);
      setFileError(null);

      try {
        const input: NewSessionInput = {
          file,
          sample: file ? null : sample,
          groundTruth,
          maxIterations: iterations,
          userPrompt: trimmed || null,
        };

        if (onNewSession) {
          await onNewSession(input);
        } else {
          const detail = await createSession(input);
          queryClient.setQueryData(sessionKey(detail.id), detail);
          void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
          navigate(`/s/${detail.id}`);
        }

        // Dọn dẹp trạng thái sau khi gửi thành công
        setText("");
        setFile(null);
        setSample(null);
        setGroundTruth(null);
      } catch (err) {
        setSubmitError(errorMessage(err));
      } finally {
        setBusy(false);
      }
      return;
    }

    if (!trimmed) return;

    setBusy(true);
    setSubmitError(null);

    if (effectiveMode === "analysis_note") {
      try {
        if (onSendNote) {
          await onSendNote(trimmed);
        } else if (sessionId) {
          const updated = await addAnalysisNote(sessionId, trimmed);
          queryClient.setQueryData(sessionKey(sessionId), updated);
          void queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
        }
        setText("");
      } catch (err) {
        setSubmitError(errorMessage(err));
      } finally {
        setBusy(false);
      }
      return;
    }

    if (effectiveMode === "refine") {
      try {
        if (onRefine) {
          await onRefine(trimmed);
        } else if (sessionId) {
          const updated = await refineTreatment(sessionId, trimmed, activeTreatmentId);
          queryClient.setQueryData(sessionKey(sessionId), updated);
          void queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
        }
        setText("");
      } catch (err) {
        setSubmitError(errorMessage(err));
      } finally {
        setBusy(false);
      }
    }
  };

  // Nhấn Enter để gửi (Shift+Enter để xuống dòng)
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (canSubmit) {
        void handleSubmit();
      } else if (isNewMode && !file && !sample) {
        setFileError("Vui lòng chọn hoặc dán ảnh để bắt đầu.");
      }
    }
  };

  // Xác định placeholder theo ngữ cảnh
  const resolvedPlaceholder = useMemo(() => {
    if (placeholder) return placeholder;
    switch (effectiveMode) {
      case "new":
        return "Mô tả yêu cầu chỉnh sửa ảnh (tùy chọn)... Nhấn Enter để gửi";
      case "analysis_note":
        return "Nhập phản hồi hoặc yêu cầu chỉnh sửa chẩn đoán... Nhấn Enter để gửi";
      case "refine":
        return "Nhập góp ý tinh chỉnh cho kết quả... Nhấn Enter để gửi";
      case "disabled":
        return "Hệ thống đang xử lý, vui lòng chờ...";
    }
  }, [placeholder, effectiveMode]);

  // Nhãn nút gửi theo ngữ cảnh
  const submitButtonLabel = useMemo(() => {
    switch (effectiveMode) {
      case "new":
        return "Bắt đầu chẩn đoán";
      case "analysis_note":
        return "Gửi bổ sung phân tích";
      case "refine":
        return "Gửi góp ý tinh chỉnh";
      case "disabled":
        return "Đang bận";
    }
  }, [effectiveMode]);

  return (
    <div className={cn("relative w-full max-w-4xl mx-auto", className)}>
      {/* Banner thông báo lỗi nếu có */}
      {(fileError || submitError) && (
        <Alert variant="destructive" className="mb-2 py-2 text-xs shadow-sm">
          <AlertCircle className="size-4" />
          <AlertDescription className="flex items-center justify-between w-full">
            <span>{fileError || submitError}</span>
            <button
              type="button"
              onClick={() => {
                setFileError(null);
                setSubmitError(null);
              }}
              className="ml-2 rounded-xs p-0.5 hover:opacity-80"
              aria-label="Đóng thông báo lỗi"
            >
              <X className="size-3.5" />
            </button>
          </AlertDescription>
        </Alert>
      )}

      {/* Vùng nhập liệu chính */}
      <div
        className={cn(
          "relative flex flex-col rounded-2xl border bg-background/95 backdrop-blur-sm shadow-md transition-all focus-within:border-primary/50 focus-within:ring-2 focus-within:ring-primary/20",
          isDragging && "border-dashed border-primary bg-primary/5 ring-2 ring-primary/30",
          disabled && "opacity-60 pointer-events-none",
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onPaste={handlePaste}
      >
        {/* Overlay trực quan khi đang kéo thả file */}
        {isDragging && (
          <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-2xl border-2 border-dashed border-primary bg-background/90 text-sm font-medium text-primary">
            <ImagePlus className="mr-2 size-5" />
            Thả ảnh vào đây để tải lên
          </div>
        )}

        {/* Thumbnail Chips danh sách ảnh đính kèm */}
        {isNewMode && (file || sample || groundTruth) && (
          <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
            {/* Chip ảnh đính kèm từ file */}
            {file && (
              <div className="flex items-center gap-2 rounded-lg border bg-muted/50 p-1.5 pr-2 text-xs shadow-xs">
                {filePreview ? (
                  <img
                    src={filePreview}
                    alt={file.name}
                    className="size-8 rounded object-cover border"
                  />
                ) : null}
                <div className="flex flex-col min-w-0 max-w-44">
                  <span className="truncate font-medium">{file.name}</span>
                  <span className="text-[10px] text-muted-foreground">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </span>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  className="size-5 text-muted-foreground hover:text-foreground"
                  aria-label={`Gỡ ảnh ${file.name}`}
                  onClick={handleRemoveFile}
                  disabled={busy}
                >
                  <X className="size-3" />
                </Button>
              </div>
            )}

            {/* Chip ảnh mẫu có sẵn */}
            {sample && !file && (
              <div className="flex items-center gap-2 rounded-lg border bg-muted/50 p-1.5 pr-2 text-xs shadow-xs">
                {sampleItem?.thumbnail_url ? (
                  <img
                    src={sampleItem.thumbnail_url}
                    alt=""
                    className="size-8 rounded object-cover border"
                  />
                ) : null}
                <div className="flex flex-col min-w-0 max-w-44">
                  <span className="truncate font-medium">{sampleItem?.label ?? sample}</span>
                  <span className="text-[10px] text-muted-foreground">Ảnh mẫu</span>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  className="size-5 text-muted-foreground hover:text-foreground"
                  aria-label="Bỏ chọn ảnh mẫu"
                  onClick={handleRemoveSample}
                  disabled={busy}
                >
                  <X className="size-3" />
                </Button>
              </div>
            )}

            {/* Chip ảnh chuẩn Ground Truth */}
            {groundTruth && (
              <div className="flex items-center gap-2 rounded-lg border border-dashed border-primary/40 bg-primary/5 p-1.5 pr-2 text-xs shadow-xs">
                {gtPreview ? (
                  <img
                    src={gtPreview}
                    alt={groundTruth.name}
                    className="size-8 rounded object-cover border"
                  />
                ) : null}
                <div className="flex flex-col min-w-0 max-w-36">
                  <span className="truncate font-medium">{groundTruth.name}</span>
                  <span className="text-[10px] text-primary/80">Ảnh chuẩn</span>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  className="size-5 text-muted-foreground hover:text-foreground"
                  aria-label="Gỡ ảnh chuẩn"
                  onClick={handleRemoveGroundTruth}
                  disabled={busy}
                >
                  <X className="size-3" />
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Khung nhập text & nút thao tác */}
        <div className="flex flex-col p-2">
          <Textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            maxLength={CHAT_INPUT_MAX_LENGTH}
            placeholder={resolvedPlaceholder}
            disabled={disabled || busy || effectiveMode === "disabled"}
            autoFocus={autoFocus}
            className="field-sizing-content min-h-11 max-h-44 resize-none border-0 bg-transparent px-2.5 py-1.5 shadow-none focus-visible:ring-0 text-sm placeholder:text-muted-foreground/70"
          />

          <div className="flex items-center justify-between gap-2 pt-1 px-1">
            {/* Nhóm thao tác bên trái */}
            <div className="flex items-center gap-1">
              {/* Nút đính kèm ảnh (chỉ hiển thị ở chế độ phiên mới) */}
              {isNewMode && (
                <>
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept={ACCEPTED_TYPES.join(",")}
                    className="hidden"
                    aria-label="Tải ảnh lên"
                    onChange={(e) => {
                      attachImageFile(e.target.files?.[0]);
                    }}
                    disabled={busy}
                  />
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        className="size-8 text-muted-foreground hover:text-foreground"
                        aria-label="Đính kèm ảnh"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={busy}
                      >
                        <Paperclip className="size-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      Đính kèm ảnh (hoặc kéo thả / dán Ctrl+V)
                    </TooltipContent>
                  </Tooltip>

                  {/* Nút Tùy chọn nâng cao */}
                  <Dialog open={advancedOpen} onOpenChange={setAdvancedOpen}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <DialogTrigger asChild>
                          <Button
                            type="button"
                            variant={
                              sample || groundTruth || iterations !== (me.data?.settings.max_iterations ?? 3)
                                ? "secondary"
                                : "ghost"
                            }
                            size="icon-sm"
                            className="size-8 text-muted-foreground hover:text-foreground"
                            aria-label="Tùy chọn nâng cao"
                            disabled={busy}
                          >
                            <SlidersHorizontal className="size-4" />
                          </Button>
                        </DialogTrigger>
                      </TooltipTrigger>
                      <TooltipContent side="top">Tùy chọn nâng cao</TooltipContent>
                    </Tooltip>
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle>Tùy chọn nâng cao</DialogTitle>
                        <DialogDescription>
                          Tùy chỉnh số vòng lặp phác đồ, chọn ảnh mẫu thử nghiệm hoặc tải ảnh chuẩn để đánh giá PSNR/SSIM.
                        </DialogDescription>
                      </DialogHeader>

                      <div className="grid gap-5 py-2">
                        {/* Số vòng tối đa */}
                        <div className="grid gap-2">
                          <label
                            id={iterationsId}
                            htmlFor="input-max-iterations"
                            className="text-sm font-medium"
                          >
                            Số vòng tối đa (max_iterations)
                          </label>
                          <ToggleGroup
                            id="input-max-iterations"
                            type="single"
                            variant="outline"
                            value={String(iterations)}
                            onValueChange={(val) => val && setIterations(Number(val))}
                            aria-labelledby={iterationsId}
                            className="w-fit"
                          >
                            {ITERATION_CHOICES.map((num) => (
                              <ToggleGroupItem
                                key={num}
                                value={String(num)}
                                className="min-w-10 text-xs"
                              >
                                {num}
                              </ToggleGroupItem>
                            ))}
                          </ToggleGroup>
                          <p className="text-xs text-muted-foreground">
                            Mỗi vòng: chẩn đoán → lập phác đồ → xử lý → đánh giá.
                          </p>
                        </div>

                        {/* Chọn ảnh mẫu (Samples) */}
                        {samples.data && samples.data.length > 0 && (
                          <div className="grid gap-2">
                            <span className="text-sm font-medium">Hoặc dùng ảnh mẫu có sẵn</span>
                            <div className="grid grid-cols-3 gap-2">
                              {samples.data.map((item) => {
                                const isSelected = sample === item.id && !file;
                                return (
                                  <button
                                    key={item.id}
                                    type="button"
                                    aria-pressed={isSelected}
                                    onClick={() => {
                                      if (isSelected) {
                                        setSample(null);
                                      } else {
                                        setSample(item.id);
                                        setFile(null);
                                        setFileError(null);
                                      }
                                    }}
                                    className={cn(
                                      "grid gap-1 rounded-md border p-1.5 text-left text-xs transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring",
                                      isSelected && "border-primary ring-2 ring-primary/40 bg-primary/5",
                                    )}
                                  >
                                    <img
                                      src={item.thumbnail_url}
                                      alt=""
                                      className="aspect-[4/3] w-full rounded object-cover"
                                    />
                                    <span className="truncate font-medium">{item.label}</span>
                                  </button>
                                );
                              })}
                            </div>
                            {sampleLabel && !file ? (
                              <p className="text-xs text-primary font-medium">
                                Đã chọn ảnh mẫu: {sampleLabel}
                              </p>
                            ) : null}
                          </div>
                        )}

                        {/* Ảnh chuẩn (Ground Truth) */}
                        <div className="grid gap-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">
                              Ảnh chuẩn (Ground Truth - tùy chọn)
                            </span>
                            {groundTruth && (
                              <button
                                type="button"
                                onClick={handleRemoveGroundTruth}
                                className="text-xs text-muted-foreground hover:text-foreground"
                              >
                                Xóa
                              </button>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Ảnh chuẩn dùng để đo PSNR/SSIM, phải cùng kích thước với ảnh gốc.
                          </p>
                          <div className="flex items-center gap-2">
                            <input
                              type="file"
                              ref={gtInputRef}
                              accept={ACCEPTED_TYPES.join(",")}
                              className="hidden"
                              aria-label="Chọn tệp ảnh chuẩn"
                              onChange={(e) => {
                                attachGroundTruth(e.target.files?.[0]);
                              }}
                            />
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => gtInputRef.current?.click()}
                            >
                              {groundTruth ? "Đổi ảnh chuẩn" : "Chọn tệp ảnh chuẩn"}
                            </Button>
                            {groundTruth && (
                              <span className="text-xs text-muted-foreground truncate max-w-48">
                                {groundTruth.name}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>

                      <DialogFooter>
                        <DialogClose asChild>
                          <Button type="button">Xong</Button>
                        </DialogClose>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </>
              )}

              {/* Đếm ký tự */}
              {text.length > 0 && (
                <span
                  className={cn(
                    "text-[11px] text-muted-foreground pl-1 select-none",
                    text.length >= 900 && "text-warning font-medium",
                    text.length >= CHAT_INPUT_MAX_LENGTH && "text-destructive font-medium",
                  )}
                >
                  {text.length}/{CHAT_INPUT_MAX_LENGTH}
                </span>
              )}
            </div>

            {/* Nhóm thao tác bên phải: Nút Gửi */}
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="icon-sm"
                disabled={!canSubmit}
                onClick={() => void handleSubmit()}
                aria-label={submitButtonLabel}
                className="size-8 rounded-full shadow-xs transition-transform active:scale-95"
              >
                {busy ? (
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                ) : (
                  <SendHorizontal className="size-4" aria-hidden />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
