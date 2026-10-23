import { ImageUp, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
// Khớp mặc định MAX_UPLOAD_MB của backend; backend vẫn là nơi kiểm tra cuối
export const MAX_UPLOAD_MB = 20;

/** Kiểm tra sớm phía client; trả câu lỗi tiếng Việt hoặc null. */
export function checkFile(file: File): string | null {
  if (file.type && !ACCEPTED_TYPES.includes(file.type)) {
    return "Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP.";
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `Tệp vượt quá ${MAX_UPLOAD_MB} MB.`;
  return null;
}

/** URL tạm để xem trước file đã chọn; tự thu hồi khi đổi file. */
export function useObjectUrl(file: File | null | undefined): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url;
}

interface DropzoneProps {
  label: string;
  hint?: string;
  file: File | null;
  onFile: (file: File | null) => void;
  disabled?: boolean;
  compact?: boolean;
}

export function Dropzone({ label, hint, file, onFile, disabled, compact }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const preview = useObjectUrl(file);
  const errorId = useId();

  const accept = (candidate: File | undefined) => {
    if (!candidate) return;
    const problem = checkFile(candidate);
    setError(problem);
    if (!problem) onFile(candidate);
  };

  return (
    <div className="grid gap-2">
      <div
        onDragOver={(event) => {
          if (disabled) return;
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) accept(event.dataTransfer.files?.[0]);
        }}
        className={cn(
          "relative flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-4 text-center transition-colors",
          compact ? "min-h-32" : "min-h-56",
          dragging ? "border-primary bg-accent" : "border-input",
          disabled && "opacity-60",
        )}
      >
        {preview ? (
          <>
            <img
              src={preview}
              alt={`Xem trước: ${file?.name ?? ""}`}
              className={cn("w-full rounded object-contain", compact ? "max-h-32" : "max-h-72")}
            />
            <div className="flex max-w-full items-center gap-2 text-sm">
              <span className="truncate">{file?.name}</span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-7"
                aria-label={`Bỏ chọn ${file?.name ?? "ảnh"}`}
                onClick={() => {
                  onFile(null);
                  setError(null);
                }}
                disabled={disabled}
              >
                <X aria-hidden />
              </Button>
            </div>
          </>
        ) : (
          <>
            <ImageUp className="size-8 text-muted-foreground" aria-hidden />
            <div className="text-sm">
              <p className="font-medium">{label}</p>
              {hint ? <p className="text-muted-foreground">{hint}</p> : null}
            </div>
            <Button
              type="button"
              variant="outline"
              onClick={() => inputRef.current?.click()}
              disabled={disabled}
              aria-describedby={error ? errorId : undefined}
            >
              Chọn ảnh
            </Button>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          className="sr-only"
          tabIndex={-1}
          aria-label={label}
          onChange={(event) => {
            accept(event.target.files?.[0]);
            event.target.value = "";
          }}
          disabled={disabled}
        />
      </div>
      {error ? (
        <p id={errorId} className="text-sm text-destructive" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
