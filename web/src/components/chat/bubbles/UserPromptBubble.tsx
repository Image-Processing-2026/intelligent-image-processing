import type { Artifact } from "@/api/types";

/** Lượt đầu của người dùng: ảnh gốc đính kèm và yêu cầu ban đầu (nếu có). */
export function UserPromptBubble({
  text,
  original,
}: {
  text: string;
  original: Artifact | null;
}) {
  return (
    <div className="flex justify-end">
      <div className="grid max-w-[85%] gap-2 rounded-lg bg-accent px-4 py-3 text-sm text-accent-foreground">
        {original ? (
          <img
            src={original.url}
            alt="Ảnh gốc đính kèm"
            width={original.width}
            height={original.height}
            className="max-h-64 w-auto rounded-md bg-muted object-contain"
          />
        ) : null}
        {text ? <p className="whitespace-pre-wrap break-words">{text}</p> : null}
      </div>
    </div>
  );
}
