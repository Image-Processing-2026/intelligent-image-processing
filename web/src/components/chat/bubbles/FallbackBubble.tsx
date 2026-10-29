import type { Message } from "@/api/types";

/** Loại tin nhắn chưa có giao diện riêng: hiện văn bản an toàn để giao diện không bao giờ lỗi. */
export function FallbackBubble({ message }: { message: Message }) {
  const text = typeof message.content?.text === "string" ? message.content.text : "";
  if (!text) return null;
  return (
    <div className="rounded-lg border px-4 py-3 text-sm">
      <p className="whitespace-pre-wrap break-words">{text}</p>
    </div>
  );
}
