import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { type NewSessionInput, SESSIONS_KEY, createSession, sessionKey } from "@/api/workspace";
import { WaitingNotice } from "@/components/WaitingNotice";
import { ChatInputBar } from "@/components/chat/ChatInputBar";

/** Màn hình chat trống: dán / chọn ảnh kèm yêu cầu (tùy chọn) rồi gửi để tạo phiên. */
export default function NewChatPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [pending, setPending] = useState(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const onNewSession = async (input: NewSessionInput) => {
    setPending(true);
    // Server ghi phiên ngay khi nhận ảnh (còn đang chẩn đoán): tải lại danh sách để sidebar hiện
    // phiên mới ngay, không phải chờ chẩn đoán xong
    const timer = window.setInterval(
      () => void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY }),
      1000,
    );
    try {
      const detail = await createSession(input);
      queryClient.setQueryData(sessionKey(detail.id), detail);
      void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
      // Người dùng đã sang phiên khác trong lúc chờ thì không kéo họ về
      if (mounted.current) navigate(`/s/${detail.id}`);
    } finally {
      window.clearInterval(timer);
      if (mounted.current) setPending(false);
    }
  };

  return (
    <div className="absolute inset-0 flex min-h-0 flex-col overflow-hidden">
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 overflow-y-auto p-6 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Phiên mới</h1>
        <p className="max-w-md text-sm text-muted-foreground">
          Dán hoặc kéo thả một ảnh, mô tả điều bạn muốn chỉnh (tùy chọn) rồi gửi. AI Image Doctor sẽ
          chẩn đoán trước khi xử lý.
        </p>
        {pending ? <WaitingNotice text="AI đang phân tích ảnh và yêu cầu của bạn..." /> : null}
      </div>
      <div className="border-t bg-background p-3">
        <div className="mx-auto max-w-3xl">
          <ChatInputBar mode="new" autoFocus onNewSession={onNewSession} />
        </div>
      </div>
    </div>
  );
}
