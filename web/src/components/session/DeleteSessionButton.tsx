import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

import { errorMessage } from "@/api/client";
import { SESSIONS_KEY, deleteSession, sessionKey } from "@/api/workspace";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";

export function DeleteSessionButton({
  sessionId,
  disabled,
}: {
  sessionId: string;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => deleteSession(sessionId),
    onSuccess: () => {
      setOpen(false);
      queryClient.removeQueries({ queryKey: sessionKey(sessionId) });
      void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
      toast.success("Đã xóa phiên xử lý.");
      navigate("/", { replace: true });
    },
    onError: (error) => toast.error(errorMessage(error)),
  });

  return (
    <AlertDialog open={open} onOpenChange={(next) => !mutation.isPending && setOpen(next)}>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
          <Trash2 aria-hidden />
          Xóa
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Xóa phiên xử lý?</AlertDialogTitle>
          <AlertDialogDescription>
            Xóa phiên và mọi ảnh trên đĩa? Không thể hoàn tác.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={mutation.isPending}>Hủy</AlertDialogCancel>
          {/* Không dùng AlertDialogAction: nó tự đóng dialog trước khi xóa xong */}
          <Button
            variant="destructive"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
            Xóa phiên
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
