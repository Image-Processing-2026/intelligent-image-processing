import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Loader2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useMe } from "@/api/auth";
import { errorMessage } from "@/api/client";
import type { SessionDetail } from "@/api/types";
import { SESSIONS_KEY, createSession, sessionKey, useSamples } from "@/api/workspace";
import { Dropzone } from "@/components/new-session/Dropzone";
import { WaitingNotice } from "@/components/WaitingNotice";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";

const ITERATION_CHOICES = [1, 2, 3, 4, 5];
const VARIANT_CHOICES = [1, 2, 3];

function NumberChoice({
  id,
  label,
  hint,
  value,
  choices,
  onChange,
}: {
  id: string;
  label: string;
  hint: string;
  value: number;
  choices: number[];
  onChange: (value: number) => void;
}) {
  return (
    <div className="grid gap-2">
      <span id={id} className="text-sm font-medium">
        {label}
      </span>
      <ToggleGroup
        type="single"
        variant="outline"
        value={String(value)}
        onValueChange={(next) => next && onChange(Number(next))}
        aria-labelledby={id}
        className="w-fit"
      >
        {choices.map((choice) => (
          <ToggleGroupItem key={choice} value={String(choice)} className="min-w-10">
            {choice}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

export default function NewSessionPage() {
  const me = useMe();
  const samples = useSamples();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [sample, setSample] = useState<string | null>(null);
  const [groundTruth, setGroundTruth] = useState<File | null>(null);
  const [iterations, setIterations] = useState(me.data?.settings.max_iterations ?? 3);
  const [variants, setVariants] = useState(me.data?.settings.num_variants ?? 3);

  const mutation = useMutation({
    mutationFn: () =>
      createSession({
        file,
        sample: file ? null : sample,
        groundTruth,
        maxIterations: iterations,
        numVariants: variants,
      }),
    onSuccess: (detail: SessionDetail) => {
      queryClient.setQueryData(sessionKey(detail.id), detail);
      void queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
      navigate(`/s/${detail.id}`);
    },
  });

  const busy = mutation.isPending;
  const ready = Boolean(file || sample) && !busy;
  const sampleLabel = samples.data?.find((item) => item.id === sample)?.label;

  return (
    <div className="mx-auto grid max-w-3xl gap-6 p-4 sm:p-6">
      <div>
        <h1 className="text-xl font-semibold">Phiên mới</h1>
        <p className="text-sm text-muted-foreground">
          Chọn một ảnh để hệ thống chẩn đoán, hỏi ý định của bạn rồi xử lý bằng thuật toán cổ điển.
        </p>
      </div>

      <form
        className="grid gap-6"
        onSubmit={(event) => {
          event.preventDefault();
          if (ready) mutation.mutate();
        }}
      >
        <fieldset disabled={busy} className="grid gap-6">
          <legend className="sr-only">Ảnh đầu vào</legend>
          <Dropzone
            label="Kéo thả ảnh vào đây"
            hint="JPEG, PNG hoặc WEBP, tối đa 20 MB"
            file={file}
            onFile={(next) => {
              setFile(next);
              if (next) setSample(null);
            }}
            disabled={busy}
          />

          {samples.data && samples.data.length > 0 ? (
            <div className="grid gap-2">
              <span className="text-sm font-medium">Hoặc dùng ảnh mẫu</span>
              <div className="grid grid-cols-3 gap-2 sm:max-w-md">
                {samples.data.map((item) => {
                  const selected = sample === item.id && !file;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => {
                        setSample(selected ? null : item.id);
                        setFile(null);
                      }}
                      className={cn(
                        "grid gap-1 rounded-md border p-1.5 text-left text-xs transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring",
                        selected && "border-primary ring-2 ring-primary/40",
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
                <p className="text-xs text-muted-foreground">Đã chọn ảnh mẫu: {sampleLabel}</p>
              ) : null}
            </div>
          ) : null}

          <Accordion type="single" collapsible className="rounded-lg border px-4">
            <AccordionItem value="options" className="border-b-0">
              <AccordionTrigger>Tùy chọn</AccordionTrigger>
              <AccordionContent className="grid gap-5">
                <NumberChoice
                  id="iterations-label"
                  label="Số vòng tối đa"
                  hint="Mỗi vòng: chẩn đoán → lập phác đồ → xử lý → đánh giá."
                  value={iterations}
                  choices={ITERATION_CHOICES}
                  onChange={setIterations}
                />
                <NumberChoice
                  id="variants-label"
                  label="Số phiên bản"
                  hint="Số phong cách kết quả để bạn chọn (cân bằng / tự nhiên / đậm nét)."
                  value={variants}
                  choices={VARIANT_CHOICES}
                  onChange={setVariants}
                />
                <div className="grid gap-2">
                  <span className="text-sm font-medium">
                    Chế độ nâng cao: ảnh chuẩn (ground truth) để đo PSNR/SSIM
                  </span>
                  <p className="text-xs text-muted-foreground">
                    Chỉ dùng khi có ảnh chuẩn tương ứng để đánh giá thử nghiệm; ảnh chuẩn phải
                    cùng kích thước ảnh gốc. Khi có ảnh chuẩn, chẩn đoán vòng 1 chạy hai lần.
                  </p>
                  <Dropzone
                    label="Ảnh chuẩn (tùy chọn)"
                    file={groundTruth}
                    onFile={setGroundTruth}
                    disabled={busy}
                    compact
                  />
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </fieldset>

        {mutation.isError ? (
          <Alert variant="destructive">
            <AlertCircle aria-hidden />
            <AlertDescription>{errorMessage(mutation.error)}</AlertDescription>
          </Alert>
        ) : null}
        {busy ? <WaitingNotice /> : null}

        <div>
          <Button type="submit" disabled={!ready}>
            {busy ? <Loader2 className="animate-spin" aria-hidden /> : null}
            Bắt đầu chẩn đoán
          </Button>
        </div>
      </form>
    </div>
  );
}
