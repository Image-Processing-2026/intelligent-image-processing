import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, Save } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { ME_QUERY_KEY, updateSettings, useMe } from "@/api/auth";
import { errorMessage } from "@/api/client";
import { useSystemInfo } from "@/api/workspace";
import { modeLabel } from "@/components/layout/BackendStatus";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { type ThemeChoice, useTheme } from "@/lib/theme";

const ITERATIONS = [1, 2, 3, 4, 5];
const VARIANTS = [1, 2, 3];
const THEMES: { value: ThemeChoice; label: string }[] = [
  { value: "light", label: "Sáng" },
  { value: "dark", label: "Tối" },
  { value: "system", label: "Theo hệ thống" },
];

function Choice({
  id,
  label,
  value,
  options,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="grid gap-2">
      <span id={id} className="text-sm font-medium">
        {label}
      </span>
      <ToggleGroup
        type="single"
        variant="outline"
        value={value}
        onValueChange={(next) => next && onChange(next)}
        aria-labelledby={id}
        className="w-fit flex-wrap"
      >
        {options.map((option) => (
          <ToggleGroupItem key={option.value} value={option.value} className="min-w-10 px-3">
            {option.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

export default function SettingsPage() {
  const me = useMe();
  const system = useSystemInfo();
  const queryClient = useQueryClient();
  const { choice, setChoice } = useTheme();
  const saved = me.data?.settings ?? { max_iterations: 3, num_variants: 3 };
  const [iterations, setIterations] = useState(saved.max_iterations);
  const [variants, setVariants] = useState(saved.num_variants);
  const dirty = iterations !== saved.max_iterations || variants !== saved.num_variants;

  const mutation = useMutation({
    mutationFn: () => updateSettings({ max_iterations: iterations, num_variants: variants }),
    onSuccess: (user) => {
      queryClient.setQueryData(ME_QUERY_KEY, user);
      toast.success("Đã lưu cài đặt mặc định.");
    },
    onError: (error) => toast.error(errorMessage(error)),
  });

  return (
    <div className="mx-auto grid max-w-3xl gap-4 p-4 sm:p-6">
      <h1 className="text-xl font-semibold">Cài đặt</h1>

      <Card>
        <CardHeader>
          <CardTitle>
            <h2 className="text-base">Mặc định cho phiên mới</h2>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form
            className="grid gap-5"
            onSubmit={(event) => {
              event.preventDefault();
              if (dirty) mutation.mutate();
            }}
          >
            <Choice
              id="default-iterations"
              label="Số vòng tối đa"
              value={String(iterations)}
              options={ITERATIONS.map((n) => ({ value: String(n), label: String(n) }))}
              onChange={(value) => setIterations(Number(value))}
            />
            <Choice
              id="default-variants"
              label="Số phiên bản"
              value={String(variants)}
              options={VARIANTS.map((n) => ({ value: String(n), label: String(n) }))}
              onChange={(value) => setVariants(Number(value))}
            />
            <div>
              <Button type="submit" size="sm" disabled={!dirty || mutation.isPending}>
                {mutation.isPending ? (
                  <Loader2 className="animate-spin" aria-hidden />
                ) : (
                  <Save aria-hidden />
                )}
                Lưu mặc định
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            <h2 className="text-base">Giao diện</h2>
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2">
          <Choice
            id="theme-choice"
            label="Chủ đề màu"
            value={choice}
            options={THEMES}
            onChange={(value) => setChoice(value as ThemeChoice)}
          />
          <p className="text-xs text-muted-foreground">Lưu trên trình duyệt này.</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            <h2 className="text-base">Hệ thống</h2>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {system.data ? (
            <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
              <dt className="text-muted-foreground">Backend</dt>
              <dd>Hoạt động</dd>
              <dt className="text-muted-foreground">Chế độ</dt>
              <dd>{modeLabel(system.data.gemini_key_configured)}</dd>
              <dt className="text-muted-foreground">Model cấu hình</dt>
              <dd>
                <code>{system.data.gemini_model}</code>
              </dd>
              <dt className="text-muted-foreground">Bộ nhớ ca bệnh</dt>
              <dd>{system.data.memory_enabled ? "Bật" : "Tắt"}</dd>
              <dt className="text-muted-foreground">Pipeline chạy đồng thời</dt>
              <dd>{system.data.pipeline_concurrency}</dd>
              <dt className="text-muted-foreground">Dung lượng ảnh tối đa</dt>
              <dd>{system.data.max_upload_mb} MB</dd>
            </dl>
          ) : system.isError ? (
            <p role="alert" className="text-sm">
              {errorMessage(system.error)}
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">Đang tải…</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
