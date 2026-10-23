import { Check, Download, Loader2, Wand2 } from "lucide-react";

import type { Artifact, ProcessResult, Treatment, Variant } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RANKING_SOURCE_LABELS, RECOMMENDED_REASON_LABELS, describeAction } from "@/lib/actions";
import { formatNumber, label } from "@/lib/labels";
import { cn } from "@/lib/utils";

interface VariantGridProps {
  variants: Variant[];
  process: ProcessResult;
  treatments: Treatment[];
  artifactById: Map<string, Artifact>;
  onRender: (variant: Variant) => void;
  renderingId: string | null;
  disabled: boolean;
}

/** Ảnh full-res mới nhất đã render cho một phiên bản (nếu có). */
function renderedArtifact(
  variant: Variant,
  treatments: Treatment[],
  artifactById: Map<string, Artifact>,
): Artifact | null {
  const latest = [...treatments]
    .filter((t) => t.source === "variant" && t.variant_id === variant.id && t.artifact_id)
    .sort((a, b) => b.version - a.version)[0];
  return latest?.artifact_id ? (artifactById.get(latest.artifact_id) ?? null) : null;
}

export function VariantGrid({
  variants,
  process,
  treatments,
  artifactById,
  onRender,
  renderingId,
  disabled,
}: VariantGridProps) {
  if (variants.length === 0) return null;
  const reason = label(RECOMMENDED_REASON_LABELS, process.recommended_reason);
  return (
    <section className="grid gap-3" aria-labelledby="variants-heading">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 id="variants-heading" className="font-medium">
          Các phiên bản
        </h3>
        {process.variant_ranking_source ? (
          <span className="text-xs text-muted-foreground">
            {label(RANKING_SOURCE_LABELS, process.variant_ranking_source)}
          </span>
        ) : null}
      </div>
      <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {variants.map((variant) => {
          const preview = variant.preview_artifact_id
            ? artifactById.get(variant.preview_artifact_id)
            : undefined;
          const rendered = renderedArtifact(variant, treatments, artifactById);
          const rendering = renderingId === variant.id;
          return (
            <li
              key={variant.id}
              className={cn(
                "grid content-start gap-2 rounded-lg border p-3 text-sm",
                variant.recommended && "border-primary/60",
                variant.chosen && "ring-2 ring-primary/40",
              )}
            >
              {preview ? (
                <img
                  src={preview.url}
                  alt={`Xem trước phiên bản ${variant.label}`}
                  className="aspect-[4/3] w-full rounded bg-muted object-contain"
                  loading="lazy"
                />
              ) : null}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="font-medium">{variant.label}</span>
                {variant.rank !== null ? (
                  <Badge variant="outline" className="font-normal">
                    Hạng {variant.rank}
                  </Badge>
                ) : null}
                {variant.recommended ? <Badge className="font-normal">Đề xuất</Badge> : null}
                {variant.chosen ? (
                  <Badge variant="outline" className="border-primary/50 font-normal text-primary">
                    Đã chọn
                  </Badge>
                ) : null}
              </div>
              {variant.recommended && process.recommended_reason ? (
                <p className="text-xs text-muted-foreground">Lý do đề xuất: {reason}</p>
              ) : null}
              {variant.description ? (
                <p className="text-muted-foreground">{variant.description}</p>
              ) : null}
              <p className="text-xs text-muted-foreground">
                Điểm ước lượng (trên ảnh xem trước): {formatNumber(variant.quality_score, 1)}
                {variant.quality_improved === false ? " · không cải thiện so với ảnh gốc" : ""}
              </p>
              {variant.critic_note ? (
                <p className="text-xs">Nhận xét: {variant.critic_note}</p>
              ) : null}
              <details className="text-xs text-muted-foreground">
                <summary className="cursor-pointer">Phác đồ ({variant.actions.length} bước)</summary>
                <ol className="mt-1 list-inside list-decimal">
                  {variant.actions.map((action, index) => (
                    <li key={index}>{describeAction(action)}</li>
                  ))}
                </ol>
              </details>
              <p className="text-[11px] text-muted-foreground">Xem trước, cạnh dài ≤ 1024px</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  variant={variant.chosen ? "outline" : "default"}
                  onClick={() => onRender(variant)}
                  disabled={disabled}
                >
                  {rendering ? (
                    <Loader2 className="animate-spin" aria-hidden />
                  ) : variant.chosen ? (
                    <Check aria-hidden />
                  ) : (
                    <Wand2 aria-hidden />
                  )}
                  {variant.chosen ? "Render lại" : "Chọn và render"}
                </Button>
                {rendered ? (
                  <Button asChild size="sm" variant="outline">
                    <a href={rendered.download_url} download>
                      <Download aria-hidden />
                      Tải về {rendered.width}×{rendered.height}
                    </a>
                  </Button>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
