import { Download, Undo2 } from "lucide-react";

import type { Artifact, Treatment, Variant } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ADJUSTMENT_LABELS, REFINE_SOURCE_LABELS, treatmentLabel } from "@/lib/actions";
import { formatDateTime, label } from "@/lib/labels";

/** Lịch sử các phiên bản phác đồ (v1, v2…): xem nguồn, góp ý, quay về một bản trước. */
export function TreatmentHistory({
  treatments,
  variants,
  activeId,
  artifactById,
  onActivate,
  disabled,
}: {
  treatments: Treatment[];
  variants: Variant[];
  activeId: string | null;
  artifactById: Map<string, Artifact>;
  onActivate: (treatment: Treatment) => void;
  disabled: boolean;
}) {
  if (treatments.length === 0) return null;
  const ordered = [...treatments].sort((a, b) => b.version - a.version);
  const byId = new Map(treatments.map((t) => [t.id, t]));
  return (
    <section className="grid gap-2" aria-labelledby="treatment-history-heading">
      <h3 id="treatment-history-heading" className="font-medium">
        Lịch sử phiên bản phác đồ
      </h3>
      <ol className="grid gap-2">
        {ordered.map((treatment) => {
          const active = treatment.id === activeId;
          const artifact = treatment.artifact_id ? artifactById.get(treatment.artifact_id) : null;
          const parent = treatment.parent_id ? byId.get(treatment.parent_id) : null;
          return (
            <li
              key={treatment.id}
              className="grid gap-1 rounded-md border p-3 text-sm"
              aria-current={active ? "true" : undefined}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{treatmentLabel(treatment, variants)}</span>
                {active ? <Badge className="font-normal">Đang dùng</Badge> : null}
                {parent ? (
                  <span className="text-xs text-muted-foreground">từ v{parent.version}</span>
                ) : null}
                <time
                  className="ml-auto text-xs text-muted-foreground"
                  dateTime={treatment.created_at}
                >
                  {formatDateTime(treatment.created_at)}
                </time>
              </div>
              {treatment.feedback_text ? (
                <p>
                  Góp ý: “{treatment.feedback_text}”
                  {treatment.refine_source ? (
                    <span className="text-muted-foreground">
                      {" "}
                      ({label(REFINE_SOURCE_LABELS, treatment.refine_source)})
                    </span>
                  ) : null}
                </p>
              ) : null}
              {treatment.adjustments.length > 0 ? (
                <p className="text-xs text-muted-foreground">
                  Điều chỉnh:{" "}
                  {treatment.adjustments
                    .map((a) => `${label(ADJUSTMENT_LABELS, a.kind)} (mức ${a.strength})`)
                    .join(", ")}
                </p>
              ) : null}
              <div className="flex flex-wrap gap-2 pt-1">
                {!active ? (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onActivate(treatment)}
                    disabled={disabled}
                  >
                    <Undo2 aria-hidden />
                    Dùng bản này
                  </Button>
                ) : null}
                {artifact ? (
                  <Button asChild size="sm" variant="ghost">
                    <a href={artifact.download_url} download>
                      <Download aria-hidden />
                      Tải v{treatment.version}
                    </a>
                  </Button>
                ) : null}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
