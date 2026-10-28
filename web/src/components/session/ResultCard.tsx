import { Download } from "lucide-react";

import type { ProcessResult, SessionDetail, Variant } from "@/api/types";
import { CompareSlider } from "@/components/result/CompareSlider";
import { IterationTimeline } from "@/components/result/IterationTimeline";
import { TechnicalJson } from "@/components/result/TechnicalJson";
import { VariantGrid } from "@/components/result/VariantGrid";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { treatmentLabel } from "@/lib/actions";
import { formatNumber } from "@/lib/labels";

import { DecisionBadge, OutcomeBadge, SourceBadge } from "./StatusBadge";

/** Nguồn thật theo vòng; ghi chú riêng khi có vòng dùng luật dự phòng vì gọi Gemini lỗi. */
export function RoundSources({ process }: { process: ProcessResult }) {
  const { diagnosis, plan } = process.sources;
  const fallback = [...diagnosis, ...plan].includes("vlm_fallback");
  return (
    <div className="grid gap-2">
      <ul className="grid gap-1.5">
        {plan.map((source, index) => (
          <li key={index} className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted-foreground">Vòng {index + 1}:</span>
            {diagnosis[index] ? <SourceBadge prefix="Chẩn đoán" source={diagnosis[index]} /> : null}
            {source ? <SourceBadge prefix="Phác đồ" source={source} /> : null}
          </li>
        ))}
      </ul>
      {fallback ? (
        <p className="text-xs text-warning">Gọi Gemini lỗi, đã dùng luật dự phòng.</p>
      ) : null}
    </div>
  );
}

/** Điểm hiển thị trung thực: điểm ước lượng (ảnh thật) hoặc PSNR/SSIM so với ảnh chuẩn. */
export function EvaluationSummary({ process }: { process: ProcessResult }) {
  const evaluation = process.final_evaluation ?? {};
  if (process.is_synthetic) {
    return (
      <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-sm">
        <dt className="text-muted-foreground">PSNR (so với ảnh chuẩn)</dt>
        <dd>{formatNumber(evaluation.psnr as number, 2)} dB</dd>
        <dt className="text-muted-foreground">SSIM (so với ảnh chuẩn)</dt>
        <dd>{formatNumber(evaluation.ssim as number, 4)}</dd>
        {evaluation.mse !== undefined && evaluation.mse !== null ? (
          <>
            <dt className="text-muted-foreground">MSE</dt>
            <dd>{formatNumber(evaluation.mse as number, 2)}</dd>
          </>
        ) : null}
      </dl>
    );
  }
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-sm">
      <dt className="text-muted-foreground">Điểm đánh giá ước lượng</dt>
      <dd>{formatNumber(evaluation.estimated_quality_score as number, 1)}</dd>
      {evaluation.brisque_score !== undefined && evaluation.brisque_score !== null ? (
        <>
          <dt className="text-muted-foreground">BRISQUE</dt>
          <dd>{formatNumber(evaluation.brisque_score as number, 2)}</dd>
        </>
      ) : null}
      {evaluation.niqe_score !== undefined && evaluation.niqe_score !== null ? (
        <>
          <dt className="text-muted-foreground">NIQE</dt>
          <dd>{formatNumber(evaluation.niqe_score as number, 2)}</dd>
        </>
      ) : null}
    </dl>
  );
}

interface ResultCardProps {
  detail: SessionDetail;
  process: ProcessResult;
  onRender: (variant: Variant) => void;
  renderingId: string | null;
  busy: boolean;
  /** Khu tinh chỉnh (S5) chèn vào giữa phần so sánh và phần chi tiết. */
  children?: React.ReactNode;
}

export function ResultCard({
  detail,
  process,
  onRender,
  renderingId,
  busy,
  children,
}: ResultCardProps) {
  const artifactById = new Map(detail.artifacts.map((artifact) => [artifact.id, artifact]));
  const final = artifactById.get(process.artifacts.final) ?? null;
  const active = detail.treatments.find((t) => t.id === detail.active_treatment_id) ?? null;
  const shown = (active?.artifact_id ? artifactById.get(active.artifact_id) : null) ?? final;
  const shownLabel = active ? treatmentLabel(active, detail.variants) : "Kết quả pipeline";
  const intermediates = process.artifacts.intermediate.map((id) => artifactById.get(id));

  return (
    <Card>
      <CardHeader className="flex flex-wrap items-center justify-between gap-2">
        <CardTitle>
          <h2 className="text-base">Kết quả xử lý</h2>
        </CardTitle>
        <div className="flex flex-wrap items-center gap-2">
          {process.outcome ? (
            <OutcomeBadge outcome={process.outcome} />
          ) : (
            <DecisionBadge decision={process.final_decision} />
          )}
          <span className="text-xs text-muted-foreground">
            {process.total_iterations} vòng đã chạy
            {process.rolled_back ? " · vòng cuối đã rollback" : ""}
          </span>
        </div>
        {process.outcome?.reason ? (
          <p className="basis-full text-xs text-muted-foreground">{process.outcome.reason}</p>
        ) : null}
      </CardHeader>
      <CardContent className="grid gap-6">
        {detail.original && shown ? (
          <div className="grid gap-2">
            <p className="text-sm">
              Đang xem: <span className="font-medium">{shownLabel}</span>
            </p>
            <CompareSlider before={detail.original} after={shown} afterLabel={shownLabel} />
            <div>
              <Button asChild variant="outline" size="sm">
                <a href={shown.download_url} download>
                  <Download aria-hidden />
                  Tải ảnh đang xem ({shown.width}×{shown.height})
                </a>
              </Button>
            </div>
          </div>
        ) : null}

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="grid content-start gap-1">
            <p className="text-xs text-muted-foreground">Đánh giá của kết quả pipeline (v1)</p>
            <EvaluationSummary process={process} />
          </div>
          <RoundSources process={process} />
        </div>

        {children}

        <VariantGrid
          variants={detail.variants}
          process={process}
          treatments={detail.treatments}
          artifactById={artifactById}
          onRender={onRender}
          renderingId={renderingId}
          disabled={busy}
        />

        <IterationTimeline history={process.history} intermediates={intermediates} />

        <TechnicalJson
          data={{ process, diagnosis: detail.diagnosis, active_treatment: active }}
        />
      </CardContent>
    </Card>
  );
}
