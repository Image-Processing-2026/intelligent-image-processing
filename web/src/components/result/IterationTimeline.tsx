import type { Artifact, HistoryItem } from "@/api/types";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { describeAction } from "@/lib/actions";
import { DEFECT_LABELS, formatNumber, label, regionLabel } from "@/lib/labels";

import { SeverityBadge } from "../session/DiagnosisCard";
import { DecisionBadge, SourceBadge } from "../session/StatusBadge";

// Chỉ số kỹ thuật của Module 1 hiển thị trước/sau (tên khóa đúng như TechnicalMetrics)
const METRIC_ROWS: { key: string; label: string; digits: number }[] = [
  { key: "brightness_mean", label: "Độ sáng trung bình", digits: 1 },
  { key: "contrast_std", label: "Tương phản (độ lệch chuẩn)", digits: 1 },
  { key: "noise_variance", label: "Phương sai nhiễu", digits: 2 },
  { key: "sharpness_laplacian_var", label: "Độ nét (Laplacian)", digits: 1 },
];

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function EvalLine({ score }: { score: Record<string, unknown> }) {
  const psnr = numberOrNull(score.psnr);
  const ssim = numberOrNull(score.ssim);
  const estimated = numberOrNull(score.estimated_quality_score);
  if (psnr !== null || ssim !== null) {
    return (
      <p className="text-sm">
        So với ảnh chuẩn: PSNR {formatNumber(psnr, 2)} dB · SSIM {formatNumber(ssim, 4)}
      </p>
    );
  }
  return <p className="text-sm">Điểm đánh giá ước lượng: {formatNumber(estimated, 1)}</p>;
}

export function IterationTimeline({
  history,
  intermediates,
}: {
  history: HistoryItem[];
  intermediates: (Artifact | undefined)[];
}) {
  if (history.length === 0) return null;
  return (
    <section className="grid gap-2" aria-labelledby="timeline-heading">
      <h3 id="timeline-heading" className="font-medium">
        Các vòng xử lý
      </h3>
      <Accordion type="multiple" className="rounded-lg border px-3">
        {history.map((item, index) => {
          const image = intermediates[index];
          const defects = (item.diagnosis?.defects ?? []).filter((d) => d.severity >= 1);
          return (
            <AccordionItem key={item.iteration} value={String(item.iteration)}>
              <AccordionTrigger className="gap-2">
                <span className="flex flex-wrap items-center gap-2 text-left">
                  <span>Vòng {item.iteration}</span>
                  <DecisionBadge decision={item.decision} />
                  {item.rolled_back ? (
                    <Badge variant="outline" className="border-warning/50 font-normal text-warning">
                      Đã rollback
                    </Badge>
                  ) : null}
                </span>
              </AccordionTrigger>
              <AccordionContent className="grid gap-4 text-sm">
                <div className="flex flex-wrap gap-2">
                  {item.diagnosis ? (
                    <SourceBadge prefix="Chẩn đoán" source={item.diagnosis.source} />
                  ) : null}
                  <SourceBadge prefix="Phác đồ" source={item.plan.source} />
                </div>

                {item.diagnosis ? (
                  <div className="grid gap-1">
                    <p className="font-medium">Chẩn đoán vòng này</p>
                    {item.diagnosis.summary ? (
                      <p className="text-muted-foreground">{item.diagnosis.summary}</p>
                    ) : null}
                    {defects.length > 0 ? (
                      <ul className="flex flex-wrap gap-2">
                        {defects.map((defect, i) => (
                          <li key={i} className="flex items-center gap-1">
                            <span>
                              {label(DEFECT_LABELS, defect.type)} @ {regionLabel(defect.region)}
                            </span>
                            <SeverityBadge severity={defect.severity} />
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-muted-foreground">Không còn lỗi cần xử lý.</p>
                    )}
                  </div>
                ) : null}

                <div className="grid gap-1">
                  <p className="font-medium">Phác đồ</p>
                  {item.plan.actions.length > 0 ? (
                    <ol className="list-inside list-decimal">
                      {item.plan.actions.map((action, i) => (
                        <li key={i}>{describeAction(action)}</li>
                      ))}
                    </ol>
                  ) : (
                    <p className="text-muted-foreground">Không có thao tác.</p>
                  )}
                  {item.plan.reasoning ? (
                    <p className="text-muted-foreground">Lý do: {item.plan.reasoning}</p>
                  ) : null}
                  {item.plan.knowledge.length > 0 ? (
                    <p className="text-xs text-muted-foreground">
                      Thẻ tri thức:{" "}
                      {item.plan.knowledge.map((id) => (
                        <code key={id} className="mr-1 rounded bg-muted px-1">
                          {id}
                        </code>
                      ))}
                    </p>
                  ) : null}
                </div>

                <div className="overflow-x-auto rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Chỉ số</TableHead>
                        <TableHead>Trước</TableHead>
                        <TableHead>Sau</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {METRIC_ROWS.map((row) => (
                        <TableRow key={row.key}>
                          <TableCell>{row.label}</TableCell>
                          <TableCell>
                            {formatNumber(numberOrNull(item.metrics_before[row.key]), row.digits)}
                          </TableCell>
                          <TableCell>
                            {formatNumber(numberOrNull(item.metrics_after[row.key]), row.digits)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <EvalLine score={item.eval_score} />

                {image ? (
                  <img
                    src={image.url}
                    alt={`Ảnh sau vòng ${item.iteration}`}
                    className="max-h-64 w-full rounded bg-muted object-contain"
                    loading="lazy"
                  />
                ) : null}
              </AccordionContent>
            </AccordionItem>
          );
        })}
      </Accordion>
    </section>
  );
}
