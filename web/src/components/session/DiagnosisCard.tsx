import type { Artifact, DiagnosisReport } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  BRIGHTNESS_LEVEL_LABELS,
  DEFECT_LABELS,
  ORIGIN_LABELS,
  PRESERVE_LABELS,
  SCENE_LABELS,
  SEVERITY_LABELS,
  formatNumber,
  formatPercent,
  label,
  regionLabel,
} from "@/lib/labels";
import { cn } from "@/lib/utils";

import { SourceBadge } from "./StatusBadge";

const SEVERITY_STYLES = [
  "border-border text-muted-foreground",
  "border-primary/40 text-primary",
  "border-warning/50 text-warning",
  "border-destructive/50 text-destructive",
];

export function SeverityBadge({ severity }: { severity: number }) {
  const level = Math.min(Math.max(severity, 0), 3);
  return (
    <Badge variant="outline" className={cn("font-normal", SEVERITY_STYLES[level])}>
      Mức {level}: {SEVERITY_LABELS[level]}
    </Badge>
  );
}

export function DiagnosisDetails({ diagnosis }: { diagnosis: DiagnosisReport }) {
  const defects = [...diagnosis.defects].sort((a, b) => b.severity - a.severity);
  const regions = Object.values(diagnosis.region_metrics ?? {});
  return (
    <div className="grid gap-4 text-sm">
      <dl className="grid gap-x-4 gap-y-1 sm:grid-cols-[max-content_1fr]">
        <dt className="text-muted-foreground">Loại cảnh</dt>
        <dd>{label(SCENE_LABELS, diagnosis.scene_type)}</dd>
        {diagnosis.lighting ? (
          <>
            <dt className="text-muted-foreground">Ánh sáng</dt>
            <dd>{diagnosis.lighting}</dd>
          </>
        ) : null}
        {diagnosis.summary ? (
          <>
            <dt className="text-muted-foreground">Tóm tắt</dt>
            <dd>{diagnosis.summary}</dd>
          </>
        ) : null}
      </dl>

      <section className="grid gap-2">
        <h3 className="font-medium">Lỗi phát hiện</h3>
        {defects.length === 0 ? (
          <p className="text-muted-foreground">Không phát hiện lỗi cần xử lý.</p>
        ) : (
          <ul className="grid gap-2">
            {defects.map((defect, index) => (
              <li key={`${defect.type}-${defect.region}-${index}`} className="rounded-md border p-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{label(DEFECT_LABELS, defect.type)}</span>
                  <span className="text-muted-foreground">@ {regionLabel(defect.region)}</span>
                  <SeverityBadge severity={defect.severity} />
                  <span className="text-xs text-muted-foreground">
                    Nguồn: {label(ORIGIN_LABELS, defect.origin)}
                  </span>
                </div>
                {defect.evidence ? (
                  <p className="mt-1 text-muted-foreground">{defect.evidence}</p>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="grid gap-2">
        <h3 className="font-medium">Cần giữ (phong cách có chủ ý)</h3>
        {diagnosis.preserve.length === 0 ? (
          <p className="text-muted-foreground">Không có.</p>
        ) : (
          <ul className="grid gap-1">
            {diagnosis.preserve.map((item, index) => (
              <li key={`${item.aspect}-${index}`}>
                <span className="font-medium">{label(PRESERVE_LABELS, item.aspect)}</span>{" "}
                <span className="text-muted-foreground">@ {regionLabel(item.region)}</span>
                {item.reason ? <span className="text-muted-foreground"> — {item.reason}</span> : null}
              </li>
            ))}
          </ul>
        )}
      </section>

      {regions.length > 0 ? (
        <section className="grid gap-2">
          <h3 className="font-medium">Số đo theo vùng</h3>
          <div className="overflow-x-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Vùng</TableHead>
                  <TableHead>Diện tích</TableHead>
                  <TableHead>Độ sáng TB</TableHead>
                  <TableHead>Mức</TableHead>
                  <TableHead>Cháy sáng</TableHead>
                  <TableHead>Bệt đen</TableHead>
                  <TableHead>So với phần còn lại</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {regions.map((item) => (
                  <TableRow key={item.region}>
                    <TableCell>
                      {regionLabel(item.region)}
                      <span className="block text-xs text-muted-foreground">{item.backend}</span>
                    </TableCell>
                    <TableCell>{formatPercent(item.area_ratio)}</TableCell>
                    <TableCell>{formatNumber(item.brightness_mean, 1)}</TableCell>
                    <TableCell>{label(BRIGHTNESS_LEVEL_LABELS, item.brightness_level)}</TableCell>
                    <TableCell>{formatPercent(item.highlight_clip_ratio)}</TableCell>
                    <TableCell>{formatPercent(item.shadow_clip_ratio)}</TableCell>
                    <TableCell>{formatNumber(item.brightness_vs_rest, 1)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </section>
      ) : null}
    </div>
  );
}

export function DiagnosisCard({
  diagnosis,
  original,
}: {
  diagnosis: DiagnosisReport | null;
  original: Artifact | null;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-wrap items-center justify-between gap-2">
        <CardTitle>
          <h2 className="text-base">Chẩn đoán ban đầu</h2>
        </CardTitle>
        {diagnosis ? <SourceBadge prefix="Chẩn đoán" source={diagnosis.source} /> : null}
      </CardHeader>
      <CardContent className="grid gap-4">
        {original ? (
          <img
            src={original.url}
            alt="Ảnh gốc"
            width={original.width}
            height={original.height}
            className="max-h-[420px] w-full rounded-md bg-muted object-contain"
          />
        ) : null}
        {original ? (
          <p className="text-xs text-muted-foreground">
            Ảnh gốc {original.width}×{original.height}
          </p>
        ) : null}
        {diagnosis ? (
          <DiagnosisDetails diagnosis={diagnosis} />
        ) : (
          <p className="text-sm text-muted-foreground">Không có dữ liệu chẩn đoán.</p>
        )}
      </CardContent>
    </Card>
  );
}
