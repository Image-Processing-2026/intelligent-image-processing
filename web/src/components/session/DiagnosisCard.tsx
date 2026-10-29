import { AlertTriangle } from "lucide-react";
import { useState } from "react";

import type { Artifact, Defect, DiagnosisReport, RegionInfo } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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

/** Dưới ngưỡng này vùng có thể chưa chuẩn (mask lấy từ ước lượng yếu). */
export const LOW_REGION_CONFIDENCE = 0.5;

export function SeverityBadge({ severity }: { severity: number }) {
  const level = Math.min(Math.max(severity, 0), 3);
  return (
    <Badge variant="outline" className={cn("font-normal", SEVERITY_STYLES[level])}>
      Mức {level}: {SEVERITY_LABELS[level]}
    </Badge>
  );
}

interface DiagnosisDetailsProps {
  diagnosis: DiagnosisReport;
  /** Lỗi gốc mà người dùng đã bỏ (hiển thị gạch ngang). */
  removed?: Defect[];
  selectedRegion?: string | null;
  /** Có thì mỗi lỗi ở vùng cụ thể có nút "Xem vùng" làm nổi vùng trong danh sách vùng. */
  onSelectRegion?: (region: string | null) => void;
}

export function DiagnosisDetails({
  diagnosis,
  removed = [],
  selectedRegion = null,
  onSelectRegion,
}: DiagnosisDetailsProps) {
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
        {defects.length === 0 && removed.length === 0 ? (
          <p className="text-muted-foreground">Không phát hiện lỗi cần xử lý.</p>
        ) : (
          <ul className="grid gap-2">
            {defects.map((defect, index) => (
              <li key={`${defect.type}-${defect.region}-${index}`} className="rounded-md border p-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{label(DEFECT_LABELS, defect.type)}</span>
                  <span className="text-muted-foreground">@ {regionLabel(defect.region)}</span>
                  <SeverityBadge severity={defect.severity} />
                  {defect.fixable === false ? <Badge variant="outline" className="text-warning">Ngoài khả năng toolbox hiện tại</Badge> : null}
                  <span className="text-xs text-muted-foreground">
                    Nguồn: {label(ORIGIN_LABELS, defect.origin)}
                  </span>
                  {defect.origin === "user" ? (
                    <Badge className="font-normal">Bạn bổ sung</Badge>
                  ) : null}
                  {onSelectRegion && defect.region !== "full" ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      aria-pressed={selectedRegion === defect.region}
                      onClick={() =>
                        onSelectRegion(selectedRegion === defect.region ? null : defect.region)
                      }
                    >
                      Xem vùng
                    </Button>
                  ) : null}
                </div>
                {defect.evidence ? (
                  <p className="mt-1 text-muted-foreground">{defect.evidence}</p>
                ) : null}
              </li>
            ))}
            {removed.map((defect, index) => (
              <li
                key={`removed-${defect.type}-${defect.region}-${index}`}
                className="rounded-md border border-dashed p-2 text-muted-foreground"
              >
                <span className="line-through">
                  {label(DEFECT_LABELS, defect.type)} @ {regionLabel(defect.region)}
                </span>{" "}
                <span className="text-xs">Bạn đã bỏ lỗi này</span>
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

interface DiagnosisCardProps {
  diagnosis: DiagnosisReport | null;
  original: Artifact | null;
  /** Ảnh gốc tô màu các vùng (chỉ có khi phiên đang chờ trả lời). */
  overlayUrl?: string | null;
  overlayRevision?: string;
  regions?: RegionInfo[];
  removed?: Defect[];
  /** Còn chỉnh sửa được (chờ trả lời): hiện danh sách vùng và nút "Xem vùng" theo lỗi. */
  inspectable?: boolean;
}

export function DiagnosisCard({
  diagnosis,
  original,
  overlayUrl = null,
  overlayRevision,
  regions = [],
  removed = [],
  inspectable = false,
}: DiagnosisCardProps) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const selected = regions.find((region) => region.name === selectedRegion) ?? null;
  const overlayVisible = Boolean(overlayUrl) && showOverlay;
  // Backend ghi đè cùng artifact: URL phải đổi theo bản cập nhật để ảnh được tải lại.
  const overlaySrc = overlayUrl && overlayRevision
    ? `${overlayUrl}${overlayUrl.includes("?") ? "&" : "?"}revision=${encodeURIComponent(overlayRevision)}`
    : overlayUrl;
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
            src={overlayVisible && overlaySrc ? overlaySrc : original.url}
            alt={overlayVisible ? "Ảnh gốc với vùng được tô màu" : "Ảnh gốc"}
            width={original.width}
            height={original.height}
            className="max-h-[420px] w-full rounded-md bg-muted object-contain"
          />
        ) : null}
        {original ? (
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-xs text-muted-foreground">
              Ảnh gốc {original.width}×{original.height}
            </p>
            {overlayUrl ? (
              <Button
                type="button"
                size="sm"
                variant="outline"
                aria-pressed={overlayVisible}
                onClick={() => setShowOverlay(!overlayVisible)}
              >
                {overlayVisible ? "Hiện ảnh gốc" : "Hiện vùng đã tìm thấy"}
              </Button>
            ) : null}
          </div>
        ) : null}

        {inspectable && regions.length > 0 ? (
          <section className="grid gap-2 text-sm" aria-label="Vùng đã tìm thấy">
            <h3 className="font-medium">Vùng đã tìm thấy</h3>
            <ul className="grid gap-1">
              {regions.map((region) => (
                <li
                  key={region.name}
                  className={cn(
                    "flex flex-wrap items-center gap-2 rounded-md border px-2 py-1",
                    region.name === selectedRegion && "border-primary bg-accent",
                  )}
                  aria-current={region.name === selectedRegion ? "true" : undefined}
                >
                  <span className="font-medium">{regionLabel(region.name)}</span>
                  <span className="text-xs text-muted-foreground">
                    {region.source} · tin cậy {formatPercent(region.confidence, 0)}
                  </span>
                  {region.confidence < LOW_REGION_CONFIDENCE ? (
                    <Badge variant="outline" className="border-warning/50 font-normal text-warning">
                      Tin cậy thấp
                    </Badge>
                  ) : null}
                </li>
              ))}
            </ul>
            {selected && selected.confidence < LOW_REGION_CONFIDENCE ? (
              <p className="flex items-start gap-1.5 text-xs text-warning" role="status">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                Vùng có thể chưa chuẩn, hãy nói rõ trong khung chat.
              </p>
            ) : null}
          </section>
        ) : null}

        {diagnosis ? (
          <DiagnosisDetails
            diagnosis={diagnosis}
            removed={removed}
            selectedRegion={selectedRegion}
            onSelectRegion={inspectable ? (region) => { setSelectedRegion(region); if (region) setShowOverlay(true); } : undefined}
          />
        ) : (
          <p className="text-sm text-muted-foreground">Không có dữ liệu chẩn đoán.</p>
        )}
      </CardContent>
    </Card>
  );
}
