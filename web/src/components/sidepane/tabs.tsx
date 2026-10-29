import type { DiagnosisReport, SessionDetail } from "@/api/types";
import { assessmentFields, measurementFields } from "@/components/studio/assessment";
import { LOW_REGION_CONFIDENCE } from "@/components/session/DiagnosisCard";
import { Badge } from "@/components/ui/badge";
import {
  BRIGHTNESS_LEVEL_LABELS,
  PRESERVE_LABELS,
  formatNumber,
  formatPercent,
  label,
  regionLabel,
} from "@/lib/labels";

/** Cặp nhãn – giá trị dạng danh sách mô tả. */
function Fields({ fields }: { fields: { label: string; value: string }[] }) {
  return (
    <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1">
      {fields.map((field, index) => (
        <div key={`${field.label}-${index}`} className="contents">
          <dt className="text-muted-foreground">{field.label}</dt>
          <dd className="text-right">{field.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-muted-foreground">{children}</p>;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-2">
      <h3 className="font-medium">{title}</h3>
      {children}
    </section>
  );
}

/** Tab Mặt nạ: overlay vùng, backend/độ tin cậy từng vùng và các vùng cần bảo vệ. */
export function MasksTab({
  detail,
  diagnosis,
}: {
  detail: SessionDetail;
  diagnosis: DiagnosisReport | null;
}) {
  const overlay = detail.region_overlay_url
    ? `${detail.region_overlay_url}${detail.region_overlay_url.includes("?") ? "&" : "?"}revision=${encodeURIComponent(detail.updated_at)}`
    : null;
  const executedOverlayId = detail.process?.artifacts.executed_overlay;
  const executedOverlay = detail.artifacts.find((a) => a.id === executedOverlayId) ?? null;
  const preserve = diagnosis?.preserve ?? [];
  return (
    <div className="grid gap-4">
      {overlay ? (
        <img
          src={overlay}
          alt="Ảnh gốc với vùng được tô màu"
          className="max-h-72 w-full rounded-md bg-muted object-contain"
        />
      ) : (
        <Empty>Chưa có ảnh phủ vùng cho phiên này.</Empty>
      )}
      <Section title="Vùng đã tìm thấy">
        {detail.regions.length === 0 ? (
          <Empty>Không có vùng riêng, xử lý trên toàn ảnh.</Empty>
        ) : (
          <ul className="grid gap-1">
            {detail.regions.map((region) => (
              <li key={region.name} className="flex flex-wrap items-center gap-2 rounded-md border px-2 py-1">
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
        )}
      </Section>
      <Section title="Cần giữ nguyên">
        {preserve.length === 0 ? (
          <Empty>Không có.</Empty>
        ) : (
          <ul className="grid gap-1">
            {preserve.map((item, index) => (
              <li key={`${item.aspect}-${index}`}>
                <span className="font-medium">{label(PRESERVE_LABELS, item.aspect)}</span>{" "}
                <span className="text-muted-foreground">
                  @ {regionLabel(item.region)}
                  {item.reason ? ` — ${item.reason}` : ""}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Section>
      {executedOverlay ? (
        <Section title="Vùng đã xử lý thực tế">
          <img
            src={executedOverlay.url}
            alt="Vùng đã xử lý thực tế"
            className="max-h-72 w-full rounded-md bg-muted object-contain"
          />
        </Section>
      ) : null}
    </div>
  );
}

/** Tab Số đo: chỉ hiển thị số liệu có trong payload (measurement, region_stats, region_metrics). */
export function MetricsTab({ diagnosis }: { diagnosis: DiagnosisReport | null }) {
  const regions = Object.values(diagnosis?.region_metrics ?? {});
  const fields = measurementFields(diagnosis);
  if (regions.length === 0 && fields.length === 0) {
    return <Empty>Chưa có số đo cho chẩn đoán này.</Empty>;
  }
  return (
    <div className="grid gap-4">
      {regions.length > 0 ? (
        <Section title="Độ sáng theo vùng">
          <ul className="grid gap-2">
            {regions.map((item) => (
              <li key={item.region} className="grid gap-0.5 rounded-md border p-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{regionLabel(item.region)}</span>
                  <Badge variant="outline" className="font-normal">
                    {label(BRIGHTNESS_LEVEL_LABELS, item.brightness_level)}
                  </Badge>
                  <span className="text-xs text-muted-foreground">{item.backend}</span>
                </div>
                <p className="text-muted-foreground">
                  Độ sáng TB {formatNumber(item.brightness_mean, 1)} · diện tích{" "}
                  {formatPercent(item.area_ratio)}
                  {item.brightness_vs_rest != null
                    ? ` · chênh so với phần còn lại ${formatNumber(item.brightness_vs_rest, 1)}`
                    : ""}
                </p>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
      {fields.length > 0 ? (
        <Section title="Phép đo đã lưu">
          <Fields fields={fields} />
        </Section>
      ) : null}
    </div>
  );
}

/** Tab Lý do: căn cứ của phác đồ, ghi chú dung hợp và ghi chú của phiên bản. */
export function ReasoningTab({
  detail,
  diagnosis,
}: {
  detail: SessionDetail;
  diagnosis: DiagnosisReport | null;
}) {
  const rounds = detail.process?.history ?? [];
  const notes = diagnosis?.fusion_notes ?? [];
  const treatmentNotes = detail.treatments.flatMap((t) =>
    t.notes.map((note) => ({ version: t.version, note })),
  );
  if (rounds.length === 0 && notes.length === 0 && treatmentNotes.length === 0) {
    return <Empty>Chưa có căn cứ suy luận. Chúng xuất hiện sau khi xử lý.</Empty>;
  }
  return (
    <div className="grid gap-4">
      {notes.length > 0 ? (
        <Section title="Ghi chú dung hợp">
          <ul className="list-inside list-disc">
            {notes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </Section>
      ) : null}
      {rounds.map((round) => (
        <Section key={round.iteration} title={`Vòng ${round.iteration}`}>
          <p className="whitespace-pre-wrap">{round.plan.reasoning || "Không có lý do được ghi."}</p>
          {round.plan.knowledge.length > 0 ? (
            <p className="text-xs text-muted-foreground">
              Thẻ tri thức: {round.plan.knowledge.join(", ")}
            </p>
          ) : null}
        </Section>
      ))}
      {treatmentNotes.length > 0 ? (
        <Section title="Ghi chú phiên bản">
          <ul className="list-inside list-disc">
            {treatmentNotes.map((item, index) => (
              <li key={index}>
                v{item.version}: {item.note}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

/** Tab Thực thi: thao tác thực tế, mức cải thiện từng lỗi và tác dụng phụ theo vòng. */
export function ExecutionTab({ detail }: { detail: SessionDetail }) {
  const rounds = detail.process?.history ?? [];
  if (rounds.length === 0) return <Empty>Chưa có thao tác nào được thực thi.</Empty>;
  return (
    <div className="grid gap-4">
      {rounds.map((round) => {
        const executed = round.executed ?? [];
        const assessment = assessmentFields(round);
        return (
          <Section key={round.iteration} title={`Vòng ${round.iteration}`}>
            {round.rolled_back ? (
              <p className="text-xs text-warning">Vòng này đã bị rollback, ảnh thử không được giữ.</p>
            ) : null}
            {executed.length === 0 ? (
              <Empty>Không có thao tác nào được ghi.</Empty>
            ) : (
              <ul className="grid gap-2">
                {executed.map((action, index) => (
                  <li key={index} className="grid gap-0.5 rounded-md border p-2">
                    <span className="font-medium">
                      {action.operation} · {regionLabel(action.region_id)}
                    </span>
                    <span className="text-muted-foreground">
                      Chế độ: {action.requested_mode ?? "—"} → {action.actual_mode ?? "—"}
                      {action.backend ? ` · ${action.backend}` : ""}
                    </span>
                    <span className="text-muted-foreground">
                      Feather {action.feather_actual ?? "—"} · cường độ ×
                      {formatNumber(action.strength_scale ?? 1, 2)}
                    </span>
                    {action.fallback_reason ? (
                      <span className="text-xs text-warning">Dự phòng: {action.fallback_reason}</span>
                    ) : null}
                    {action.skipped ? (
                      <span className="text-xs text-warning">
                        Bỏ qua{action.skip_reason ? `: ${action.skip_reason}` : ""}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
            {assessment.length > 0 ? <Fields fields={assessment} /> : null}
          </Section>
        );
      })}
    </div>
  );
}
