import type { HistoryItem } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { DEFECT_LABELS, label, regionLabel } from "@/lib/labels";
import { DEFECT_STATUS_LABELS, SIDE_EFFECT_LABELS } from "./assessment";

export function RoundAssessment({ round }: { round: HistoryItem }) {
  const changes = round.defect_status ?? [];
  const effects = round.side_effects ?? [];
  if (!changes.length && !effects.length) return null;
  return (
    <section className="studio-outcomes" aria-label={`Đánh giá theo vùng · Vòng ${round.iteration}`}>
      <h3 className="font-medium text-sm">Đánh giá theo vùng · Vòng {round.iteration}</h3>
      {round.rolled_back ? <p className="studio-rollback">Đánh giá ảnh thử đã bị loại bỏ. Các lỗi và tác dụng phụ dưới đây không phải đánh giá của ảnh kết quả đang giữ.</p> : null}
      {changes.length ? <>
        <p className="text-xs text-muted-foreground">Mức lỗi so với ảnh gốc trên đúng vùng được theo dõi. Điểm toàn ảnh chỉ là chỉ số bổ sung.</p>
        <ul>{changes.map((d, i) => <li key={i}>
          <div><strong>{label(DEFECT_LABELS, d.type)}</strong><span>{regionLabel(d.region)} · Mức {d.before} → {d.after}</span></div>
          <Badge variant="outline" data-status={d.status}>{label(DEFECT_STATUS_LABELS, d.status)}</Badge>
        </li>)}</ul>
      </> : null}
      {effects.length ? <div className="grid gap-2 text-sm"><h4>Tác dụng phụ đã ghi nhận</h4>
        <ul>{effects.map((e, i) => <li key={i}>
          <div><strong>{label(SIDE_EFFECT_LABELS, e.kind)}</strong><span>{regionLabel(e.region)} · Mức {e.severity}</span>
            {e.evidence ? <p>{e.evidence}</p> : null}</div>
        </li>)}</ul>
      </div> : null}
    </section>
  );
}
