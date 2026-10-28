import { SlidersHorizontal } from "lucide-react";
import type { SessionDetail } from "@/api/types";
import { CompareSlider } from "@/components/result/CompareSlider";
import { IterationTimeline } from "@/components/result/IterationTimeline";
import { SourceBadge, DecisionBadge } from "@/components/session/StatusBadge";
import { describeAction } from "@/lib/actions";
import { roundImages } from "./assessment";
import { RoundAssessment } from "./RoundAssessment";

export function ProcessingPanel({ detail, roundIndex }: { detail: SessionDetail; roundIndex: number }) {
  const process = detail.process;
  const round = process?.history[roundIndex];
  if (!process || !round) return null;
  const artifacts = new Map(detail.artifacts.map((a) => [a.id, a]));
  const { before, after } = roundImages(detail, roundIndex);
  const afterLabel = round.rolled_back ? `Ảnh thử vòng ${round.iteration} · đã loại bỏ` : `Sau vòng ${round.iteration}`;
  return (
    <section className="studio-processing" aria-label="Phác đồ và thực thi">
      <header className="studio-panel-title"><SlidersHorizontal aria-hidden /><h2>Phác đồ & thực thi</h2><span>Vòng {round.iteration}</span></header>
      <div className="studio-panel-body">
        {round.rolled_back ? <p className="studio-rollback" role="status">Ảnh thử của vòng {round.iteration} đã bị loại bỏ. Pipeline giữ lại ảnh trước vòng này; ảnh thử dưới đây chỉ để đối chiếu.</p> : null}
        {before && after ? <CompareSlider before={before} after={after} beforeLabel={`Đầu vào vòng ${round.iteration}`} afterLabel={afterLabel} />
          : after ? <img src={after.url} alt={afterLabel} className="w-full rounded-md" />
            : <p className="text-sm text-muted-foreground">Vòng này không có ảnh trung gian được lưu.</p>}
        <div className="flex flex-wrap gap-2">
          {round.diagnosis ? <SourceBadge prefix="Chẩn đoán" source={round.diagnosis.source} /> : null}
          <SourceBadge prefix="Phác đồ" source={round.plan.source} />
        </div>
        <div><p className="studio-eyebrow">CÁC THAO TÁC TRONG PHÁC ĐỒ</p>
          {round.plan.actions.length ? <ol className="studio-actions">{round.plan.actions.map((a, i) => (
            <li key={i}><span>{i + 1}</span><p>{describeAction(a)}</p></li>
          ))}</ol> : <p className="mt-2 text-sm text-muted-foreground">Phác đồ không có thao tác.</p>}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs"><span>Quyết định đã ghi:</span><DecisionBadge decision={round.decision} /></div>
        <RoundAssessment round={round} />
        <details className="studio-round-details"><summary>Chi tiết các vòng & chỉ số</summary>
          <IterationTimeline history={process.history} intermediates={process.artifacts.intermediate.map((id) => artifacts.get(id))} />
        </details>
      </div>
    </section>
  );
}
