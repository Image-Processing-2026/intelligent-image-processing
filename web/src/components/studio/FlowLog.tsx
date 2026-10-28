import { Brain, ChevronRight, Info, LogIn, LogOut, Workflow } from "lucide-react";
import { useState } from "react";
import type { SessionDetail } from "@/api/types";
import { buildFlowSteps, type FlowField } from "./flow-data";
import type { StudioPhase } from "./PhaseNav";

function Fields({ values }: { values: FlowField[] }) {
  return values.length ? <dl className="studio-log-fields">{values.map((field, i) => (
    <div key={i}><dt>{field.label}</dt><dd>{field.value}</dd></div>
  ))}</dl> : <p className="studio-missing">Chưa có dữ liệu riêng được ghi nhận.</p>;
}

export function FlowLog({ detail, phase, roundIndex, onRoundChange, busy }: {
  detail: SessionDetail; phase: StudioPhase; roundIndex: number;
  onRoundChange: (index: number) => void; busy: boolean;
}) {
  const steps = buildFlowSteps(detail, phase, roundIndex);
  const [selected, setSelected] = useState(phase === "analysis" ? "diagnosis" : phase === "processing" ? "plan" : "summary");
  const step = steps.find((s) => s.id === selected) ?? steps[0];
  return (
    <aside className="studio-log" data-phase={phase} aria-label="Flow log">
      <header className="studio-log-heading"><Workflow aria-hidden /><h2>Flow log</h2><span>Bản ghi thực tế</span></header>
      <p className="studio-log-subtitle">Input → Căn cứ & quyết định → Output</p>
      {phase === "processing" && detail.process?.history.length ? (
        <div className="studio-rounds" aria-label="Chọn vòng xử lý">
          {detail.process.history.map((r, i) => <button key={r.iteration} type="button" aria-pressed={roundIndex === i}
            onClick={() => onRoundChange(i)}>Vòng {r.iteration}</button>)}
        </div>
      ) : null}
      {busy ? <p className="studio-log-pending" role="status">Đang chờ kết quả từ máy chủ. Log được cập nhật khi yêu cầu hoàn tất.</p> : null}
      <div className="studio-log-steps" aria-label="Các bước đã ghi nhận">
        {steps.map((s, i) => <button type="button" key={s.id} aria-label={`${String(i + 1).padStart(2, "0")} ${s.title} · ${s.actor}`} aria-pressed={s.id === step?.id} onClick={() => setSelected(s.id)}>
          <span className="studio-step-number">{String(i + 1).padStart(2, "0")}</span>
          <span><strong>{s.title}</strong><small>{s.actor}</small></span><ChevronRight aria-hidden />
        </button>)}
      </div>
      {step ? <section className="studio-inspector" aria-label={`Chi tiết bước: ${step.title}`}>
        <h3>{step.title}</h3>
        <div><h4><LogIn aria-hidden /> Đầu vào</h4><Fields values={step.input} /></div>
        <div><h4><Brain aria-hidden /> Căn cứ & quyết định</h4>
          {step.rationale ? <p className="studio-rationale">{step.rationale}</p>
            : <p className="studio-missing">Backend không lưu phần giải thích riêng cho bước này.</p>}
        </div>
        <div><h4><LogOut aria-hidden /> Đầu ra</h4><Fields values={step.output} />
          {step.image ? <img src={step.image.url} alt={`Đầu ra: ${step.title}`} className="studio-log-image" loading="lazy" /> : null}
        </div>
        {step.id === "execute" ? <p className="studio-missing">Phác đồ không xác nhận thành công của từng công cụ; hiện chưa có log thực thi riêng từng thao tác.</p> : null}
      </section> : !busy ? <p className="studio-log-subtitle">Chưa có bản ghi cho phase này.</p> : null}
      <footer><Info aria-hidden /><span>Hiển thị dữ liệu đã lưu, không mô phỏng tiến độ trực tiếp.</span></footer>
    </aside>
  );
}
