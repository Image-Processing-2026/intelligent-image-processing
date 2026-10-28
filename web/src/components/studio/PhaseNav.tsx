import { Images, ScanSearch, SlidersHorizontal } from "lucide-react";

export type StudioPhase = "analysis" | "processing" | "result";

const PHASES = [
  { id: "analysis", title: "Phân tích", hint: "Hiểu ảnh & ý định", icon: ScanSearch },
  { id: "processing", title: "Xử lý", hint: "Phác đồ & thực thi", icon: SlidersHorizontal },
  { id: "result", title: "Kết quả", hint: "So sánh & tinh chỉnh", icon: Images },
] as const;

export function PhaseNav({ value, onChange, canProcess, hasResult }: {
  value: StudioPhase;
  onChange: (phase: StudioPhase) => void;
  canProcess: boolean;
  hasResult: boolean;
}) {
  return (
    <nav className="studio-phase-nav" aria-label="Các phase của phiên">
      {PHASES.map(({ id, title, hint, icon: Icon }, index) => (
        <button key={id} type="button" className="studio-phase" data-phase={id}
          aria-label={`0${index + 1} ${title}`}
          aria-pressed={value === id} disabled={id === "processing" ? !canProcess : id === "result" && !hasResult}
          onClick={() => onChange(id)}>
          <span className="studio-phase-number">0{index + 1}</span>
          <span className="min-w-0"><strong>{title}</strong><small>{hint}</small></span>
          <Icon aria-hidden />
        </button>
      ))}
    </nav>
  );
}
