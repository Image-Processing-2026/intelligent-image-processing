// Kiểu dữ liệu của API mới (docs/frontend/spec.md §3).

export interface UserSettings {
  max_iterations: number;
  num_variants: number;
}

export interface User {
  id: string;
  username: string;
  settings: UserSettings;
  created_at: string;
}

export type SessionStatus = "uploaded" | "awaiting_answers" | "running" | "done" | "failed";
export type PlanSource = "vlm" | "rule_based" | "vlm_fallback";

export interface Artifact {
  id: string;
  kind: string;
  url: string;
  download_url: string;
  width: number;
  height: number;
}

export interface Defect {
  type: string;
  region: string;
  severity: number;
  evidence: string;
  origin: "vlm" | "rule" | "measured" | "user";
}

export interface PreserveItem {
  aspect: string;
  region: string;
  reason: string;
}

export interface RegionMetrics {
  region: string;
  backend: string;
  area_ratio: number;
  brightness_mean: number;
  brightness_std: number;
  brightness_level: "underexposed" | "normal" | "overexposed";
  highlight_clip_ratio: number;
  shadow_clip_ratio: number;
  brightness_vs_rest: number | null;
}

export interface DiagnosisReport {
  iteration: number;
  scene_type: string;
  lighting: string;
  subjects: string[];
  defects: Defect[];
  preserve: PreserveItem[];
  summary: string;
  region_metrics: Record<string, RegionMetrics>;
  source: PlanSource;
}

export interface Choice {
  value: string;
  label: string;
  keep?: string | null;
  fix?: string | null;
  style?: string | null;
}

export interface Question {
  id: string;
  text: string;
  choices: Choice[];
  default: string;
}

export type MessageKind =
  | "diagnosis"
  | "questions"
  | "answers"
  | "result"
  | "feedback"
  | "refine_result"
  | "render_result"
  | "notice";

export interface Message {
  id: number;
  role: "system" | "user" | "assistant";
  kind: MessageKind;
  content: Record<string, unknown>;
  created_at: string;
}

export interface SessionSummary {
  id: string;
  title: string;
  status: SessionStatus;
  created_at: string;
  updated_at: string;
  thumbnail_url: string | null;
  final_decision: string | null;
  /** Phiên cũ (trước R-01) không có kết cục. */
  outcome_status?: OutcomeStatus | null;
}

export type OutcomeStatus = "ACCEPTED" | "UNVERIFIED" | "NOT_MET" | "KEPT_ORIGINAL" | "NO_CHANGE";

/** Kết cục trung thực của ảnh được trả; chỉ ACCEPTED ứng với final_decision SHIP. */
export interface Outcome {
  status: OutcomeStatus;
  accepted_by: "review" | "reference" | null;
  candidate_sha256: string | null;
  reason: string;
}

export interface ReviewFinding {
  crop_id: string;
  issue: string;
  severity: number;
  text: string;
}

export interface ReviewResult {
  status: "completed" | "unavailable" | "not_run";
  verdict: "ACCEPT" | "REVISE" | "REJECT" | "UNCERTAIN" | null;
  goal: string;
  preserve: string[];
  findings: ReviewFinding[];
  confidence: number;
  reason: string;
  candidate_sha256: string | null;
  model: string;
  prompt_version: string;
  latency_ms: number | null;
  error: string | null;
}

export interface SessionOptions {
  max_iterations: number;
  num_variants: number;
  has_ground_truth: boolean;
  sample?: string | null;
}

export interface SessionDetail extends SessionSummary {
  options: SessionOptions;
  original: Artifact | null;
  ground_truth: Artifact | null;
  messages: Message[];
  diagnosis: DiagnosisReport | null;
  questions: Question[] | null;
  process: ProcessResult | null;
  variants: Variant[];
  treatments: Treatment[];
  active_treatment_id: string | null;
  ratings: Rating[];
  error: { code: string; message: string } | null;
  artifacts: Artifact[];
}

export interface Sample {
  id: string;
  label: string;
  thumbnail_url: string;
}

export interface ProcessResult {
  run_id: string;
  total_iterations: number;
  final_decision: string;
  final_evaluation: Record<string, unknown>;
  /** Phiên cũ không có outcome/review: UI quay về hiển thị final_decision. */
  outcome?: Outcome | null;
  review?: ReviewResult | null;
  history: HistoryItem[];
  treatment: RegionAction[];
  intent: Record<string, unknown> | null;
  recommended_variant: string | null;
  recommended_reason: string | null;
  variant_ranking_source: string | null;
  rolled_back: boolean;
  is_synthetic: boolean;
  case_id: string | null;
  artifacts: { final: string; intermediate: string[] };
  sources: { diagnosis: string[]; plan: string[] };
}

export interface RegionAction {
  region_id: string;
  target_prompt: string;
  operation: string;
  detected_issue: string;
  parameters: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Variant {
  id: string;
  run_id: string;
  style_id: string;
  label: string;
  description: string;
  rank: number | null;
  quality_score: number | null;
  quality_improved: boolean | null;
  critic_note: string;
  actions: RegionAction[];
  preview_artifact_id: string | null;
  recommended: boolean;
  chosen: boolean;
}

export interface Adjustment {
  kind: string;
  region: string;
  strength: number;
}

export interface Treatment {
  id: string;
  parent_id: string | null;
  version: number;
  source: "pipeline" | "variant" | "refine";
  variant_id: string | null;
  actions: RegionAction[];
  feedback_text: string | null;
  adjustments: Adjustment[];
  notes: string[];
  refine_source: string | null;
  quality_score: number | null;
  artifact_id: string | null;
  run_id: string;
  created_at: string;
}

export interface HistoryItem {
  iteration: number;
  plan: {
    iteration: number;
    reasoning: string;
    actions: RegionAction[];
    knowledge: string[];
    source: string;
  };
  metrics_before: Record<string, unknown>;
  metrics_after: Record<string, unknown>;
  eval_score: Record<string, unknown>;
  decision: string;
  diagnosis: DiagnosisReport | null;
  rolled_back: boolean;
}

export interface Rating {
  id: string;
  run_id: string;
  treatment_id: string | null;
  rating: "good" | "bad";
  note: string;
  created_at: string;
  updated_at: string;
}

export interface SystemInfo {
  backend: string;
  gemini_key_configured: boolean;
  gemini_model: string;
  memory_enabled: boolean;
  pipeline_concurrency: number;
  registration_open: boolean;
  max_upload_mb: number;
  region_runtime?: Record<string, unknown>;
}

export interface MemoryStats {
  enabled: boolean;
  cases: number;
  choices: Record<string, number>;
  feedback: Record<string, number>;
  preferred_style: string | null;
}
