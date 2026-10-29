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
  fixable?: boolean;
  confidence?: number;
  contraindications?: string[];
}

/** Một chỉnh sửa của người dùng lên chẩn đoán (chat bổ sung phân tích). */
export interface DiagnosisEdit {
  op: "add_defect" | "remove_defect" | "keep";
  type: string | null;
  aspect: string | null;
  region: string;
  severity: number;
  source_text: string;
}

export interface RegionInfo {
  name: string;
  source: string;
  confidence: number;
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
  rejected?: Defect[];
  ask_user?: Defect[];
  fusion_notes?: string[];
  measurement?: ({ width: number; height: number; tile_size: number; notes?: string[] } &
    Partial<Record<"exposure" | "noise" | "sharpness" | "color" | "compression" | "haze", Record<string, MeasurementEstimate>>>) | null;
  region_stats?: Record<string, { region: string; area_fraction: number; values: Record<string, MeasurementEstimate>; mask_source: string; mask_confidence: number }>;
  diff?: { defects?: DefectChange[]; side_effects?: ProcessingSideEffect[] } | null;
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
  | "user_prompt"
  | "diagnosis"
  | "questions"
  | "answers"
  | "result"
  | "feedback"
  | "refine_result"
  | "render_result"
  | "analysis_note"
  | "analysis_update"
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
  draft_edits: DiagnosisEdit[];
  draft_diagnosis: DiagnosisReport | null;
  region_overlay_url: string | null;
  regions: RegionInfo[];
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
  artifacts: { final: string; intermediate: string[]; executed_overlay?: string | null };
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

/** Báo cáo thực thi một action (executor ghi): yêu cầu so với thực tế. */
export interface ExecutedAction {
  region_id: string;
  operation: string;
  requested_mode?: string | null;
  actual_mode?: string | null;
  backend?: string;
  fallback_reason?: string | null;
  feather_requested?: number | null;
  feather_actual?: number | null;
  strength_scale?: number;
  mask_area?: number | null;
  skipped?: boolean;
  skip_reason?: string | null;
}

export interface HistoryItem {
  executed?: ExecutedAction[];
  defect_status?: DefectChange[];
  side_effects?: ProcessingSideEffect[];
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

export interface MeasurementEstimate { value: number; confidence: number; n_samples?: number }

export interface DefectChange {
  type: string;
  region: string;
  before: number;
  after: number;
  status: "resolved" | "improved" | "unchanged" | "worse";
  confidence: number;
}

export interface ProcessingSideEffect {
  kind: string;
  region: string;
  severity: number;
  value: number;
  before?: number | null;
  after?: number | null;
  evidence: string;
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
