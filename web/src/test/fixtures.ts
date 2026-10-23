// Dữ liệu mẫu CHỈ dùng trong test (không có trong bản chạy thật).
import type {
  ProcessResult,
  SessionDetail,
  SessionSummary,
  Treatment,
  Variant,
} from "@/api/types";

export const SESSION_ID = "b".repeat(32);

export const SUMMARY: SessionSummary = {
  id: SESSION_ID,
  title: "Ảnh mẫu: Thiếu sáng",
  status: "awaiting_answers",
  created_at: "2026-10-22T10:00:00+00:00",
  updated_at: "2026-10-22T10:00:00+00:00",
  thumbnail_url: `/api/v1/workspace/sessions/${SESSION_ID}/artifacts/${"c".repeat(32)}`,
  final_decision: null,
};

export const DETAIL: SessionDetail = {
  ...SUMMARY,
  options: { max_iterations: 3, num_variants: 3, has_ground_truth: false, sample: "underexposed" },
  original: {
    id: "d".repeat(32),
    kind: "original",
    url: `/api/v1/workspace/sessions/${SESSION_ID}/artifacts/${"d".repeat(32)}`,
    download_url: `/api/v1/workspace/sessions/${SESSION_ID}/artifacts/${"d".repeat(32)}?download=1`,
    width: 640,
    height: 480,
  },
  ground_truth: null,
  messages: [
    {
      id: 1,
      role: "assistant",
      kind: "diagnosis",
      created_at: "2026-10-22T10:00:01+00:00",
      content: {
        diagnosis: {
          iteration: 1,
          scene_type: "indoor",
          lighting: "Ánh sáng yếu",
          subjects: [],
          defects: [
            { type: "underexposed", region: "full", severity: 2, evidence: "mean 40", origin: "rule" },
          ],
          preserve: [{ aspect: "warm_tone", region: "full", reason: "Đèn vàng" }],
          summary: "Ảnh trong nhà thiếu sáng",
          region_metrics: {},
          source: "rule_based",
        },
      },
    },
    {
      id: 2,
      role: "assistant",
      kind: "questions",
      created_at: "2026-10-22T10:00:01+00:00",
      content: {
        questions: [
          {
            id: "style",
            text: "Bạn thích kết quả theo phong cách nào?",
            default: "balanced",
            choices: [
              { value: "natural", label: "Tự nhiên (chỉnh nhẹ tay)" },
              { value: "balanced", label: "Cân bằng" },
            ],
          },
        ],
      },
    },
  ],
  diagnosis: null,
  questions: null,
  process: null,
  variants: [],
  treatments: [],
  active_treatment_id: null,
  ratings: [],
  error: null,
  artifacts: [],
};

const FINAL_ID = "e".repeat(32);
export const RUN_ID = "f".repeat(32);

export const FINAL_ARTIFACT = {
  id: FINAL_ID,
  kind: "final",
  url: `/api/v1/workspace/sessions/${SESSION_ID}/artifacts/${FINAL_ID}`,
  download_url: `/api/v1/workspace/sessions/${SESSION_ID}/artifacts/${FINAL_ID}?download=1`,
  width: 640,
  height: 480,
};

const art = (id: string, kind: string, width = 640, height = 480) => ({
  id,
  kind,
  url: `/api/v1/workspace/sessions/${SESSION_ID}/artifacts/${id}`,
  download_url: `/api/v1/workspace/sessions/${SESSION_ID}/artifacts/${id}?download=1`,
  width,
  height,
});

export const ITER_ARTIFACT = art("1".repeat(32), "intermediate", 512, 384);
export const PREVIEW_NATURAL = art("2".repeat(32), "variant_preview", 600, 450);
export const PREVIEW_VIVID = art("3".repeat(32), "variant_preview", 600, 450);
export const RENDER_ARTIFACT = art("4".repeat(32), "render");

const GAMMA = {
  region_id: "full",
  target_prompt: "full",
  operation: "gamma_correct",
  detected_issue: "underexposed",
  parameters: { gamma: 1.4 },
};

export const VARIANTS: Variant[] = [
  {
    id: "5".repeat(32),
    run_id: RUN_ID,
    style_id: "natural",
    label: "Tự nhiên",
    description: "Chỉnh nhẹ tay",
    rank: 1,
    quality_score: 80.5,
    quality_improved: true,
    critic_note: "",
    actions: [GAMMA],
    preview_artifact_id: PREVIEW_NATURAL.id,
    recommended: true,
    chosen: false,
  },
  {
    id: "6".repeat(32),
    run_id: RUN_ID,
    style_id: "vivid",
    label: "Đậm nét",
    description: "Rực, tương phản cao",
    rank: 2,
    quality_score: 78.1,
    quality_improved: true,
    critic_note: "",
    actions: [GAMMA, { ...GAMMA, operation: "clahe", parameters: { clip_limit: 2 } }],
    preview_artifact_id: PREVIEW_VIVID.id,
    recommended: false,
    chosen: false,
  },
];

export const PIPELINE_TREATMENT: Treatment = {
  id: "7".repeat(32),
  parent_id: null,
  version: 1,
  source: "pipeline",
  variant_id: null,
  actions: [GAMMA],
  feedback_text: null,
  adjustments: [],
  notes: [],
  refine_source: null,
  quality_score: null,
  artifact_id: FINAL_ID,
  run_id: RUN_ID,
  created_at: "2026-10-22T10:02:00+00:00",
};

export const PROCESS: ProcessResult = {
  run_id: RUN_ID,
  total_iterations: 2,
  final_decision: "SHIP",
  final_evaluation: { estimated_quality_score: 71.25, brisque_score: null, niqe_score: null },
  history: [
    {
      iteration: 1,
      plan: {
        iteration: 1,
        reasoning: "Ảnh tối, tăng gamma.",
        actions: [GAMMA],
        knowledge: ["underexposed_global"],
        source: "rule_based",
      },
      metrics_before: { brightness_mean: 35.8, contrast_std: 20.4 },
      metrics_after: { brightness_mean: 92.1, contrast_std: 30.2 },
      eval_score: { estimated_quality_score: 71.25 },
      decision: "SHIP",
      diagnosis: null,
      rolled_back: false,
    },
  ],
  treatment: [GAMMA],
  intent: { style: "natural" },
  recommended_variant: "natural",
  recommended_reason: "intent",
  variant_ranking_source: "score",
  rolled_back: false,
  is_synthetic: false,
  case_id: null,
  artifacts: { final: FINAL_ID, intermediate: [ITER_ARTIFACT.id] },
  sources: { diagnosis: ["rule_based", "rule_based"], plan: ["rule_based", "rule_based"] },
};

export const DONE: SessionDetail = {
  ...DETAIL,
  status: "done",
  final_decision: "SHIP",
  questions: null,
  process: PROCESS,
  artifacts: [FINAL_ARTIFACT, ITER_ARTIFACT, PREVIEW_NATURAL, PREVIEW_VIVID],
  variants: VARIANTS,
  treatments: [PIPELINE_TREATMENT],
  active_treatment_id: PIPELINE_TREATMENT.id,
  messages: [
    ...DETAIL.messages,
    {
      id: 3,
      role: "user",
      kind: "answers",
      created_at: "2026-10-22T10:01:00+00:00",
      content: {
        answers: { style: "natural" },
        labels: [
          { question: "Bạn thích kết quả theo phong cách nào?", answer: "Tự nhiên (chỉnh nhẹ tay)" },
        ],
        notes: "",
      },
    },
    {
      id: 4,
      role: "assistant",
      kind: "result",
      created_at: "2026-10-22T10:02:00+00:00",
      content: { run_id: RUN_ID, final_artifact_id: FINAL_ID, final_decision: "SHIP" },
    },
  ],
};

/** Phiên sau khi render phiên bản "Tự nhiên" (v2 đang dùng). */
export const RENDERED: SessionDetail = {
  ...DONE,
  artifacts: [...DONE.artifacts, RENDER_ARTIFACT],
  variants: VARIANTS.map((v, i) => ({ ...v, chosen: i === 0 })),
  treatments: [
    PIPELINE_TREATMENT,
    {
      ...PIPELINE_TREATMENT,
      id: "8".repeat(32),
      parent_id: PIPELINE_TREATMENT.id,
      version: 2,
      source: "variant",
      variant_id: VARIANTS[0].id,
      artifact_id: RENDER_ARTIFACT.id,
    },
  ],
  active_treatment_id: "8".repeat(32),
  messages: [
    ...DONE.messages,
    {
      id: 5,
      role: "assistant",
      kind: "render_result",
      created_at: "2026-10-22T10:03:00+00:00",
      content: {
        label: "Tự nhiên",
        artifact_id: RENDER_ARTIFACT.id,
        width: 640,
        height: 480,
        variant_id: VARIANTS[0].id,
      },
    },
  ],
};

export const REFINE_ARTIFACT = art("9".repeat(32), "refine");
export const REFINE_TREATMENT: Treatment = {
  ...PIPELINE_TREATMENT,
  id: "a1".repeat(16),
  parent_id: PIPELINE_TREATMENT.id,
  version: 2,
  source: "refine",
  feedback_text: "da hơi vàng",
  adjustments: [{ kind: "cooler", region: "full", strength: 1 }],
  notes: ["Giảm ám vàng toàn ảnh"],
  refine_source: "rules",
  artifact_id: REFINE_ARTIFACT.id,
  created_at: "2026-10-22T10:04:00+00:00",
};

/** Phiên sau một góp ý "da hơi vàng" (v2 đang dùng). */
export const REFINED: SessionDetail = {
  ...DONE,
  artifacts: [...DONE.artifacts, REFINE_ARTIFACT],
  treatments: [PIPELINE_TREATMENT, REFINE_TREATMENT],
  active_treatment_id: REFINE_TREATMENT.id,
  messages: [
    ...DONE.messages,
    {
      id: 6,
      role: "user",
      kind: "feedback",
      created_at: "2026-10-22T10:04:00+00:00",
      content: { text: "da hơi vàng", base_version: 1 },
    },
    {
      id: 7,
      role: "assistant",
      kind: "refine_result",
      created_at: "2026-10-22T10:04:01+00:00",
      content: {
        version: 2,
        base_version: 1,
        treatment_id: REFINE_TREATMENT.id,
        artifact_id: REFINE_ARTIFACT.id,
        adjustments: REFINE_TREATMENT.adjustments,
        notes: REFINE_TREATMENT.notes,
        source: "rules",
        quality_score: 74.5,
      },
    },
  ],
};
