# Module 4: AI Agent Orchestration & Backend Integration

**Owner:** Person 4  
**Directory:** `src/agent/` and `src/api/`

---

## 1. Responsibilities
This module is the **"Brain & Conductor"** of the system:
1. **`vlm_diagnostician.py`:** Crafts structured prompts to Gemini Multimodal VLM combining the image and technical metrics from Module 1, asking what is wrong, where, and what order of treatment is optimal.
2. **`planner.py` & `executor.py`:** Validates proposed plans against the **fixed agent toolbox** and dispatches actions to Modules 1, 2, and 3.
3. **`state.py` & `graph.py`:** Implements the LangGraph closed-loop state machine (`Analyze -> Diagnose -> Plan -> Process -> Evaluate -> Decide`).
4. **`src/api/`:** Exposes the end-to-end pipeline through a FastAPI REST backend.

---

## 2. The LangGraph State Workflow
```
[Start]
   │
   ▼
analyze_node (Module 1)
   │
   ▼
diagnose_node (Gemini VLM)
   │
   ▼
plan_node (Validate TreatmentPlan)
   │
   ▼
process_node (Modules 2 & 3: detect_regions + apply_operations)
   │
   ▼
evaluate_node (Module 1: PSNR/SSIM or BRISQUE/NIQE)
   │
   ▼
decide_node ──(Quality OK or Iteration >= MAX_ITERATIONS?)──► [Ship / End]
   │ (Not yet & Iteration < MAX_ITERATIONS)
   ▼
re_plan_node ──► process_node (Loop)
```

---

## 3. Important Rules for Person 4 & AI Sessions
- Hard-cap the loop at `MAX_ITERATIONS = 3` to prevent infinite LLM execution.
- Maintain full iteration history in LangGraph state (storing intermediate images, applied filters, parameters, and metric deltas).
- Reject any operation proposed by the VLM that does not exist in the fixed toolbox.
- Write internal reasoning comments in **Vietnamese with proper diacritics**.
- Write logging in **English**.
