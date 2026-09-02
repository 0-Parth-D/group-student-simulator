# Design Decisions — KG + Learning Progression Layer

## 0. Source of the progression

Nodes and edges in `backend/config/knowledge_graph.yaml` come from the **NCIEA
Learning Progressions Framework** (Hess et al., 2010/2011) — strands NO, PRF, and SE,
middle-school grade span. Every construct cites the LPF learning target and progress
indicators it is drawn from; every prerequisite edge cites the PI ordering that
justifies it. See `docs/lpf-source-mapping.md` for the full citation table and the
open questions for content review.

DLM linkage levels are used only for the per-node mastery-level wording, and PISA
attributes are a separate overlay. Neither determines the graph structure.

## 1. Mastery Scale: 4-Level Stack

We map student mastery to four levels, using the DLM linkage-level taxonomy
(IP → DP → PP → T → S) as the source of the level descriptions. Constructs added from
LPF PRF progress indicators have no vetted DLM alignment yet and carry `lpf_stack`
descriptions written from LPF wording instead.

| Level | Label    | Operational Definition                                       | DLM Linkage Equivalent |
|-------|----------|--------------------------------------------------------------|------------------------|
| 0     | missing  | Construct never observed in any past task record             | (below IP)             |
| 1     | bad      | Attempted ≥1 task targeting this construct; stack_level ≤ 1  | IP / DP                |
| 2     | partial  | Demonstrated understanding at stack_level 2–3               | PP                     |
| 3     | good     | Reached stack_level ≥ target_stack_level − 1               | T / S                  |

### Definitions of "missing / bad / good" for routing
- **missing**: `mastery_level == 0` — node was never activated; LLM must not reference it.
- **bad**:     `mastery_level == 1` — node is active but performance shows only precursor understanding.
- **good**:    `mastery_level >= 3` — node is at or near grade-level target; prerequisite is satisfied.

### Reading level descriptions in code

`app/construct_text.mastery_descriptor(construct_id, level)` resolves the prose for a
level. Because the table above gives ranges rather than a lookup, the code picks one
canonical DLM level per mastery level:

| Mastery | `lpf_stack` key | `dlm_linkage_levels` key |
|---------|-----------------|--------------------------|
| 0       | (none)          | `IP`                     |
| 1       | `bad`           | `DP`                     |
| 2       | `partial`       | `PP`                     |
| 3       | `good`          | `T`                      |

`T` anchors mastery 3 because `target_stack_level` is 3 for every construct; `S`
describes above-target work the scale does not reach. A construct carrying `lpf_stack`
has no descriptor at level 0, so the accessor falls back to a generic phrase built from
the construct label. Every construct must supply one of the two blocks —
`tests/test_expected_behavior.py` enforces this.

---

## 2. Per-Turn Mastery Update Policy (v2 — per-student learning profiles)

After **each student reply** on math turns (not pure social chat):

1. Classify teacher turn (`turn_classifier.py`): `social`, `math_scaffold`, `math_eval`, `vague`, `mixed`.
2. On math turns, detect teacher scaffolding (`scaffold_detector.py`) and apply session-local `scaffold_boost` per construct.
3. Evaluate student reply quality (`infer_reply_quality`).
4. Update mastery via `update_mastery_state()` with per-student `LearningProfile`:
   - `gain_rate`, `slip_rate`, `retention`, `scaffold_sensitivity`, `forget_rate`
   - Per-construct gain via `effective_gain(construct_id)` = `gain_rate × construct_gain_modifiers[construct]`
   - Correct replies may level up probabilistically; scaffold boost increases gain chance.
   - Wrong/confused replies may drop unless recent scaffold protects the student.
5. Decay `scaffold_boost` each turn by `forget_rate`.
6. Re-route behavior + misconceptions via `behavior_router.py`.

Social turns skip mastery updates and use personality-driven conversational prompts.

### Teacher scaffolding

When the teacher explains a construct (LCD, equivalent fractions, etc.), `scaffold_boost[construct]` increases before the student speaks. If boost exceeds threshold, the student prompt allows using **that step only** — not the full solution.

---

## 2b. Legacy fixed-step policy (superseded)

```
if reply_quality == "correct":
    node.mastery_level = min(node.mastery_level + 1, 3)
elif reply_quality == "partial":
    pass
elif reply_quality in ("wrong", "confused"):
    node.mastery_level = max(node.mastery_level - 1, 0)
```

Still available via `update_observation()` which delegates to `update_mastery_state()` with default profile.

---

## 2c. Conversational turn routing

| Turn mode | Prompt focus |
|-----------|----------------|
| `social` | Big Five voice, small talk; no LP mistake blocks |
| `math_scaffold` | LP behavior + misconceptions + mastery |
| `math_eval` | Same as scaffold; may include scaffold detection |
| `mixed` | Brief social + math if asked |
| `vague_acknowledgment` | Stall injection; consecutive vague acknowledgments trigger stronger stall |
| `vague_directive` | Teacher directs next step ("go on", "what next") — student may continue |

### Stall policy (Phase 1)

When **2 or more** of the last 3 teacher turns are classified as `vague`, `should_student_stall()` activates:

- System prompt adds `[STALL — teacher gave no direction]` block
- User message gets `STALL_INJECT` reminder
- Scaffold boost is **not** applied on stall turns
- Mastery state is **not** updated on stall turns

Phase 2 will split `vague` into `vague_acknowledgment` vs `vague_directive`; stall logic uses `STALL_LABEL` constant for easy migration.

---

## 2d. Prompt priority: cognitive state over personality (Phase 1)

Math-turn system prompts are assembled in this order:

1. **`[COGNITIVE STATE — NON-NEGOTIABLE]`** — behavior mode, construct gaps, required error patterns from the LP router. Explicit override clause when traits conflict.
2. **`[HOW YOU EXPRESS THIS STATE]`** — BF-TC as expression style only (tone, length), not what the student knows.
3. LP detail blocks (construct profile, mastery, PISA, playbook, etc.)

Rationale: Ji et al. (2025) PCL — separate role identity from surface expression. Fixes Jordan-style failures where High Neuroticism anxiety rules overrode WRONG-mode confidence.

Social turns still use full BF-TC trait blocks (no cognitive override).

Demo student learning parameters (BKT-inspired, hand-set):

| Student | gain_rate | slip_rate | retention | scaffold_sensitivity | forget_rate |
|---------|-----------|-----------|-----------|----------------------|-------------|
| Alex    | 0.35      | 0.05      | 0.95      | 0.5                  | 0.02        |
| Jordan  | 0.10      | 0.20      | 0.70      | 1.4                  | 0.10        |
| Sam     | 0.20      | 0.12      | 0.82      | 0.9                  | 0.05        |

Per-construct gain modifiers (defaults): `part_whole` 1.2, `fraction_equivalence` 0.6, `ratio_concept` 1.0, `proportional_reasoning` 0.5.

---

## 2e. Phase 2 — Smarter signals (LLM gpt-4o-mini)

### Teacher-turn classifier (`turn_classifier.py`)

Labels: `math_eval`, `math_scaffold`, `vague_acknowledgment`, `vague_directive`, `social`, `off_topic`, `mixed`.

- LLM classification with last-3-turn context; heuristic fallback on failure.
- Stall policy (`STALL_LABEL = vague_acknowledgment`) does **not** trigger on `vague_directive` ("go on", "what next").

### Mastery quality (`knowledge_graph.py`)

- `infer_reply_quality_semantic()` — rubric scores 0–3 via LLM; maps to `correct` / `partial` / `wrong` / `confused`.
- Heuristic fallback when LLM fails.
- Mastery updates gated: only `math_eval` and `math_scaffold` teacher turns update mastery.

### Worked-example cap (`behavior_router.py`, `scaffold_detector.py`)

When `math_scaffold` + `scaffold_is_worked_example()`:
- Construct added to `session.scaffolded_constructs`
- `mode_caps[construct] = PARTIAL_ATTEMPT_THEN_STUCK`

### Task disambiguation (`task_answer.py`, `task_tagger.py`)

Word problems run `disambiguate_word_problem()` before KC tagging.

---

## 2f. Phase 3 — Retrieval + evaluation

### Misconception instance index (`misconception_store.py`)

- Persistent local Qdrant at `backend/qdrant_storage` (`QDRANT_PATH` env).
- Collection `misconception_instances`: 3+ concrete error utterances per catalog `misc_id`.
- `retrieve_error_instance()` — per-turn, filtered by `active_misc_id`, query includes teacher utterance + construct.
- Catalog collection `misconceptions` retained for session-start retrieval.

### Detection patterns (deterministic eval)

- Each `stack_level == 1` catalog entry carries `detection_patterns` (regex) used by
  `eval/deterministic_checks.py` (`expert_slip`, Morgan `persona_direction`).
- Level 0 avoidance/shutdown patterns may exist for documentation but are **not**
  treated as L1 math slips.
- Patterns prefer wrong answer forms and procedure language over task-specific nouns.
- Semantic paraphrase matching over `misconception_instances` is future work; regex
  remains the CI / `--skip-llm` gate.

### Few-shot error anchor (`build_error_anchor`)

Injected into `student_prompt` via `error_anchor` (replaces flat `prompt_cue` when instance retrieved).

### Error enactment verification (`response_refinement.py`)

After `is_too_expert()` check, `check_error_enacted()` (gpt-4o-mini YES/NO) with up to 3 generate attempts.

### Offline eval (`backend/eval/`)

- `cognitive_fidelity.py` — construct/misconception fidelity (1–5)
- `persona_stability.py` — Big Five probe from sampled turns
- `run_group_battery.py` — `cd backend && python eval/run_group_battery.py`
- `run_group_demo.py` — scripted phone-plans group walkthrough

---

## 3. Architecture

**Supported path:** LP (`force_lp=True`, default) — declarative `knowledge_graph.yaml`, misconception catalog, behavior router.

**Deprecated path:** Legacy LLM KG (`kg.py`, `force_lp=False`, `/api/pipeline/run`) — kept for offline eval only.

| In scope (v2)                                    | Out of scope (later)          |
|--------------------------------------------------|-------------------------------|
| Confrey rational-number constructs (7 nodes)     | Full CCSS Grade 6-8 map       |
| PISA competency weak-attribute routing           | Real student data (ASSIST)    |
| Per-student learning profiles                    | DKT / BKT full estimation     |
| Teacher scaffolding + scaffold_boost           | Neo4j required in production  |
| Conversational turn modes                        | Full evaluation pipeline      |
| Synthetic demo students (alex/jordan/sam)        |                               |
| Behavior router: CONFUSED / PARTIAL / NORMAL     |                               |


**v1 routing rule:**
> The LLM never adds new nodes to the KG directly.
> Only `behavior_router.py` writes mastery level updates,
> and only nodes already present in `knowledge_graph.yaml` are valid.
