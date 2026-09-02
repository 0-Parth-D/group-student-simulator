# Architecture — why the simulator is built this way

This note explains the decisions behind the group student simulator. It is for collaborators who need to change behavior without rediscovering the rationale.

## What the system is

A teacher facilitates a **three-student math discussion**. Each student is an LLM role-play constrained by:

1. A **learning-progression knowledge graph** (what they know / almost know / do not know)
2. A **misconception catalog** (how wrong ideas sound in student language)
3. A **Big Five personality** used as *expression style*, not as knowledge
4. A **group speak policy** (who must speak, who may speak, who must listen)

The LLM never invents new math constructs. It only enacts the state the router already computed.

## Why a declarative knowledge graph

Constructs and prerequisite edges live in `backend/config/knowledge_graph.yaml`. They come from the NCIEA **Learning Progressions Framework** (Hess et al., 2010/2011) — strands NO, PRF, and SE for middle school. Every node cites an LPF learning target / progress indicator.

We do **not** let the LLM build the graph at session start. Early prototypes did that and produced inconsistent, non-replicable student knowledge. The rule is:

> Only nodes already in `knowledge_graph.yaml` are valid.  
> Only the behavior / mastery code writes level updates.

PISA competency attributes and DLM linkage-level wording are overlays. They do not define graph structure.

## Mastery scale (0–3)

| Level | Label | Meaning in routing |
|-------|--------|--------------------|
| 0 | missing | Never activated; the student must not reference it as known |
| 1 | bad | Precursor / faulty understanding; likely error enactment |
| 2 | partial | Some of the target idea; incomplete or fragile |
| 3 | good | At or near the grade-level target |

Level prose is resolved by `app.construct_text.mastery_descriptor`. Fraction/ratio nodes use DLM linkage wording (IP/DP/PP/T); later linear-change nodes use `lpf_stack` text written from LPF indicators.

## Per-turn loop (math turns)

1. Classify the teacher move (`social`, `math_scaffold`, `math_eval`, `vague_*`, `mixed`, …).
2. Detect scaffolding and apply a short-lived `scaffold_boost` per construct.
3. Route behavior: confused / partial / normal, plus which misconception to enact.
4. Generate a student reply (cognitive state first, personality second).
5. Score reply quality and update mastery with that student’s learning profile (`gain_rate`, `slip_rate`, `retention`, `scaffold_sensitivity`).

Social turns skip mastery updates. Vague acknowledgments (“ok”, “yes”) can trigger a stall so the student does not invent the next teaching move.

## Cognitive state over personality

Math prompts put a non-negotiable **cognitive state** block above Big Five style. Personality controls tone and length, not whether Jordan suddenly “knows” equivalent fractions. This follows the PCL split between role identity and surface expression (Ji et al., 2025) and fixes failures where high neuroticism overrode a WRONG-mode instruction.

## Tasks and answers

`backend/config/task_metadata.json` is the task bank: constructs, target stack level, PISA Q-matrix, expected answer / strategy. Word problems are disambiguated before tagging. Expected answers are computed from metadata strategies, not hardcoded per demo scenario.

## Misconceptions

`backend/config/misconception_catalog.json` stores error types with detection patterns (regex) used by deterministic eval. At runtime, optional local **Qdrant** retrieves concrete error utterances; if Qdrant is off or locked, metadata fallback still works.

## Group facilitation

Group sessions reuse the same student reply pipeline, with extra policy:

- **Speak constraints** — parse the teacher utterance into must / may / must-not speakers. Default parser is **hybrid**: LLM parse plus a regex veto on `must_not` (so “Jordan, please watch” cannot be overridden by a chatty model).
- **Self-select** — on open / discuss / critique moves, students volunteer instead of all answering.
- **Multi-round peer continuation** — after an open floor, peers may keep talking (LangGraph loop) until nobody volunteers or a safety cap is hit.
- **Track B receptivity** — session-fixed scaffold sensitivity; dumping the full answer does not raise mastery and surfaces a teaching warning.

PST starter group: Maya + Jordan on `phone_plans_linear_01` (`/api/pst/sessions`). Research default: Alex, Maya, Jordan (`/api/group-sessions`).

Social teacher turns turn **claim lock off** and skip the game-flow peer chain so greetings do not restated Plan A/B. Math turns keep opposing claims (Maya table/Plan B vs Jordan rate/Plan A).

## 1:1 path

`backend/app/sessions.py` remains as a library (teacher opener + KG summary helpers). There is **no** `/api/sessions/*` HTTP API in this repo.

## What we deliberately left out

- Demo frontend (pst-training-game owns UI)
- Coach / gate / hint / reflection agents (frontend-only)
- Local Llama / LoRA student backends
- Required Neo4j (optional; YAML graph is the default)
- Letting the model add KG nodes
- Full CCSS grade 6–8 coverage (current graph is the LPF slice we teach with)
