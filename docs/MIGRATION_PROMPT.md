# Migration prompt (reference)

Use this prompt in a **new chat** when re-syncing or bootstrapping the clean backend from `student-simulation`.

**Repos:**

| Role | Path |
|------|------|
| SOURCE | `…/implementations/student-simulation` |
| TARGET | `…/implementations/group-student-simulator` |
| FE consumer | `…/implementations/pst-training-game` (read-only) |

**Automated sync (preferred):**

```bash
cd group-student-simulator
python scripts/sync_from_student_simulation.py --dry-run
python scripts/sync_from_student_simulation.py
cd backend && pytest
```

---

## Task prompt (paste below into another agent)

```
# Task: Extract / maintain a clean group-student simulator backend from student-simulation

Study BOTH repositories before changing anything.

SOURCE: student-simulation (monolith)
TARGET: group-student-simulator (backend-only LP service for pst-training-game)
FE: pst-training-game (VITE_AI_SOURCE=backend) — contract reference only

## Goal

Backend-only FastAPI that:
1. Simulates teacher + 2–3 LP students (personality, KG, misconceptions, orchestration).
2. Exposes /api/pst/* for pst-training-game.
3. Keeps /api/group-sessions/* + Live HUD for research/eval.
4. Does NOT include demo frontend, notebooks, finetune, or FE coach/gate/hint agents.

## MUST copy

App: group_sessions, group_orchestrator, group_speak_policy, group_context, group_http,
game_flow_orchestrator, group_graph, group_constraint_llm, fight_stance, pst_api, live_hud,
turn_classifier, turn_logic, prompts, student_prompt_layers, personality, response_refinement,
behavior_router, misconception_store, knowledge_graph, kg_store, kg_config, constructs,
construct_text, learning_profile, learning_receptivity, scaffold_detector, help_seeking,
task_tagger, task_answer, demo_students, expected_behavior, eval_log, llm, llm_config,
models, data, config, mistakes, pisa, prediction, sessions.py (library only).

Config: knowledge_graph.yaml, misconception_catalog.json, task_metadata.json,
pisa_attributes.json, rational_number_constructs.json.

Docs: PST-CONNECTOR-API.md, LIVE-HUD-API.md.

Tests: test_pst_api, test_fight_*, test_group_*, test_turn_classifier,
test_response_refinement, test_live_hud, test_learning_receptivity, test_scaffold_detector,
test_student_prompt_layers, test_expected_behavior, test_llm_config, test_detection_patterns,
test_facilitation_gold, test_believability.

Eval: run_group_battery, run_group_demo, group_deterministic_checks, group_quality,
group_discourse, score_pst_merged_export, fixtures/group_facilitation_gold.yaml.

## MUST NOT copy

frontend/, notebooks/, backend/finetune/, llama_infer.py, /api/sessions/* HTTP routes,
FE coach/gate/hint/reflection agents.

## TARGET main.py

Slim: health, profiles, group routes, pst router, CORS. No static frontend mount.

## Behavior to preserve

1. Fight: Maya Plan B / Jordan Plan A; FE_SEED_LINES match scenario.js
2. CLAIM LOCK off on social/off_topic/stall (claim_lock_active)
3. Social wave: no game-flow after social teacher turns
4. Critic: no claim/misc on social turns
5. Prompt diet for phone plans

## Verify

pytest backend/tests
POST /api/pst/sessions → maya,jordan, live_hud
Greeting turn → social, no forced Plan restatement

Deliver: MIGRATION.md update, README, passing tests.
```

See also [`MIGRATION.md`](../MIGRATION.md) for the file-level inventory.
