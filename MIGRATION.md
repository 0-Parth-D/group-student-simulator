# Migration from student-simulation

Extract of the **group + PST** backend from `student-simulation` into this repo. Target is the sole LP backend for `pst-training-game` (`VITE_AI_SOURCE=backend`).

SOURCE: `…/implementations/student-simulation`  
TARGET: this repo  
FE consumer (read-only contract): `…/implementations/pst-training-game`

## What already existed in TARGET

This repo previously used a package split (`src/app/group|student|knowledge|llm`) plus a stand-in `frontend/` and 1:1 `/api/sessions`. That layout was **superseded**. SOURCE group/PST implementations are copied into SOURCE-parity `backend/app/` modules. The old `src/`, `frontend/`, root `config/`, root `tests/`, and root `eval/` trees were removed so there is a single source of truth.

No intentional TARGET divergences were kept on the group/PST path.

## Files copied (from SOURCE)

### App modules → `backend/app/`

`group_sessions.py`, `group_orchestrator.py`, `group_speak_policy.py`, `group_context.py`, `group_http.py`, `game_flow_orchestrator.py`, `group_graph.py`, `group_constraint_llm.py`, `fight_stance.py`, `pst_api.py`, `live_hud.py`, `turn_classifier.py`, `turn_logic.py`, `prompts.py`, `student_prompt_layers.py`, `personality.py`, `response_refinement.py`, `behavior_router.py`, `misconception_store.py`, `knowledge_graph.py`, `kg_store.py`, `kg_config.py`, `constructs.py`, `construct_text.py`, `learning_profile.py`, `learning_receptivity.py`, `scaffold_detector.py`, `help_seeking.py`, `task_tagger.py`, `task_answer.py`, `demo_students.py`, `expected_behavior.py`, `eval_log.py`, `llm.py`, `llm_config.py`, `models.py`, `data.py`, `config.py`, `mistakes.py`, `pisa.py`, `prediction.py`, `session_shared.py` (opener template + `kg_summary`; replaces 1:1 `sessions.py`).

### Config → `backend/config/`

`knowledge_graph.yaml`, `misconception_catalog.json`, `task_metadata.json`, `pisa_attributes.json`, `rational_number_constructs.json`

### Infra

`backend/run.py`, `backend/requirements.txt`, root `requirements.txt` (`-r backend/requirements.txt`), `.env.example`

### Tests → `backend/tests/`

MUST: `test_pst_api`, `test_fight_session`, `test_fight_stance`, `test_group_orchestrator`, `test_group_off_task`, `test_game_flow_orchestrator`, `test_turn_classifier`, `test_response_refinement`, `test_live_hud`, `test_learning_receptivity`, `test_scaffold_detector`, `test_student_prompt_layers`, `test_expected_behavior`, `test_llm_config`, `test_detection_patterns`, `test_facilitation_gold`, `test_believability`

Companions: `test_eval_log`, `test_help_seeking`, `test_ocean_personality`, `test_group_graph`, `test_group_deterministic_checks`, `test_group_quality`, `test_group_script`

### Eval → `backend/eval/`

`run_group_battery.py`, `run_group_demo.py`, `group_deterministic_checks.py`, `group_quality.py`, `group_discourse.py`, `group_script.py`, `deterministic_checks.py`, `score_pst_merged_export.py`, `believability.py`, `cognitive_fidelity.py`, `persona_stability.py`, `log_schema.py`, `score_group_quality.py`, `score_group_export.py`, `score_facilitation_gold.py`, plus `eval/fixtures/` (`group_facilitation_gold.yaml`, gold Python, `teacher_scripts.yaml`, `battery_fixtures.py`)

### Docs

`backend/docs/PST-CONNECTOR-API.md`, `backend/docs/LIVE-HUD-API.md`, `docs/design-decisions.md`, `docs/lpf-source-mapping.md`

## Files adapted

| File | Change |
|------|--------|
| `backend/app/main.py` | Slim FastAPI: health, profiles, group routes, PST router, CORS. **No** static frontend mount, **no** `/api/sessions/*`. |
| Root `run.py` | PYTHONPATH → `backend/` (was `src/`). |
| `pytest.ini` | `pythonpath = backend`, `testpaths = backend/tests`. |
| `.env.example` | Group/PST vars only; llama/finetune knobs stripped. |
| `docs/architecture.md` | Module paths `app.*` (not `app.knowledge.*`); 1:1 HTTP removed. |
| `docs/evaluation.md` | Paths under `backend/eval/`; 1:1 battery dropped. |
| `README.md` | Backend-only + pst-game env. |

## Files omitted (SKIP)

| Item | Why |
|------|-----|
| `frontend/` | pst-training-game owns UI |
| `notebooks/` | research scratch |
| `backend/finetune/`, `llama_infer.py` | training / local Llama |
| `/api/sessions/*` | 1:1 tutoring not in this product |
| `eval/run_battery.py`, `eval/run_eval.py` | 1:1 batteries |
| `eval/behavioral_consistency.py` | 1:1 judge only |
| SOURCE demo frontend, PRODUCT.md | not this service |
| `.env`, API keys, `qdrant_storage/` | secrets / local indexes |
| FE agents (`coach.js`, gate, hint, reflection) | stay on pst-training-game |

## Import refactors

None on the copied modules. They keep `from app.xxx` as in SOURCE. Canonical layout is `backend/app/*.py` (not the previous `src/app/{group,student,knowledge,llm}` packages).

`sessions.py` is **not** copied — group code uses `session_shared.py` instead (no 1:1 `SessionStore`).

## Re-sync (2026-03-02)

From `student-simulation` via `scripts/sync_from_student_simulation.py`:

- **Added:** `reasoning_warrant.py`, `fight_progress.py`, PST fixture replay (`discourse_progress`, `run_pst_script_replay`, `compare_pst_fixture_runs`, `pst_phone_plans_facilitation.*`), and matching tests.
- **Patched on copy:** `group_sessions.py` → `session_shared` for `TEACHER_OPENER_TEMPLATE`.
- **Still skipped:** `sessions.py`, `llama_infer.py`, `run_battery.py`, `run_eval.py`, `behavioral_consistency.py`, `main.py`.

```bash
python scripts/sync_from_student_simulation.py
pytest backend/tests
```

## API parity checklist (PST-CONNECTOR-API)

- [x] `POST /api/pst/sessions` → defaults `profile_ids=maya,jordan`, `task_id=phone_plans_linear_01`, auto-`start()`, returns `live_hud`
- [x] `POST /api/pst/sessions/{id}/turn` → default `stream=true` NDJSON: `speak_plan` → `thinking` → `reply`* → `done`
- [x] Teacher enrichment: `board_note`, `proximity` appended to message
- [x] `live_hud` on `done`: roster with `implied_plan`, `claim_id`, `active_misc_id`, barriers, personality, knowledge
- [x] Stubs `/api/pst/audio/*`, `/api/pst/board/analyze` → 501
- [x] Research `/api/group-sessions/*` + `/live` + `/roster` unchanged
- [x] CORS via `CORS_ORIGINS` (empty = allow all)

## Behavior preserved from SOURCE

1. **Fight scenario:** Maya Plan B (table), Jordan Plan A (rate/fee-blind); pinned `active_misc_id`; `FE_SEED_LINES` match `pst-training-game/src/data/scenario.js` `seededMessages`.
2. **CLAIM LOCK:** On for math turns with `implied_plan`; **OFF** on `social` / `off_topic` / stall (`claim_lock_active` in `turn_classifier.py`).
3. **Social wave:** No game-flow peer chain after pure social teacher turns (`social_or_off_topic` stop).
4. **Critic:** No claim/misconception demand on social turns (`response_refinement.py`).
5. **Prompt diet:** One cognitive spine, one playbook pattern, PISA gated off for phone plans.

## What remains FE-only

In `pst-training-game` (not this repo):

- Coach warnings / coaching panel (`src/prompts/coach.js`)
- Gate (whether a teacher message is sent / blocked)
- Hint bar
- Reflection
- Mic / Whisper transcription (`VITE_OPENAI_API_KEY`)
- Board vision / canvas
- Seeded Maya/Jordan openers rendered in the chat UI (backend `FE_SEED_LINES` must stay in sync)

When `VITE_AI_SOURCE=backend`, `discussionEngine.js` replaces the local orch/student loop with streamed `/api/pst` turns; checker inject is skipped; coach still runs on the FE.

## Follow-up issues (not implemented)

1. **Session phase / breakthrough memory** — Jordan can revert to coefficient-only after solving \(x=100\) because claim lock + pinned misc never release. Add `fight_phase: fight | nuanced | resolved` and inject breakthrough bullets into the system prompt.
2. **Rolling summary** — long sessions exceed `GROUP_TRANSCRIPT_WINDOW=10` and `GROUP_PRIVATE_HIST_MAX=12`; add a per-student session summary updated each turn.
3. **Jordan restraint** — honor teacher “let Maya speak” for 1–2 turns in the orchestrator.

## Re-sync from SOURCE

When `student-simulation` changes, refresh TARGET without copying 1:1 HTTP or `main.py`:

```bash
cd group-student-simulator
python scripts/sync_from_student_simulation.py --dry-run
python scripts/sync_from_student_simulation.py
cd backend && pytest
```

Prompt / checklist for another agent: [`docs/MIGRATION_PROMPT.md`](docs/MIGRATION_PROMPT.md).

## Last verification (2026-09-01)

- `python scripts/sync_from_student_simulation.py` — 100 files synced; `main.py` preserved
- `pytest backend/tests` — **345 passed** (includes `test_detection_patterns`, `test_facilitation_gold`, `test_believability`)
- App modules match SOURCE except `main.py` (slim) and omitted `llama_infer.py`
