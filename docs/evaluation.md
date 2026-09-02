# Evaluation

The simulator is scored with **control-based** checks: each turn is compared to the predicted behavior mode, mastery, misconception targets, and scripted teacher expectations — not to a human gold dialogue.

Run scripts from `backend/` so `app` and `eval` import.

## Quick checks (no API key)

```bash
pytest backend/tests
```

`--skip-llm` on batteries skips LLM *judges* after generation. Session generation itself still calls the student model, so a TAMU key is required for demo/battery runs.

## Group demo

One scripted phone-plans conversation:

```bash
cd backend
python eval/run_group_demo.py
```

Exports land under `backend/eval/results/` (gitignored except `.gitkeep`).

## Group battery

Sweeps the task bank so group quality is a property of the system, not of one problem:

```bash
cd backend
python eval/run_group_battery.py
python eval/run_group_battery.py --skip-llm
```

Scores combine:

- **Deterministic group checks** — observers stay quiet, named students reply, off-task handling, etc.
- **Discourse / quality judges** (LLM) — whether the group talk looks like facilitation, not three parallel 1:1s
- **Cognitive fidelity / persona / believability** — judges on student turns

Score a saved group export:

```bash
cd backend
python eval/score_group_export.py path/to/export.json
python eval/score_group_quality.py path/to/export.json --skip-llm
```

Score a pst-training-game merged export (`integration.backend_export`):

```bash
cd backend
python eval/score_pst_merged_export.py path/to/merged.json
```

Facilitation gold (who should speak after a teacher move):

```bash
cd backend
python eval/score_facilitation_gold.py --parser hybrid
```

Judges use `OPENAI_MODEL`. Student generation uses `STUDENT_OPENAI_MODEL` when set. Keeping those distinct avoids self-judging bias.

## Fixtures

- `backend/eval/fixtures/teacher_scripts.yaml` — scripted teacher moves
- `backend/eval/fixtures/group_facilitation_gold.yaml` — speak-constraint gold set
- `backend/eval/fixtures/battery_fixtures.py` — loads scripts and maps turns to roles

Detection patterns for deterministic math-slip checks live on catalog entries (`stack_level == 1`) in `backend/config/misconception_catalog.json`.
