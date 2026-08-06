# Evaluation

The simulator is scored with **control-based** checks: each turn is compared to the predicted behavior mode, mastery, misconception targets, and scripted teacher expectations — not to a human gold dialogue.

## Quick checks (no API key)

```bash
pytest
```

`--skip-llm` on batteries skips LLM *judges* after generation. Session generation itself still calls the student model, so a TAMU key is required for demo/battery runs.

## Group demo

One scripted phone-plans conversation with Jordan, Sam, and Alex:

```bash
python eval/run_group_demo.py
```

Exports land under `eval/results/` (gitignored except `.gitkeep`).

## Group battery

Sweeps the task bank so group quality is a property of the system, not of one problem:

```bash
python eval/run_group_battery.py
python eval/run_group_battery.py --skip-llm
```

Scores combine:

- **Deterministic group checks** — observers stay quiet, named students reply, off-task handling, etc.
- **Discourse / quality judges** (LLM) — whether the group talk looks like facilitation, not three parallel 1:1s
- **Cognitive fidelity / persona / believability** — shared with the 1:1 battery where applicable

Score a saved group export:

```bash
python eval/score_group_export.py path/to/export.json
python eval/score_group_quality.py path/to/export.json --skip-llm
```

Facilitation gold (who should speak after a teacher move):

```bash
python eval/score_facilitation_gold.py --parser hybrid
```

## 1:1 battery

Still maintained. Each scenario is a profile × bank task with a scripted teacher follow-up sequence:

```bash
python eval/run_battery.py
python eval/run_battery.py --skip-llm
python eval/run_eval.py path/to/scenario.json
```

Turn-level deterministic gates include:

- **expert_slip** — a struggling student should not produce the fully correct procedure unprompted
- **persona_direction** — replies should not contradict the intended persona polarity

LLM judges (when not skipped):

| Metric | Question |
|--------|----------|
| Cognitive fidelity | Did the student enact the intended construct / misconception state? |
| Behavioral consistency | Did the reply match the expected behavior for that teacher move? |
| Persona stability | Does sampled talk still sound like this Big Five profile? |
| Believability | Would this pass as a middle-schooler (not an expert tutor in disguise)? |

Judges use `OPENAI_MODEL`. Student generation uses `STUDENT_OPENAI_MODEL` when set. Keeping those distinct avoids self-judging bias.

## Fixtures

- `eval/fixtures/teacher_scripts.yaml` — 1:1 battery teacher moves / scaffold cues
- `eval/fixtures/group_facilitation_gold.yaml` — speak-constraint gold set
- `eval/fixtures/battery_fixtures.py` — loads scripts and maps turns to roles

Detection patterns for deterministic math-slip checks live on catalog entries (`stack_level == 1`) in `config/misconception_catalog.json`.
