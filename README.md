# Middle-School Student Simulator

Simulated middle-school math students for **small-group facilitation practice**. Teachers talk with three students at once. Each student has a personality, a learning-progression mastery state, and catalogued misconceptions.

This is the shareable codebase. A stand-in web UI is included so you can run locally; the backend API is the contract for later frontends.

## Requirements

- Python 3.11+
- A [TAMU Chat](https://chat-api.tamu.ai) API key
- macOS or Windows

## Setup

```bash
python -m venv .venv
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

```env
TAMU_CHAT_API_KEY=sk-your-tamu-key-here
OPENAI_MODEL=protected.gpt-5.4
```

Optional: set `STUDENT_OPENAI_MODEL` (for example `protected.Claude Sonnet 4.6`) so student replies use a different model than judges/classifiers.

## Run the UI

```bash
python run.py
```

Open http://localhost:8000

1. Pick a group (phone-plans demo is the recommended first run).
2. Prepare session → Start practice.
3. Facilitate: call a student by name, open the floor (“Discuss together”), or ask someone to watch.

Facilitator answer key for the default task: [`docs/phone-plans.md`](docs/phone-plans.md).

## Tests

Deterministic unit tests do not need an API key:

```bash
pytest
```

## Evaluation

Live group demo (needs a TAMU key — generates a full conversation):

```bash
python eval/run_group_demo.py
```

Batteries also generate sessions (key required). `--skip-llm` skips LLM *judges* after the run:

```bash
python eval/run_group_battery.py --skip-llm
python eval/run_battery.py --skip-llm
```

See [`docs/evaluation.md`](docs/evaluation.md).

## Project layout

```
config/          Declarative research content (KG, tasks, misconception catalog)
src/app/
  group/         Small-group sessions, speak policy, peer continuation
  student/       Profiles, personality, learning, 1:1 session core (API + eval)
  knowledge/     Learning progression, tasks, misconceptions, turn analysis
  llm/           TAMU Chat client and role profiles
  main.py        FastAPI app
eval/            Batteries, judges, fixtures
tests/
frontend/        Temporary static UI (group practice only)
docs/
```

## API (group-first)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health + LP stack status |
| GET | `/api/profiles` | Student profiles |
| POST | `/api/group-sessions` | Create a 3-student group session |
| POST | `/api/group-sessions/{id}/start` | Start discussion |
| POST | `/api/group-sessions/{id}/message` | Teacher move |
| POST | `/api/group-sessions/{id}/advance` | Nudge peer continuation |
| GET | `/api/group-sessions/{id}/export` | Export transcript + eval fields |
| DELETE | `/api/group-sessions/{id}` | End session |

One-to-one tutoring endpoints (`/api/sessions/...`) remain for the eval battery and later tooling. They are not exposed in the stand-in UI.

## Design notes

Why mastery is 4-level, why the KG is declarative, and how group speak constraints work: [`docs/architecture.md`](docs/architecture.md).
