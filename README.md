# Group Student Simulator

Backend-only FastAPI service for **small-group LP student simulation**. This is the sole learning-progression backend for the ArguMath PST training game (`pst-training-game` with `VITE_AI_SOURCE=backend`).

Teachers talk with 2–3 simulated students. Each student has a Big Five personality, a learning-progression mastery state, and catalogued misconceptions. Gate, coach, hint, and reflection agents stay on the **frontend**.

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

Edit `.env` and set at least:

```env
TAMU_CHAT_API_KEY=sk-your-tamu-key-here
OPENAI_MODEL=protected.gpt-5.4
```

Optional: `STUDENT_OPENAI_MODEL` (for example `protected.Claude Sonnet 4.6`) so student replies use a different model than judges/classifiers. See `.env.example` for group, CORS, and Qdrant knobs.

## Run

```bash
cd backend
python run.py
```

API: http://localhost:8000  
Health: http://localhost:8000/api/health

From the repo root, `python run.py` also works (same app, port 8000).

## pst-training-game

In the game repo `.env` / `.env.local`:

```env
VITE_AI_SOURCE=backend
VITE_STUDENT_SIM_URL=http://localhost:8000
```

Then start this backend **before** the game. The game calls:

| Method | Path | Role |
|--------|------|------|
| `POST` | `/api/pst/sessions` | Briefing Start — Maya + Jordan, phone plans, auto-start |
| `POST` | `/api/pst/sessions/{id}/turn` | Teacher turn (NDJSON stream by default) |
| `DELETE` | `/api/pst/sessions/{id}` | End Discussion |

Audio/board stubs return `501` (mic/Whisper and board vision stay on the FE). Full contract: [`backend/docs/PST-CONNECTOR-API.md`](backend/docs/PST-CONNECTOR-API.md). Live HUD: [`backend/docs/LIVE-HUD-API.md`](backend/docs/LIVE-HUD-API.md).

## Tests

Deterministic unit tests do not need an API key:

```bash
pytest backend/tests
```

or `cd backend && pytest`.

## Evaluation

From `backend/` (TAMU key required for live generation):

```bash
python eval/run_group_demo.py
python eval/run_group_battery.py --skip-llm
```

See [`docs/evaluation.md`](docs/evaluation.md). Facilitator answer key: [`docs/phone-plans.md`](docs/phone-plans.md).

## Project layout

```
group-student-simulator/
  backend/
    app/           Group sessions, PST facade, LP stack
    config/        KG, tasks, misconception catalog
    docs/          PST connector + Live HUD API
    eval/          Group batteries and judges
    tests/
    run.py
  data/            Local Qdrant index (gitignored)
  docs/            Architecture, evaluation, migration notes
  scripts/         sync_from_student_simulation.py
  run.py           Dev entry (same as backend/run.py)
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health + LP stack status |
| `GET` | `/api/profiles` | Student profiles |
| `POST` | `/api/pst/sessions` | Create + auto-start FE starter (Maya/Jordan) |
| `POST` | `/api/pst/sessions/{id}/turn` | Teacher turn (stream default) |
| `DELETE` | `/api/pst/sessions/{id}` | End PST session |
| `POST` | `/api/group-sessions` | Research group session (default 3 students) |
| `POST` | `/api/group-sessions/{id}/start` | Start discussion |
| `POST` | `/api/group-sessions/{id}/message` | Teacher move (`?stream=true` for NDJSON) |
| `POST` | `/api/group-sessions/{id}/advance` | Nudge peer continuation |
| `GET` | `/api/group-sessions/{id}/live` | Live HUD snapshot |
| `GET` | `/api/group-sessions/{id}/roster` | Roster + knowledge |
| `GET` | `/api/group-sessions/{id}/export` | Transcript + eval fields |
| `DELETE` | `/api/group-sessions/{id}` | End session |

There is **no** 1:1 tutoring API (`/api/sessions/*`) and **no** demo frontend in this repo.

## Design notes

Why mastery is 4-level and how group speak constraints work: [`docs/architecture.md`](docs/architecture.md).  
What was copied from `student-simulation`: [`MIGRATION.md`](MIGRATION.md).

## Sync from student-simulation

When the monolith repo changes, refresh this backend:

```bash
python scripts/sync_from_student_simulation.py
cd backend && pytest
```

See [`docs/MIGRATION_PROMPT.md`](docs/MIGRATION_PROMPT.md) for the full agent checklist.
