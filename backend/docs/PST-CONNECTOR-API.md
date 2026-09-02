# PST Connector API (backend → pst-training-game)

Stable facade for the training game when `VITE_AI_SOURCE=backend` (Phase 3).  
Owns **student replies + group orchestration** for the FE starter roster (Maya + Jordan).  
Gate, coach, checker, hint, and reflection stay on **frontend prompts** (local OpenAI or mock).

Base URL: `VITE_STUDENT_SIM_URL` (default `http://localhost:8000`)

Related: [LIVE-HUD-API.md](./LIVE-HUD-API.md) (HUD payloads still available on group-session routes / embedded in turn `done`).

---

## Lifecycle

```text
POST /api/pst/sessions          → create + auto-start (Briefing Start)
POST /api/pst/sessions/{id}/turn → teacher message (stream by default)
DELETE /api/pst/sessions/{id}   → End Discussion
```

---

## Endpoints

| Method | Path | Notes |
|--------|------|--------|
| `POST` | `/api/pst/sessions` | Defaults: `profile_ids=maya,jordan`, `task_id=phone_plans_linear_01`; calls `start()` |
| `POST` | `/api/pst/sessions/{id}/turn` | `stream=true` (default) NDJSON; `stream=false` full JSON |
| `DELETE` | `/api/pst/sessions/{id}` | Teardown |
| `POST` | `/api/pst/audio/transcribe` | **Stub** → `501` |
| `POST` | `/api/pst/audio/speech` | **Stub** → `501` |
| `POST` | `/api/pst/board/analyze` | **Stub** → `501` |

Legacy `/api/group-sessions/*` remains unchanged for research/evals.

---

## Create session

```http
POST /api/pst/sessions
Content-Type: application/json

{}
```

Optional body:

```json
{
  "task_id": "phone_plans_linear_01",
  "task_text": null,
  "profile_ids": null
}
```

Response includes `session_id`, `started: true`, `profile_ids`, `display_names`, `students`, `live_hud`.

---

## Teacher turn (stream)

```http
POST /api/pst/sessions/{session_id}/turn
Content-Type: application/json

{
  "message": "What should we do first?"
}
```

Optional body fields (appended into the teacher message before LP respond):

```json
{
  "message": "What should we do first?",
  "board_note": "wrote \"Plan A = 25\"",
  "proximity": { "Maya": "near", "Jordan": "far" }
}
```

- `board_note` → `[Also wrote on the blackboard: …]`
- `proximity` → `[Standing near: …; farther from: …]` (keys may be display names or profile ids)

`done` / non-stream responses include `live_hud` (see [LIVE-HUD-API.md](./LIVE-HUD-API.md)).

### NDJSON events

| `type` | Meaning |
|--------|---------|
| `speak_plan` | Speakers / may_speak / thinking ids |
| `thinking` | Profile ids currently generating |
| `reply` | One finalized student (`kind`: primary \| game_flow \| peer) |
| `done` | Full turn payload (same shape as non-stream), including `live_hud` |
| `error` | `{ "detail": "..." }` |

Non-stream: `POST .../turn?stream=false` → single JSON object (same as `done` body without wrapping).

---

## Stubs

```json
{ "detail": "not implemented", "stub": true }
```

Status `501`. Paths are reserved. **Mic / Whisper stay on the FE** (`VITE_OPENAI_API_KEY`); student TTS was removed from the game. Board vision (`/board/analyze`) is deferred until a multimodal backend path exists.

---

## curl examples

```bash
# create
SID=$(curl -s -X POST http://localhost:8000/api/pst/sessions \
  -H "Content-Type: application/json" -d "{}" | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# streamed turn
curl -N -X POST "http://localhost:8000/api/pst/sessions/$SID/turn" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What should we do first?\"}"

# delete
curl -s -X DELETE "http://localhost:8000/api/pst/sessions/$SID"
```

Expect only Maya/Jordan speakers in stream events.
