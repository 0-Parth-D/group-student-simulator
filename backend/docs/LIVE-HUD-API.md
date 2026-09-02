# Live HUD API (backend → pst-training-game)

Structured payloads for a **top-right live panel**: why each student spoke, turn metrics, roster personalities, and task knowledge levels.

Base URL: `VITE_STUDENT_SIM_URL` (default `http://localhost:8000`)

For the FE switchable student/orch path, see [PST-CONNECTOR-API.md](./PST-CONNECTOR-API.md) (`/api/pst/*`).

---

## Endpoints

| Method | Path | When to call |
|--------|------|--------------|
| `GET` | `/api/group-sessions/{session_id}/live` | Poll or refresh full HUD anytime during session |
| `GET` | `/api/group-sessions/{session_id}/roster` | Static roster + knowledge (lighter than `/live`) |
| `POST` | `/api/group-sessions` | Create session — includes `live_hud` (roster baseline) |
| `POST` | `/api/group-sessions/{session_id}/message` | Each teacher message — includes updated `live_hud` |
| `POST` | `/api/group-sessions/{session_id}/message?stream=true` | Same turn, but **NDJSON stream**: `speak_plan` → `thinking` → `reply`* → `done` (each finalized student as soon as refinement finishes) |
| `POST` | `/api/group-sessions/{session_id}/advance` | Peer nudge — includes updated `live_hud` |

Legacy flat fields (`speak_decisions`, `speak_reasons`, `students`, …) are **unchanged** on `/message` and `/advance`. Prefer `live_hud` for the UI.

---

## `live_hud` top-level shape

```json
{
  "session_id": "uuid",
  "turn": 3,
  "orchestration": "teacher_directed_self_select",
  "task": {
    "task_id": "phone_plans_linear_01",
    "problem_type": "linear_compare_word",
    "description": "…",
    "required_constructs": ["unit_rate", "linear_relationship", …],
    "target_stack_level": 3,
    "expected_answer": "100 texts",
    "facilitator_solution": "…"
  },
  "roster": [ /* see Roster entry */ ],
  "current_turn": { /* see Current turn */ },
  "turn_history": [ /* condensed orchestration_log */ ]
}
```

---

## Roster entry (`live_hud.roster[]`)

One object per student in the session (`alex`, `maya`, `jordan`).

```json
{
  "profile_id": "alex",
  "display_name": "Alex",
  "barrier": {
    "category": "epistemic",
    "label": "Rate of change",
    "summary": "Can do arithmetic but treats 10¢/text as flat +10…",
    "discourse_role": "superficial_acceptance"
  },
  "personality": {
    "Openness": "High",
    "Conscientiousness": "Low",
    "Extraversion": "Low",
    "Agreeableness": "High",
    "Neuroticism": "Low",
    "persona_blurb": "…",
    "trait_meanings": {
      "Openness": {
        "level": "High",
        "label": "Curious",
        "display_label": "Curiosity",
        "summary": "…",
        "behaviors": ["…", "…", "…"]
      }
    },
    "combination_summary": "Quiet and cooperative, fairly steady under pressure…"
  },
  "big_five": { "openness": 0.5, "conscientiousness": 0.6, … },
  "knowledge": {
    "task_id": "phone_plans_linear_01",
    "primary_construct": "unit_rate",
    "behavior_mode": "NORMAL_ERROR_PROFILE",
    "likely_correctness": "partial",
    "student_stack_level": 2,
    "target_stack_level": 3,
    "predicted_behavior": "…",
    "receptivity": 0.55,
    "mastery_state": { "unit_rate": 2, … },
    "task_construct_levels": { "unit_rate": 2, "linear_relationship": 1 },
    "construct_summary": [ … ],
    "pisa_profile": [ … ],
    "error_types": [ … ],
    "stall_active": false
  },
  "learning_profile": { … }
}
```

**UI suggestion:** Left column = `barrier` + `personality`; knowledge card = `knowledge.task_construct_levels`, stack levels, `receptivity`.

---

## Current turn (`live_hud.current_turn`)

Updated after each `/message` or `/advance`.

```json
{
  "turn": 3,
  "teacher_message": "Alex, can you explain…",
  "constraint_mode": "direct",
  "turn_mode": "math_scaffold",
  "pedagogical_move": "",
  "teaching_warning": false,
  "vague_warning": false,
  "observers": ["jordan"],
  "must_speak": ["alex"],
  "may_speak": [],
  "stop_reason": "orchestrator_stop",
  "speak_decisions": [
    {
      "profile_id": "alex",
      "will_speak": true,
      "score": 1.0,
      "reason": "must_speak (directly addressed)"
    }
  ],
  "speak_reasons": { "alex": "must_speak (directly addressed)" },
  "speak_events": [
    {
      "order": 1,
      "profile_id": "alex",
      "display_name": "Alex",
      "source": "primary",
      "will_speak": true,
      "volunteer_score": 1.0,
      "speak_reason": "must_speak (directly addressed)",
      "orchestrator": {
        "expected_move": null,
        "responding_to": null,
        "context": null
      },
      "help_seeking": false,
      "is_question": false,
      "reply_preview": "Like… 10 cents is 10, right…"
    },
    {
      "order": 2,
      "profile_id": "maya",
      "display_name": "Maya",
      "source": "game_flow",
      "speak_reason": "game orchestrator → relating: bridges Alex confusion",
      "orchestrator": {
        "expected_move": "relating",
        "responding_to": "Alex",
        "context": "bridges Alex confusion"
      },
      "reply_preview": "Oh wait, 10 cents is $0.10…"
    }
  ],
  "learning_events": [ … ],
  "receptivity": { "alex": 0.55, "maya": 0.62, "jordan": 0.48 }
}
```

**UI suggestion:** Render `speak_events[]` as a vertical timeline (order, name, reason, preview). Badge `source`: `primary` | `game_flow` | `peer_continuation`.

---

## Reply-level fields (on `/message` responses)

Each item in `replies[]` and `multi_round_replies[]` now includes:

```json
{
  "speaker_id": "maya",
  "content": "…",
  "help_seeking": false,
  "is_question": false,
  "source": "primary",
  "speak_reason": "must_speak (directly addressed)",
  "orchestrator": {
    "expected_move": "claim",
    "responding_to": "teacher",
    "context": "…"
  }
}
```

`orchestrator` is present on `game_flow` replies only.

---

## Turn history (`live_hud.turn_history[]`)

Last ~12 orchestration log entries (teacher waves + game-flow rounds).

```json
{
  "turn": 3,
  "kind": "game_flow",
  "mode": "direct",
  "speakers": ["maya"],
  "observers": [],
  "round": 2,
  "stop_reason": "orchestrator_stop",
  "orchestrator": { "next_speaker": "Jordan", … },
  "teacher_message_preview": "Maya, what do you think…"
}
```

---

## Recommended frontend flow

1. **Briefing / session start:** `POST /api/group-sessions` → render `live_hud.roster`.
2. **Each teacher send:** `POST …/message` → replace panel from `live_hud` (or merge `current_turn` + refresh roster knowledge if mastery updates matter).
3. **Optional poll:** `GET …/live` every N seconds if you need refresh without sending a message.
4. **Nudge peers:** `POST …/advance` → append `speak_events` from `live_hud.current_turn`.

---

## Module reference

- Builder: `backend/app/live_hud.py`
- PST barriers: `PST_GROUP_BARRIERS` (alex/maya/jordan aligned with game scenario)
- Tests: `backend/tests/test_live_hud.py`
