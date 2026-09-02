"""Track C — group discourse authenticity (peer talk quality)."""

from __future__ import annotations

import json
from typing import List, Optional

from app.llm import complete
from app.llm_config import LLMRole

GROUP_DISCOURSE_PROMPT = """You rate whether a middle-school student's peer reply sounds like a real child in a small-group math discussion (not a tutor).

Student profile_id: {profile_id}
Behavior mode: {behavior_mode}
Persona hint: {persona_hint}

Recent transcript:
{transcript_window}

This student's reply: "{reply}"

Score 1-5:
  5 = child-like: hedges, agrees/disagrees, builds briefly, stays at their knowledge level
  3 = mixed / generic student
  1 = tutor-like: lectures classmates, explains like a teacher, over-expert

Also set:
  tutor_like: true if the reply mainly teaches/lectures peers
  peer_move: one of agree | disagree | build | question | other

Return JSON only:
{{"score": int, "tutor_like": bool, "peer_move": str, "rationale": str}}
"""


def _persona_hint(profile_id: str) -> str:
    hints = {
        "alex": "quiet, brief, procedural/conceptual mix; waits to be invited",
        "maya": "anxious, hedges, stronger on procedures; should not lecture",
        "jordan": "procedurally strong, equation-first; explains carefully without tutoring peers",
        "riley": "confident, talkative, often wrong; guesses rather than careful reasoning",
    }
    return hints.get((profile_id or "").lower(), "middle-school math student")


def group_discourse_fidelity(
    *,
    reply: str,
    profile_id: str,
    behavior_mode: str = "",
    transcript_window: Optional[List[str]] = None,
) -> dict:
    """LLM judge for peer-discourse authenticity (layer G)."""
    window = "\n".join(transcript_window or []) or "(none)"
    prompt = GROUP_DISCOURSE_PROMPT.format(
        profile_id=profile_id or "student",
        behavior_mode=behavior_mode or "unknown",
        persona_hint=_persona_hint(profile_id),
        transcript_window=window[:2500],
        reply=(reply or "").replace('"', "'")[:500],
    )
    try:
        raw = complete(
            LLMRole.EVAL_GROUP_DISCOURSE,
            "You evaluate middle-school peer math talk. Return JSON only.",
            prompt,
        )
        data = json.loads(raw)
        score = int(data.get("score", 0))
        return {
            "score": score,
            "tutor_like": bool(data.get("tutor_like", False)),
            "peer_move": data.get("peer_move") or "other",
            "rationale": data.get("rationale") or "",
            "profile_id": profile_id,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "score": 0,
            "tutor_like": False,
            "peer_move": "other",
            "rationale": str(exc),
            "profile_id": profile_id,
            "error": str(exc),
        }
