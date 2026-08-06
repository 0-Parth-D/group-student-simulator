"""Behavioral consistency scoring vs teacher move type."""

import json

from app.llm import complete
from app.llm.roles import LLMRole

BEHAVIORAL_PROMPT = """Teacher move type: {teacher_label}
Expected student behavior: {expected_behavior}
Actual response: "{reply}"
Did the student behave correctly? JSON: {{"consistent": bool, "deviation": str|null}}
"""


def behavioral_consistency(turn: dict) -> dict:
    """Score whether student behavior matched teacher turn expectations."""
    prompt = BEHAVIORAL_PROMPT.format(
        teacher_label=turn.get("teacher_label", "math_scaffold"),
        expected_behavior=turn.get("expected_behavior", "stay in character"),
        reply=(turn.get("reply") or "").replace('"', "'")[:500],
    )
    try:
        raw = complete(
            LLMRole.EVAL_BEHAVIORAL,
            "You evaluate tutoring dialogue consistency. Return JSON only.",
            prompt,
        )
        return json.loads(raw)
    except Exception as exc:
        return {"consistent": False, "deviation": str(exc)}
