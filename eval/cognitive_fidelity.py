"""Cognitive fidelity scoring for simulated student turns."""

import json

from app.llm import complete
from app.llm.roles import LLMRole

COGNITIVE_FIDELITY_PROMPT = """Student profile: {student_id}
Behavior mode: {behavior_mode}
Mastery level: {mastery}/3 on construct {construct}
Active misconception: {misconception_id} ({error_pattern})
Expected behavior for this turn: {expected_behavior}
Student response: "{reply}"
Did the response reflect the expected knowledge state and behavior mode?
Score 1-5 (5 = fully aligned). Return JSON: {{"score": int, "violated_constraint": str|null}}
"""


def cognitive_fidelity(turn: dict) -> dict:
    """Score one math turn for construct-level fidelity."""
    prompt = COGNITIVE_FIDELITY_PROMPT.format(
        student_id=turn.get("profile_id", turn.get("student_id", "student")),
        behavior_mode=turn.get("behavior_mode", "unknown"),
        mastery=turn.get("mastery", turn.get("student_stack_level", 1)),
        construct=turn.get("construct_id", turn.get("primary_construct", "fraction_equivalence")),
        misconception_id=turn.get("misconception_id", "unknown"),
        error_pattern=turn.get("error_pattern", ""),
        expected_behavior=turn.get("expected_behavior", "stay in character at mastery level"),
        reply=(turn.get("reply") or "").replace('"', "'")[:500],
    )
    try:
        raw = complete(
            LLMRole.EVAL_COGNITIVE,
            "You evaluate student simulation fidelity. Return JSON only.",
            prompt,
        )
        return json.loads(raw)
    except Exception as exc:
        return {"score": 0, "violated_constraint": str(exc)}
