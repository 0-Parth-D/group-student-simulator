"""Eval turn record builder (used by sessions and eval battery)."""

from copy import deepcopy
from typing import Dict, List, Optional


def build_eval_turn_record(
    *,
    turn: int,
    teacher_message: str,
    reply: str,
    teacher_label: str,
    stall_active: bool,
    profile_id: str,
    task_id: str,
    behavior_profile: dict,
    task_metadata: dict,
    misconceptions: Optional[List[dict]] = None,
    mastery_state: Optional[Dict[str, int]] = None,
    expected_behavior: str = "",
    vague_warning: bool = False,
    help_seeking: bool = False,
    is_question: bool = False,
) -> dict:
    """Build a rich turn record for offline evaluation."""
    bp = behavior_profile or {}
    active_misc = (misconceptions or [{}])[0] if misconceptions else {}
    return {
        "turn": turn,
        "role": "student",
        "type": "math",
        "teacher_message": teacher_message,
        "reply": reply,
        "teacher_label": teacher_label,
        "stall_active": stall_active,
        "vague_warning": vague_warning,
        "help_seeking": help_seeking,
        "is_question": is_question,
        "profile_id": profile_id,
        "task_id": task_id or task_metadata.get("task_id", ""),
        "behavior_mode": bp.get("behavior_mode", ""),
        "likely_correctness": bp.get("likely_correctness", ""),
        "primary_construct": bp.get("primary_construct", ""),
        "construct_id": bp.get("primary_construct", ""),
        "student_stack_level": bp.get("student_stack_level", 0),
        "mastery": bp.get("student_stack_level", 0),
        "error_types": list(bp.get("error_types") or []),
        "error_pattern": "; ".join((bp.get("error_types") or [])[:2]),
        "misconception_id": active_misc.get("id", ""),
        "expected_answer": task_metadata.get("expected_answer", ""),
        "expected_behavior": expected_behavior,
        "typical_errors": deepcopy(task_metadata.get("typical_errors") or []),
        "mastery_state": dict(mastery_state or {}),
    }
