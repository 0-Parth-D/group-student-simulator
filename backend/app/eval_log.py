"""Eval turn record builder (used by sessions and eval battery)."""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from app.config import GENERATION_DEBUG_HIST_MAX, REFINE_MODE
from app.expected_behavior import expected_behavior, role_for_turn


_TURN_STATE_KEYS = (
    "behavior_mode",
    "student_stack_level",
    "target_stack_level",
    "likely_correctness",
    "stall_active",
    "help_seeking",
    "primary_construct",
    "help_seek_style",
)


def learning_expected_behavior_text(
    behavior_profile: dict,
    *,
    misconception: Union[dict, str, None] = None,
    turn_mode: str = "math_scaffold",
    is_peer_turn: bool = False,
    student=None,
    turn_role: str = "",
) -> str:
    """Top-level expected_behavior string (no LLM; used when GENERATION_DEBUG is off)."""
    bp = behavior_profile or {}
    construct_id = (
        bp.get("primary_construct") or bp.get("construct_id") or ""
    )
    mastery = int(
        bp.get("student_stack_level")
        or (
            (student.construct_mastery or {}).get(construct_id, 0)
            if student is not None
            else 0
        )
        or 0
    )
    return expected_behavior(
        construct_id,
        mastery,
        misconception=misconception,
        turn_role=role_for_turn(
            turn_mode=turn_mode or "",
            is_peer_turn=is_peer_turn,
            is_first_turn=False,
            turn_role=turn_role or "",
        ),
        behavior_mode=bp.get("behavior_mode") or "",
        stall_active=bool(bp.get("stall_active")),
    )


def _misconception_snapshot(
    misconception: Union[dict, str, None],
) -> Optional[dict]:
    if not misconception:
        return None
    if isinstance(misconception, str):
        return {"id": misconception, "prompt_cue": "", "description": ""}
    return {
        "id": str(misconception.get("id") or ""),
        "prompt_cue": str(misconception.get("prompt_cue") or ""),
        "description": str(misconception.get("description") or ""),
    }


def build_generation_debug(
    *,
    system_prompt: str,
    history: list,
    turn_mode: str,
    is_peer_turn: bool,
    behavior_profile: dict,
    targets: dict,
    personality_export: Optional[dict] = None,
    misconception: Union[dict, str, None] = None,
    draft: str = "",
    final_reply: str = "",
    revisions: int = 0,
    critic: Optional[dict] = None,
    hist_max: Optional[int] = None,
    llm_params: Optional[dict] = None,
    retrieval: Optional[dict] = None,
) -> dict:
    """Assemble per-turn generation debug payload for eval_log exports."""
    cap = GENERATION_DEBUG_HIST_MAX if hist_max is None else max(0, int(hist_max))
    hist = list(history or [])
    if cap > 0 and len(hist) > cap:
        hist = hist[-cap:]
    bp = behavior_profile or {}
    turn_state = {k: bp.get(k) for k in _TURN_STATE_KEYS}
    out = {
        "system_prompt": system_prompt or "",
        "prompt_chars": len(system_prompt or ""),
        "history": deepcopy(hist),
        "turn_mode": turn_mode or "",
        "is_peer_turn": bool(is_peer_turn),
        "turn_state": turn_state,
        "targets": dict(targets or {}),
        "personality_export": dict(personality_export or {}),
        "misconception": _misconception_snapshot(misconception),
        "draft": draft or "",
        "final_reply": final_reply or "",
        "revisions": int(revisions),
        "critic": critic,
        "refine_mode": REFINE_MODE,
    }
    if llm_params is not None:
        out["llm_params"] = dict(llm_params)
    if retrieval is not None:
        out["retrieval"] = dict(retrieval)
    return out


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
    generation_debug: Optional[dict] = None,
) -> dict:
    """Build a rich turn record for offline evaluation."""
    bp = behavior_profile or {}
    active_misc = (misconceptions or [{}])[0] if misconceptions else {}
    record: Dict[str, Any] = {
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
    if generation_debug is not None:
        record["generation_debug"] = generation_debug
    return record
