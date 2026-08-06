from typing import List, Optional, Tuple

from app.knowledge.mistakes import match_lp_constructs_by_keywords
from app.student.prediction import merge_concepts
from app.llm.prompts import student_prompt
from app.knowledge.turn_classifier import (
    TurnMode,
    classify_teacher_turn,
    is_vague_acknowledgment,
)

STALL_LABEL = "vague_acknowledgment"

VAGUE_STALL_LABELS = frozenset({"vague_acknowledgment", "vague_directive"})

# Praise-only acks after a scaffold — brief reply, not a stall turn.
_PRAISE_ACKS = frozenset(
    {"correct", "right", "good", "good job", "yes", "nice", "exactly", "perfect"}
)

STALL_INJECT = (
    "\n[SYSTEM: The teacher gave no new direction. "
    "Do NOT compute another step or state a final answer. "
    "Ask what they want — e.g. 'What should I do next?' or 'Okay — what part?']"
)

MATH_PROMPT_TURNS = frozenset(
    {
        "math_scaffold",
        "math_eval",
        "mixed",
        "vague_acknowledgment",
        "vague_directive",
    }
)


def is_vague(teacher_utt: str, turn_mode: TurnMode) -> bool:
    return is_vague_acknowledgment(turn_mode)


def should_student_stall(
    turn_history: List[dict],
    current_label: str = STALL_LABEL,
    teacher_utt: str = "",
) -> bool:
    """True when teacher gave no substantive direction after the student has responded."""
    if current_label not in VAGUE_STALL_LABELS:
        return False
    if not turn_history:
        return False

    stripped = (teacher_utt or "").strip().lower().rstrip("!.?")
    if current_label == STALL_LABEL and stripped in _PRAISE_ACKS:
        return False

    recent = (turn_history + [{"teacher_label": current_label}])[-3:]
    vague_ack_count = sum(
        1 for entry in recent if entry.get("teacher_label") == STALL_LABEL
    )
    if vague_ack_count >= 2:
        return True

    # Battery turns 2–3: vague "okay" / "go on" after the opener — stall, don't advance.
    return True


def build_lp_merged_concepts(base_top_p: list, teacher_utt: str) -> list:
    keyword_hits = match_lp_constructs_by_keywords(teacher_utt, base_top_p)
    return merge_concepts(base_top_p, keyword_hits)


def build_student_system(
    student,
    predicted_behavior: str,
    merged_concepts: list,
    task_text: str = "",
    error_playbook: Optional[dict] = None,
    error_anchor: Optional[str] = None,
    behavior_profile: Optional[dict] = None,
    turn_mode: TurnMode = "math_scaffold",
    scaffold_boost: Optional[dict] = None,
    stall_active: bool = False,
    task_metadata: Optional[dict] = None,
) -> str:
    return student_prompt(
        student,
        predicted_behavior,
        merged_concepts,
        task_text=task_text,
        error_playbook=error_playbook,
        error_anchor=error_anchor,
        behavior_profile=behavior_profile,
        turn_mode=turn_mode,
        scaffold_boost=scaffold_boost,
        stall_active=stall_active,
        task_metadata=task_metadata,
    )


def prepare_student_user_message(
    teacher_msg: str,
    turn_mode: TurnMode,
    stall_active: bool = False,
) -> tuple[str, bool]:
    """Return (user_message_for_student, vague_warning)."""
    vague_warning = stall_active or is_vague(teacher_msg, turn_mode)
    if stall_active or turn_mode in VAGUE_STALL_LABELS:
        return teacher_msg + STALL_INJECT, vague_warning
    return teacher_msg, False


def low_agreeableness_reminder_pairs(turn: int) -> list[dict]:
    """Return user/assistant reminder pairs for even turns (Fix 7, passive wording)."""
    if turn % 2 != 0:
        return []
    reminder = (
        "[INTERNAL REMINDER: On math steps, do NOT thank or praise the teacher. "
        "Stay passive on problem work — minimal answers, no hostility.]"
    )
    return [
        {"role": "user", "content": reminder},
        {"role": "assistant", "content": "[noted]"},
    ]


def build_turn_context(
    session,
    teacher_utt: str,
    turn_mode: Optional[TurnMode] = None,
    stall_active: bool = False,
    error_anchor: Optional[str] = None,
) -> Tuple[list, str, list, TurnMode]:
    """Return (merged_concepts, student_system_prompt, active_concepts_api_list, turn_mode)."""
    if turn_mode is None:
        context = [
            {"role": m.get("role", "unknown"), "content": m.get("content", "")}
            for m in getattr(session, "messages", [])[-6:]
        ]
        turn_mode = classify_teacher_turn(
            teacher_utt, session.task_text, context=context
        )

    if turn_mode in MATH_PROMPT_TURNS:
        merged = build_lp_merged_concepts(session.top_p_concepts, teacher_utt)
    else:
        merged = list(session.top_p_concepts)

    s_sys = build_student_system(
        session.student,
        session.predicted_behavior,
        merged,
        task_text=session.task_text,
        error_playbook=(
            None
            if stall_active
            else session.error_playbook
            if turn_mode not in ("social", "off_topic")
            else None
        ),
        error_anchor=error_anchor,
        behavior_profile=session.behavior_profile if turn_mode not in ("social", "off_topic") else None,
        turn_mode=turn_mode,
        scaffold_boost=getattr(session, "scaffold_boost", None),
        stall_active=stall_active,
        task_metadata=session.task_metadata if turn_mode not in ("social", "off_topic") else None,
    )
    active = (
        lp_concepts_to_api_list(session, merged)
        if turn_mode not in ("social", "off_topic")
        else []
    )
    return merged, s_sys, active, turn_mode


def lp_concepts_to_api_list(session, merged: Optional[List[str]] = None) -> list:
    from app.knowledge.kg_config import MASTERY_LABELS, get_construct_def

    concepts = merged or session.top_p_concepts
    items = []
    for cid in concepts:
        cdef = get_construct_def(cid) or {}
        level = session.mastery_state.get(cid, 0)
        items.append(
            {
                "concept": cdef.get("label", cid),
                "state": MASTERY_LABELS.get(level, "Unknown"),
                "construct_id": cid,
            }
        )
    return items


def format_group_transcript_snippet(
    transcript: List[dict],
    n: int = 10,
    display_names: Optional[dict] = None,
) -> str:
    """Format the last N shared transcript lines for a student user message."""
    names = display_names or {}
    lines = ["Transcript so far:"]
    for entry in (transcript or [])[-n:]:
        speaker_type = entry.get("speaker_type", "")
        speaker_id = entry.get("speaker_id", "")
        content = entry.get("content", "")
        if speaker_type == "teacher":
            label = "Teacher"
        else:
            label = names.get(speaker_id) or (speaker_id.capitalize() if speaker_id else "Student")
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def prepare_peer_user_message(speaker_name: str, content: str) -> str:
    """Format a peer utterance as the incoming user message for another student."""
    return f"[{speaker_name}]: {content}"


def group_awareness_block(self_name: str, other_first_names: List[str]) -> str:
    """Short system-prompt addendum so the student stays in-character in a group."""
    others = [n for n in other_first_names if n]
    if len(others) >= 2:
        others_text = f"{', '.join(others[:-1])} and {others[-1]}"
    elif others:
        others_text = others[0]
    else:
        others_text = "your classmates"
    return (
        f"You are in a small group with {others_text} working on the same problem. "
        "You may agree, disagree, or build on what others said — but keep your own "
        "knowledge level and behavior mode. If you are unsure, hedge. Do not lecture "
        "classmates or sound like a teacher or tutor. Keep answers short in group "
        "discussion. Do not ask the teacher what to do next. If the teacher tells you "
        "to watch or listen, stay silent."
    )
