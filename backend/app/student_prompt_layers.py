"""Layered student prompt sections: turn state (WHAT) and voice card (HOW)."""

from __future__ import annotations

from typing import Optional

from app.eval_log import learning_expected_behavior_text
from app.models import Student
from app.personality import (
    build_cognitive_state_block,
    build_personality_expression_block,
    ocean_combination_summary,
    response_length_hint,
)
from app.turn_classifier import TurnMode

VOICE_FOOTER = (
    "This voice applies every turn. Turn state above says WHAT to attempt; "
    "this block says HOW to sound."
)

_VALID_PEER_MOVES = frozenset({"asking", "claim", "evidence", "relating", "none"})


def peer_move_block(
    expected_move: str,
    *,
    prev_name: str = "",
    context: str = "",
) -> str:
    """System-prompt block steering peer discourse toward a specific talk move."""
    move = (expected_move or "").strip().lower()
    if not move or move not in _VALID_PEER_MOVES or move == "none":
        return ""
    peer = (prev_name or "your classmate").strip()
    lines = [f"[PEER MOVE — {move}]"]
    if move == "relating":
        lines.append(
            f"React to what {peer} just said. Name their point; agree or disagree "
            "with one reason. Do not restate your opening claim verbatim."
        )
    elif move == "evidence":
        lines.append(
            f"React to what {peer} just said. Add ONE new detail — a number, "
            "table row, or equation fragment. Do not repeat the same line."
        )
    elif move == "claim":
        lines.append(
            f"Respond to {peer}. Restate your position with one new hook — "
            "not a verbatim repeat of your last turn."
        )
    elif move == "asking":
        lines.append(
            "Short confusion or question only — do not solve or tutor."
        )
    if context:
        lines.append(f"Context: {context}")
    return "\n".join(lines)


def critique_move_block(
    *,
    peer_name: str = "Jordan",
    peer_text: str = "",
    context: str = "",
) -> str:
    """Teacher-direct critique prompt slice (Maya → Jordan)."""
    peer = (peer_name or "Jordan").strip()
    lines = [
        "[PEER CRITIQUE — teacher asked you to evaluate classmate]",
        f"React to {peer}'s last point. Add one new objection — the $20 starting fee, "
        "a starting-cost blind spot, or a table row that contradicts rate-only reasoning. "
        "Do NOT restate your opening table line verbatim. Stay hedged; do not explain "
        "the full crossover.",
    ]
    claim = (peer_text or "").strip()
    if claim:
        snippet = claim if len(claim) <= 200 else claim[:197] + "..."
        lines.append(f'{peer} said: "{snippet}"')
    if context:
        lines.append(f"Context: {context}")
    return "\n".join(lines)


def reasoning_press_block(
    expectation,
    *,
    student: Optional[Student] = None,
    peer_context: str = "",
) -> str:
    """Teacher asked for justification — LP depth + personality tone."""
    _ = student  # reserved for future per-student voice hints
    if expectation is None:
        return ""
    lines = [
        "[REASONING PRESS — teacher asked you to explain why]",
        "Give a warrant for your claim at your mastery level — not a full expert proof.",
        f"Depth: {expectation.tier}. Tone: {expectation.tone}.",
        f"LP guide: {expectation.lp_brief}",
        expectation.rewrite_tone_hint,
        "Do not restate only the claim; add at least one reason (number, rate, table row, or fee).",
    ]
    if expectation.tier != "rich":
        lines.append(
            "Do not tutor, set equations equal, or solve crossover unless the group already did."
        )
    if peer_context:
        lines.append(f"Context: {peer_context}")
    return "\n".join(lines)


def build_turn_state_block(
    student: Student,
    behavior_profile: Optional[dict],
    error_playbook: Optional[dict],
    turn_mode: TurnMode,
    *,
    stall_active: bool = False,
    misconception=None,
    is_peer_turn: bool = False,
) -> str:
    """Cognitive WHAT-to-attempt block plus optional expected-behavior line."""
    if turn_mode in ("social", "off_topic") or not behavior_profile:
        return ""

    cognitive = build_cognitive_state_block(
        student,
        behavior_profile,
        error_playbook,
        turn_mode,
        stall_active=stall_active,
    )
    if not cognitive:
        return ""

    lines = [cognitive]
    construct_id = (
        behavior_profile.get("primary_construct")
        or behavior_profile.get("construct_id")
        or ""
    )
    if construct_id:
        eb = learning_expected_behavior_text(
            behavior_profile,
            misconception=misconception,
            turn_mode=turn_mode,
            is_peer_turn=is_peer_turn,
            student=student,
        ).strip()
        if eb:
            lines.extend(["", f"Turn goal: {eb}"])
    return "\n".join(lines)


def build_voice_card(
    student: Student,
    behavior_profile: Optional[dict],
    turn_mode: TurnMode,
    *,
    profile_id: str = "",
    stall_active: bool = False,
) -> str:
    """Compact OCEAN voice block — placed last before classroom floor rules."""
    lines = ["[VOICE — apply to every reply]"]

    pid = profile_id or (student.student_id or "").lower()
    summary = ocean_combination_summary(student, pid or None)
    if summary:
        lines.append(summary)

    expression = build_personality_expression_block(
        student,
        behavior_profile,
        turn_mode,
        stall_active=stall_active,
    )
    if expression:
        lines.append(expression)
    elif turn_mode in ("social", "off_topic", "mixed"):
        lines.append(response_length_hint(student, turn_mode))

    lines.extend(["", VOICE_FOOTER])
    return "\n".join(lines)
