"""Shared context policy for group teacher-wave and LangGraph peer turns."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

from app.config import GROUP_PRIVATE_HIST_MAX, GROUP_TRANSCRIPT_WINDOW
from app.turn_logic import format_group_transcript_snippet, group_awareness_block

if TYPE_CHECKING:
    from app.group_sessions import GroupSession, StudentSlot


MULTI_ROUND_USER_LINE = (
    "Continue the peer discussion only if you have something new to add. "
    "Do not ask the teacher what to do next. Do not wait, whisper, or narrate silence."
)
MULTI_ROUND_AWARENESS_LINE = (
    "Stay in character as a middle-school student. Add a short, concrete math thought "
    "or reaction to a classmate — do not tutor, do not ask the teacher what to do next, "
    "and do not describe waiting or whispering."
)
GAME_FLOW_AWARENESS_LINE = (
    "You are in a live peer discussion. Sound like a real 6–8th grader: everyday words, "
    "short or run-on sentences, hedging (I think, kinda, wait), thinking out loud. "
    "A light filler at the start (okay so, wait, um) is fine when natural — never inside math. "
    "Never sound like a textbook, tutor, or adult explaining math. Keep it brief."
)


def wrap_user_message(
    group: "GroupSession",
    profile_id: str,
    core_msg: str,
    *,
    multi_round: bool = False,
    transcript_window: int | None = None,
    include_transcript: bool = True,
) -> str:
    """Build the user-message wrap shared by teacher-wave and peer continuation."""
    self_name = group.display_names.get(profile_id, profile_id.capitalize())
    task_text = (group.task_text or "").strip()
    parts: list[str] = []
    if include_transcript:
        window = (
            GROUP_TRANSCRIPT_WINDOW
            if transcript_window is None
            else int(transcript_window)
        )
        snippet = format_group_transcript_snippet(
            group.transcript,
            n=window,
            display_names=group.display_names,
        )
        parts.extend([snippet, ""])
    if task_text:
        parts.append(f"Task: {task_text}")
        parts.append("")
    if multi_round:
        parts.append(MULTI_ROUND_USER_LINE)
        parts.append("")
    parts.append(f"Respond as {self_name}.")
    parts.append("")
    parts.append(core_msg)
    return "\n".join(parts)


def awareness_block(
    self_name: str,
    other_first_names: Sequence[str],
    *,
    multi_round: bool = False,
    game_flow: bool = False,
) -> str:
    """Group awareness system addendum; optional multi-round / game-flow peer cue."""
    block = group_awareness_block(self_name, list(other_first_names))
    if game_flow:
        return f"{block}\n{GAME_FLOW_AWARENESS_LINE}"
    if multi_round:
        return f"{block}\n{MULTI_ROUND_AWARENESS_LINE}"
    return block


def trim_private_hist(
    slot: "StudentSlot",
    *,
    max_messages: int | None = None,
) -> None:
    """Cap a student slot's private chat history to the last N messages."""
    cap = GROUP_PRIVATE_HIST_MAX if max_messages is None else int(max_messages)
    if cap < 1:
        return
    hist = slot.student_hist
    if len(hist) > cap:
        slot.student_hist = hist[-cap:]


def eligible_speakers(
    profile_ids: Sequence[str],
    may_speak: Sequence[str] | None,
    observers: Sequence[str] | None,
) -> List[str]:
    """Eligible = last may_speak (or all − observers); never include observers."""
    observer_set = set(observers or [])
    if may_speak:
        pool = [pid for pid in may_speak if pid in profile_ids]
    else:
        pool = list(profile_ids)
    return [pid for pid in pool if pid not in observer_set]
