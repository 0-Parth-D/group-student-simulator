"""Game-style conversation flow orchestrator (mirrors pst-training-game orchestrator agent).

After the constraint-resolved primary speaker(s) reply, this module decides whether
another student should speak — using the same rules as the frontend — and generates
peer replies until a natural pause or safety cap.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from app.llm import complete_chat
from app.llm_config import LLMRole
from app.turn_logic import format_group_transcript_snippet, prepare_peer_user_message

if TYPE_CHECKING:
    from app.group_sessions import GroupSession, GroupSessionStore

logger = logging.getLogger(__name__)

# 3-student research roster (Alex + Maya + Jordan)
ORCHESTRATOR_SYSTEM_3 = """You are a conversation flow controller for a 3-student group discussion simulation.

## Students (respect Extraversion — do not equalize floor time)
- Alex: Low Extraversion — quiet, struggling, epistemic barrier (rate of change). Often INCORRECT on rates. Speaks mainly when directly asked or briefly when confused. Does NOT dominate debates.
- Maya: Low–moderate Extraversion — observant, communicative barrier (missing warrants). Partial table reasoning; bridges sometimes. Shorter turns; not the loudest voice.
- Jordan: High Extraversion — procedurally fluent, equations-first, shallow on meaning. Jumps in first on open questions, corrects mistakes, eager to move forward. Prefer Jordan for unprompted peer pushback unless Jordan restraint applies.

## Competing reasonings
Alex, Maya, and Jordan should sound like they are reasoning DIFFERENTLY — do not collapse them into one shared correct explanation. When a student just stated a claim, prefer continuing with a peer who would challenge, extend, or ask "why" — unless the teacher needs to intervene.

## Task
Given the conversation so far, look at the LAST message and decide: should another student speak next, or should the discussion pause for the teacher to intervene?

## Decision Rules (priority order)
1. ADDRESSED STUDENT GOES FIRST: If the teacher explicitly addressed ONE student by name to DO something ("Maya, can you explain...", "Alex, what do you think?", "Jordan, slow down"), that student MUST be next. Others do NOT jump in until that student has spoken. Highest priority.
2. UNANSWERED QUESTIONS: If a recent student message ended with a question aimed at another student or the group and no one has answered it → continue; the most appropriate student answers. Don't leave questions hanging.
3. If the last student asked the TEACHER a direct question ("Teacher, is that right?") → stop (teacher must respond).
4. If 4+ student messages have appeared since the teacher's last message → stop (teacher needs a chance).
5. If the last speaker said something another student would naturally react to → continue:
   - Jordan gave a fast answer with no explanation → Alex might be briefly confused, Maya might agree — OR stop for teacher.
   - Alex said something incorrect or confusing → prefer Jordan to correct (High E), unless Jordan restraint.
   - Maya gave an incomplete explanation → Jordan might add; Alex only if directly confused/asked.
   - Someone said something incorrect → Jordan corrects (default), not a long Alex monologue.
   - Someone asked a classmate a question → that classmate answers.
6. If a natural pause is reached (agreement, nothing to react to) → stop.
7. Don't let the same student speak twice in a row unless answering a direct question to them.
8. JORDAN RESTRAINT: If the teacher redirected to Maya or Alex within the last 2–3 messages, Jordan should NOT jump in until that student has spoken once. After that, Jordan may re-enter.
9. EXTRAVERSION: Do not give Low-E Alex the most turns in open debate. Prefer High-E Jordan for volunteering/corrections; keep Alex/Maya turns short and rarer unless addressed.

## Expected Talk Move
When you continue, also predict what KIND of contribution the next speaker would naturally make:
- "asking" — asks for help or clarification, admits confusion
- "claim" — states an answer or position, often without full reasoning
- "evidence" — gives reasoning, justification, or a worked example
- "relating" — reacts to, builds on, or agrees with a classmate
- "none" — brief social/procedural talk
Match the persona and the moment.

## Output (JSON only)
{
  "action": "continue" | "stop",
  "next_speaker": "Alex" | "Maya" | "Jordan",
  "responding_to": "teacher" | "Alex" | "Maya" | "Jordan",
  "expected_move": "asking" | "claim" | "evidence" | "relating" | "none",
  "context": "brief reason this student would speak now"
}"""

# FE starter scenario (Maya vs Jordan conflicting claims) — mirrors pst-training-game orchestrator.js
ORCHESTRATOR_SYSTEM_2 = """You are a conversation flow controller for a 2-student group discussion simulation.

## Students
- Maya: Intermediate, observant, communicative barrier (missing warrants). Claims Plan B is cheaper based on her table (up to 50 texts). Responds when addressed, when her claim is challenged, or when she notices something on the board that matches her table.
- Jordan: Procedurally fluent, technological/dominant barrier (equations only). Claims Plan A is better ("10 cents is way less than 30"). Jumps in first on open questions, defends his claim fast, eager to move forward. He can set equations up but does NOT spontaneously solve the crossover; he ignores the $20 fee at low usage until pressed.

## Task
Given the conversation so far, look at the LAST message and decide: should another student speak next, or should the discussion pause for the teacher to intervene?

## Decision Rules (priority order)
1. ADDRESSED STUDENT GOES FIRST: If the teacher explicitly addressed ONE student by name to DO something ("Maya, can you explain...", "Jordan, slow down"), that student MUST be next. The other does NOT jump in until that student has spoken. Highest priority.
2. UNANSWERED QUESTIONS: If a recent student message ended with a question aimed at the other student or the group and no one has answered it → continue; the other student answers. Don't leave questions hanging.
3. If the last student asked the TEACHER a direct question ("Teacher, is that right?") → stop (teacher must respond).
4. If 3+ student messages have appeared since the teacher's last message → stop (teacher needs a chance).
5. If the last speaker said something the other student would naturally react to → continue:
   - Jordan dismissed Maya's table → Maya defends it or pushes back.
   - Maya cited a number from her table that contradicts Jordan's claim → Jordan reacts (defends his equation view or gets curious).
   - Someone said something incorrect → the other may push back — but WITHOUT full reasoning unless the teacher has pressed for it.
   - Someone asked the classmate a question → that classmate answers.
6. If a natural pause is reached (agreement, standoff, nothing to react to) → stop. A standoff where both restate their claims and neither gives reasons IS a natural stop — that's the teacher's moment.
7. Don't let the same student speak twice in a row unless answering a direct question to them.
8. JORDAN RESTRAINT: If the teacher redirected to Maya within the last 2–3 messages, Jordan should NOT jump in. Give her space.

## Expected Talk Move
When you continue, also predict what KIND of contribution the next speaker would naturally make:
- "asking" — asks for help or clarification, admits confusion
- "claim" — states an answer or position, often without full reasoning
- "evidence" — gives reasoning, justification, or a worked example
- "relating" — reacts to, builds on, or agrees with the classmate
- "none" — brief social/procedural talk
Match the persona and the moment: Maya leans claim/relating; Jordan leans claim/evidence. Early turns both restate claims — reasons are EARNED by good teacher questions.

## Output (JSON only)
{
  "action": "continue" | "stop",
  "next_speaker": "Maya" | "Jordan",
  "responding_to": "teacher" | "Maya" | "Jordan",
  "expected_move": "asking" | "claim" | "evidence" | "relating" | "none",
  "context": "brief reason this student would speak now"
}"""

# Backward-compatible alias (3-student default)
ORCHESTRATOR_SYSTEM = ORCHESTRATOR_SYSTEM_3

VALID_MOVES = frozenset({"asking", "claim", "evidence", "relating", "none"})
VALID_SPEAKERS = ("Alex", "Maya", "Jordan")  # default when no roster passed
VALID_ACTIONS = frozenset({"continue", "stop"})


@dataclass(frozen=True)
class OrchestratorDecision:
    action: str
    next_speaker: Optional[str]
    responding_to: Optional[str]
    expected_move: Optional[str]
    context: str


def _extract_json(raw: str) -> Optional[dict]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def count_students_since_teacher(transcript: Sequence[dict]) -> int:
    count = 0
    for entry in reversed(transcript or []):
        if entry.get("speaker_type") == "teacher":
            break
        if entry.get("speaker_type") == "student":
            count += 1
    return count


def roster_display_names(group: "GroupSession") -> tuple[str, ...]:
    """Display names in profile_ids order for the active session."""
    names: List[str] = []
    for pid in group.profile_ids or []:
        names.append(group.display_names.get(pid, pid.capitalize()))
    return tuple(names)


def student_msg_cap_for_roster(group: "GroupSession") -> int:
    """FE 2-student stops at 3+; 3-student research path keeps 4+."""
    n = len(group.profile_ids or [])
    return 3 if n <= 2 else 4


def build_orchestrator_system(group: "GroupSession") -> str:
    names = set(roster_display_names(group))
    if names == {"Maya", "Jordan"} or (
        len(group.profile_ids or []) == 2
        and "alex" not in {p.lower() for p in group.profile_ids}
    ):
        return ORCHESTRATOR_SYSTEM_2
    return ORCHESTRATOR_SYSTEM_3


def _last_student_display(group: "GroupSession") -> tuple[Optional[str], Optional[str]]:
    for entry in reversed(group.transcript or []):
        if entry.get("speaker_type") == "student":
            pid = entry.get("speaker_id")
            name = group.display_names.get(pid, (pid or "").capitalize())
            return name, pid
    return None, None


def _display_to_profile(group: "GroupSession", display_name: str) -> Optional[str]:
    target = (display_name or "").strip().lower()
    if not target:
        return None
    for pid in group.profile_ids:
        name = (group.display_names.get(pid) or pid).strip().lower()
        if name == target or pid.lower() == target:
            return pid
    return None


def _profile_to_display(group: "GroupSession", profile_id: str) -> str:
    return group.display_names.get(profile_id, profile_id.capitalize())


def parse_orchestrator_response(
    raw: str,
    valid_speakers: Sequence[str] | None = None,
) -> OrchestratorDecision:
    fallback = OrchestratorDecision("stop", None, None, None, "")
    data = _extract_json(raw)
    if not data:
        return fallback
    action = str(data.get("action") or "stop").lower().strip()
    if action not in VALID_ACTIONS:
        return fallback
    if action == "stop":
        return OrchestratorDecision("stop", None, None, None, str(data.get("context") or ""))
    speakers = tuple(valid_speakers) if valid_speakers else VALID_SPEAKERS
    speaker = data.get("next_speaker")
    if speaker not in speakers:
        return fallback
    responding = data.get("responding_to") or "group"
    if responding not in ("teacher", *speakers):
        responding = "group"
    move = data.get("expected_move")
    if move not in VALID_MOVES:
        move = None
    return OrchestratorDecision(
        action="continue",
        next_speaker=speaker,
        responding_to=responding,
        expected_move=move,
        context=str(data.get("context") or ""),
    )


def build_orchestrator_user_message(group: "GroupSession") -> str:
    transcript = format_group_transcript_snippet(
        group.transcript,
        n=20,
        display_names=group.display_names,
    )
    cap = student_msg_cap_for_roster(group)
    return (
        f"{transcript}\n\n"
        "Decide the next speaker (JSON only). "
        f"Remember: stop when the group has reached a natural pause or {cap}+ student "
        "messages since the teacher's last message."
    )


def decide_orchestrator_turn(group: "GroupSession") -> OrchestratorDecision:
    """Call the game-style orchestrator LLM on the current group transcript."""
    cap = student_msg_cap_for_roster(group)
    if count_students_since_teacher(group.transcript) >= cap:
        return OrchestratorDecision("stop", None, None, None, "student_cap_since_teacher")

    speakers = roster_display_names(group)
    user = build_orchestrator_user_message(group)
    try:
        raw = complete_chat(
            LLMRole.GAME_FLOW_ORCHESTRATOR,
            build_orchestrator_system(group),
            [{"role": "user", "content": user}],
        )
    except Exception as exc:
        logger.warning("game_flow orchestrator LLM failed: %s", exc)
        return OrchestratorDecision("stop", None, None, None, "orchestrator_error")

    return parse_orchestrator_response(raw, valid_speakers=speakers)


def _extraversion_rank(group: "GroupSession", display_name: str) -> float:
    pid = _display_to_profile(group, display_name)
    if not pid or pid not in group.students:
        return 0.5
    label = getattr(group.students[pid].student, "Extraversion", "Low")
    return 1.0 if str(label) == "High" else 0.25


def _jordan_restraint_active(group: "GroupSession") -> bool:
    """True if teacher recently redirected to a quieter roster student and they haven't spoken."""
    restraint = getattr(group, "listen_restraint", None) or {}
    for pid, rem in restraint.items():
        if rem > 0 and group.display_names.get(pid, "").lower() == "jordan":
            return True
    transcript = group.transcript or []
    last_teacher = None
    for entry in reversed(transcript):
        if entry.get("speaker_type") == "teacher":
            last_teacher = entry
            break
    if not last_teacher:
        return False
    text = (last_teacher.get("content") or "").lower()
    roster = set(roster_display_names(group))
    # Quiet / redirected targets present on this roster only.
    named_quiet: List[str] = []
    if "Maya" in roster and "maya" in text:
        named_quiet.append("Maya")
    if "Alex" in roster and "alex" in text:
        named_quiet.append("Alex")
    if not named_quiet:
        return False
    # Restraint lifts once a named quiet student has spoken after that teacher turn.
    seen_teacher = False
    for entry in transcript:
        if entry is last_teacher:
            seen_teacher = True
            continue
        if not seen_teacher:
            continue
        if entry.get("speaker_type") != "student":
            continue
        name = group.display_names.get(entry.get("speaker_id"), "")
        if name in named_quiet:
            return False
    return True


def _pick_alternate_speaker(
    group: "GroupSession",
    *,
    exclude: Sequence[str],
    prefer_high_e: bool = True,
) -> Optional[str]:
    """Pick an eligible alternate from the session roster; optionally prefer High E."""
    observers = set(group.last_observers or [])
    restraint = _jordan_restraint_active(group)
    listen_hold = getattr(group, "listen_restraint", None) or {}
    speakers = roster_display_names(group)
    ranked: List[tuple[float, int, str]] = []
    for roster_i, name in enumerate(speakers):
        if name in exclude:
            continue
        pid = _display_to_profile(group, name)
        if not pid or pid in observers:
            continue
        if listen_hold.get(pid, 0) > 0:
            continue
        if restraint and name == "Jordan":
            continue
        rank = _extraversion_rank(group, name) if prefer_high_e else 0.0
        ranked.append((rank, -roster_i, name))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def _apply_speaker_guard(
    group: "GroupSession",
    decision: OrchestratorDecision,
) -> OrchestratorDecision:
    """Block same-student back-to-back unless they were directly addressed."""
    if decision.action != "continue" or not decision.next_speaker:
        return decision
    last_name, _ = _last_student_display(group)
    if not last_name:
        return decision
    # Different speaker already — keep (extraversion bias may still adjust).
    if decision.next_speaker != last_name:
        return decision
    # Same as last speaker: only OK if answering a question aimed at them.
    if decision.responding_to == decision.next_speaker:
        return decision
    alt = _pick_alternate_speaker(group, exclude=(last_name,))
    if not alt:
        return decision
    return OrchestratorDecision(
        action="continue",
        next_speaker=alt,
        responding_to=decision.responding_to,
        expected_move=decision.expected_move,
        context=(decision.context or "") + "; no back-to-back",
    )


def _apply_extraversion_bias(
    group: "GroupSession",
    decision: OrchestratorDecision,
) -> OrchestratorDecision:
    """Prefer High-E volunteers for open debate; keep Low-E when addressed or asking."""
    if decision.action != "continue" or not decision.next_speaker:
        return decision
    if decision.responding_to == decision.next_speaker:
        return decision
    if decision.expected_move == "asking":
        return decision
    # 2-student FE path: do not steal turns via High-E bias (only Maya/Jordan).
    if len(group.profile_ids or []) <= 2:
        return decision

    if _extraversion_rank(group, decision.next_speaker) >= 0.9:
        return decision  # already High E

    last_name, _ = _last_student_display(group)
    exclude = [decision.next_speaker]
    if last_name:
        exclude.append(last_name)
    alt = _pick_alternate_speaker(group, exclude=exclude, prefer_high_e=True)
    if not alt or _extraversion_rank(group, alt) < 0.9:
        return decision
    return OrchestratorDecision(
        action="continue",
        next_speaker=alt,
        responding_to=decision.responding_to,
        expected_move=decision.expected_move or "claim",
        context=(decision.context or "") + "; high-E preference",
    )


def _eligible_profile(
    group: "GroupSession",
    display_name: str,
    observers: Sequence[str],
) -> Optional[str]:
    pid = _display_to_profile(group, display_name)
    if not pid or pid in observers:
        return None
    return pid


def prepare_game_flow_peer_message(
    prev_name: str,
    content: str,
    *,
    expected_move: Optional[str] = None,
    context: str = "",
) -> str:
    base = prepare_peer_user_message(prev_name, content)
    move = (expected_move or "").strip().lower()
    if move and move in VALID_MOVES and move != "none":
        hint = f" Lean toward a {move} contribution."
        if context:
            hint += f" ({context})"
        base += (
            "\n\n[Group discussion — respond like a real middle-school student: "
            "short, casual, maybe hedging (I think, kinda, wait). No textbook tone."
            f"{hint}]"
        )
    else:
        base += (
            "\n\n[Group discussion — respond like a real middle-school student: "
            "short, casual, maybe hedging. No textbook tone.]"
        )
    return base


def _apply_move_bias(
    group: "GroupSession",
    decision: OrchestratorDecision,
) -> OrchestratorDecision:
    """Rule-based expected_move nudges for 2-student PST phone-plans fights."""
    if decision.action != "continue" or len(group.profile_ids or []) > 2:
        return decision
    last_entry = next(
        (
            e
            for e in reversed(group.transcript or [])
            if e.get("speaker_type") == "student"
        ),
        None,
    )
    if not last_entry:
        return decision
    last_text = (last_entry.get("content") or "").lower()
    teacher_msgs = [
        e.get("content", "")
        for e in (group.transcript or [])
        if e.get("speaker_type") == "teacher"
    ]
    last_teacher = (teacher_msgs[-1] if teacher_msgs else "").lower()

    move = decision.expected_move
    if re.search(
        r"\b(?:table|slower|coefficient|equation)\b", last_text
    ) and re.search(r"\b(?:rate|0\.?10|0\.?30|coefficient)\b", last_text):
        move = move or "relating"
    if "maya" in last_teacher and re.search(
        r"\b(?:wrong|critique|check|pointed out|fee)\b", last_teacher
    ):
        if decision.next_speaker == "Maya":
            move = "evidence"
    if move and move != decision.expected_move:
        return OrchestratorDecision(
            action=decision.action,
            next_speaker=decision.next_speaker,
            responding_to=decision.responding_to,
            expected_move=move,
            context=(decision.context or "") + "; move_bias",
        )
    return decision


def run_game_flow_continuation(
    group: "GroupSession",
    *,
    store: "GroupSessionStore",
    initial_reply_count: int = 0,
    max_rounds: int | None = None,
    student_cap: int | None = None,
    on_reply=None,
) -> tuple[List[dict], str]:
    """Generate peer replies until orchestrator stops or caps hit.

    ``on_reply(reply_dict)`` is invoked after each finalized peer reply (for streaming).
    """
    from app.config import (
        GROUP_GAME_FLOW_MAX_ROUNDS,
        GROUP_GAME_FLOW_STUDENT_CAP,
    )
    from app.group_context import trim_private_hist

    cap = GROUP_GAME_FLOW_MAX_ROUNDS if max_rounds is None else int(max_rounds)
    total_cap = (
        GROUP_GAME_FLOW_STUDENT_CAP if student_cap is None else int(student_cap)
    )
    msg_cap = student_msg_cap_for_roster(group)
    observers = list(group.last_observers or [])
    replies: List[dict] = []
    stop_reason = ""

    if cap <= 0:
        return [], "disabled"

    for round_idx in range(cap):
        students_so_far = initial_reply_count + len(replies)
        if students_so_far >= total_cap:
            stop_reason = "student_cap"
            break
        if count_students_since_teacher(group.transcript) >= msg_cap:
            stop_reason = "orchestrator_student_cap"
            break

        decision = decide_orchestrator_turn(group)
        decision = _apply_speaker_guard(group, decision)
        decision = _apply_extraversion_bias(group, decision)
        decision = _apply_move_bias(group, decision)
        if decision.action != "continue" or not decision.next_speaker:
            stop_reason = decision.context or "orchestrator_stop"
            break

        # Emit early thinking cue for the next peer before generation starts.
        if on_reply is not None:
            try:
                on_reply(
                    {
                        "_event": "thinking",
                        "profile_ids": [
                            _eligible_profile(group, decision.next_speaker, observers)
                            or ""
                        ],
                        "expected_move": decision.expected_move,
                        "responding_to": decision.responding_to,
                    }
                )
            except Exception:
                pass

        pid = _eligible_profile(group, decision.next_speaker, observers)
        if not pid:
            stop_reason = "observer_or_ineligible"
            break

        last_name, last_pid = _last_student_display(group)
        if last_name and last_pid:
            last_entry = next(
                (
                    e
                    for e in reversed(group.transcript or [])
                    if e.get("speaker_type") == "student"
                ),
                None,
            )
            last_content = (last_entry or {}).get("content") or ""
            incoming = prepare_game_flow_peer_message(
                last_name,
                last_content,
                expected_move=decision.expected_move,
                context=decision.context,
            )
        else:
            incoming = prepare_game_flow_peer_message(
                "Peer",
                group.last_teacher_message or "Continue the discussion.",
                expected_move=decision.expected_move,
                context=decision.context,
            )

        reply_text, _, _, _, help_seeking, is_question = store.student_reply(
            group,
            pid,
            incoming,
            "peer",
            group.turn,
            allow_help_seeking=False,
            multi_round=True,
            game_flow=True,
            expected_move=decision.expected_move,
            peer_prev_name=last_name or "",
        )
        trim_private_hist(group.students[pid])

        reply = {
            "speaker_id": pid,
            "content": reply_text,
            "help_seeking": help_seeking,
            "is_question": is_question,
            "source": "game_flow",
            "speak_reason": (
                f"game orchestrator → {decision.expected_move or 'continue'}: "
                f"{decision.context or 'peer reaction'}"
            ).strip(),
            "orchestrator": {
                "expected_move": decision.expected_move,
                "responding_to": decision.responding_to,
                "context": decision.context,
            },
        }
        replies.append(reply)
        if on_reply is not None:
            try:
                on_reply(reply)
            except Exception:
                logger.exception("on_reply callback failed")

        group.orchestration_log.append(
            {
                "turn": group.turn,
                "kind": "game_flow",
                "mode": group.last_mode,
                "speakers": [pid],
                "observers": observers,
                "round": round_idx + 1,
                "orchestrator": {
                    "next_speaker": decision.next_speaker,
                    "responding_to": decision.responding_to,
                    "expected_move": decision.expected_move,
                    "context": decision.context,
                },
                "stop_policy": "game_orchestrator_or_cap",
            }
        )

        if not (reply_text or "").strip():
            stop_reason = "empty_reply"
            break

    if not stop_reason and replies:
        stop_reason = "orchestrator_stop"
    if not stop_reason and not replies:
        stop_reason = "orchestrator_stop"

    logger.info(
        "game_flow session=%s rounds=%s stop=%s",
        group.session_id,
        len(replies),
        stop_reason,
    )
    return replies, stop_reason
