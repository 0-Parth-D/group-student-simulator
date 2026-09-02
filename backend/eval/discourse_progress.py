"""Deterministic discourse metrics for PST phone-plans replay (no LLM)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from app.fight_progress import detect_crossover_in_text, maya_critique_is_rich
from app.reasoning_warrant import (
    detect_teacher_reasoning_press,
    has_warrant_signal,
    warrant_expectation,
)

JORDAN_PASSIVE_PHRASES = (
    "not sure what i'm supposed",
    "not sure what im supposed",
    "write it down or just say",
    "what am i supposed to do",
)

CROSSOVER_UNLOCK_CUE_RE = re.compile(
    r"\b(?:set\s+(?:them|the\s+equations?)\s+equal|equate|same\s+cost|crossover|"
    r"break[- ]?even|when\s+is\s+each\s+plan\s+better|which\s+plan\s+is\s+better|"
    r"when\s+are\s+they\s+equal|flip\s+point)\b",
    re.I,
)


def _maya_critique_rich(text: str) -> bool:
    return maya_critique_is_rich(text)

_TOKEN_RE = re.compile(r"[a-z0-9.]+|plan\s*[ab]", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _student_turns(transcript: Sequence[dict]) -> List[dict]:
    return [e for e in transcript if e.get("speaker_type") == "student"]


def _teacher_turns(transcript: Sequence[dict]) -> List[dict]:
    return [e for e in transcript if e.get("speaker_type") == "teacher"]


def peer_reference_rate(transcript: Sequence[dict], display_names: Dict[str, str]) -> float:
    """Fraction of student turns that reference the prior speaker or their numbers."""
    students = _student_turns(transcript)
    if len(students) < 2:
        return 0.0
    hits = 0
    peers = 0
    for i, entry in enumerate(students):
        if i == 0:
            continue
        prev = students[i - 1]
        prev_name = display_names.get(prev.get("speaker_id", ""), "").lower()
        prev_pid = (prev.get("speaker_id") or "").lower()
        text = (entry.get("content") or "").lower()
        if not text:
            continue
        peers += 1
        if prev_name and prev_name in text:
            hits += 1
            continue
        if prev_pid and prev_pid in text:
            hits += 1
            continue
        prev_tokens = set(_TOKEN_RE.findall((prev.get("content") or "").lower()))
        if prev_tokens & set(_TOKEN_RE.findall(text)):
            hits += 1
    return hits / peers if peers else 0.0


def repetition_streak(transcript: Sequence[dict]) -> int:
    """Longest run of identical normalized student claims."""
    norms = [_norm(e.get("content", "")) for e in _student_turns(transcript)]
    norms = [n for n in norms if len(n) >= 20]
    if not norms:
        return 0
    best = streak = 1
    for i in range(1, len(norms)):
        if norms[i] == norms[i - 1]:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
    return best


def jordan_passive_hits(transcript: Sequence[dict]) -> int:
    count = 0
    for entry in _student_turns(transcript):
        if (entry.get("speaker_id") or "").lower() != "jordan":
            continue
        lower = (entry.get("content") or "").lower()
        if any(p in lower for p in JORDAN_PASSIVE_PHRASES):
            count += 1
    return count


def unprompted_crossover(transcript: Sequence[dict]) -> int:
    """Count crossover solves before any teacher crossover cue."""
    unlocked = False
    count = 0
    for entry in transcript:
        if entry.get("speaker_type") == "teacher":
            if CROSSOVER_UNLOCK_CUE_RE.search(entry.get("content") or ""):
                unlocked = True
            continue
        if entry.get("speaker_type") != "student":
            continue
        if not unlocked and detect_crossover_in_text(entry.get("content") or ""):
            count += 1
    return count


def breakthrough_count(export: dict) -> int:
    bullets = export.get("breakthrough_bullets") or []
    return len(bullets)


def maya_critique_quality(transcript: Sequence[dict]) -> Dict[str, Any]:
    """After teacher asks Maya to find wrong things, did she go beyond table repeat?"""
    teachers = _teacher_turns(transcript)
    critique_idx = None
    for i, t in enumerate(teachers):
        msg = (t.get("content") or "").lower()
        if "maya" in msg and re.search(
            r"\b(?:wrong|find what|check him|pointed out|help\b.+?\bunderstand)\b",
            msg,
        ):
            critique_idx = i
            break
    if critique_idx is None:
        return {"found_critique_prompt": False, "maya_critique_rich": False}
    # Next Maya student line after that teacher turn in full transcript
    seen_critique_teacher = False
    maya_reply = ""
    for entry in transcript:
        if entry.get("speaker_type") == "teacher":
            if entry is teachers[critique_idx]:
                seen_critique_teacher = True
            continue
        if not seen_critique_teacher:
            continue
        if (entry.get("speaker_id") or "").lower() == "maya":
            maya_reply = entry.get("content") or ""
            break
    rich = _maya_critique_rich(maya_reply)
    return {
        "found_critique_prompt": True,
        "maya_critique_rich": rich,
        "maya_reply_snippet": maya_reply[:120],
    }


def _reasoning_press_pairs(
    transcript: Sequence[dict],
    *,
    task_id: str = "phone_plans_linear_01",
) -> List[Dict[str, Any]]:
    """Teacher reasoning presses and the next student reply per addressed speaker."""
    pairs: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    for entry in transcript:
        if entry.get("speaker_type") == "teacher":
            msg = entry.get("content") or ""
            if detect_teacher_reasoning_press(msg, task_id=task_id):
                pending = {"teacher_msg": msg, "addressed": None, "student_reply": ""}
            else:
                pending = None
            continue
        if entry.get("speaker_type") != "student" or pending is None:
            continue
        pid = (entry.get("speaker_id") or "").lower()
        if pending.get("addressed") and pending["addressed"] != pid:
            continue
        if not detect_teacher_reasoning_press(
            pending["teacher_msg"],
            addressed_profile=pid,
            task_id=task_id,
        ):
            continue
        pending["addressed"] = pid
        pending["student_reply"] = entry.get("content") or ""
        pairs.append(dict(pending))
        pending = None
    return pairs


def reasoning_warrant_metrics(
    transcript: Sequence[dict],
    *,
    task_id: str = "phone_plans_linear_01",
    slot_flags: Optional[dict] = None,
) -> Dict[str, Any]:
    """Warrant quality on teacher reasoning-press turns."""
    pairs = _reasoning_press_pairs(transcript, task_id=task_id)
    slot_flags = slot_flags or {}
    hits = 0
    bare = 0
    for pair in pairs:
        pid = pair.get("addressed") or ""
        flags = slot_flags.get(pid) or {}
        exp = warrant_expectation(
            profile_id=pid,
            behavior_profile={
                "behavior_mode": flags.get("behavior_mode", "WRONG"),
                "student_stack_level": flags.get("student_stack_level", 2),
                "primary_construct": flags.get("primary_construct", "rate_comparison"),
            },
            fight_phase=flags.get("fight_phase", "fight"),
            crossover_unlocked=bool(flags.get("crossover_unlocked")),
            acknowledged_fee=bool(flags.get("acknowledged_fee")),
            fee_press_count=int(flags.get("fee_press_count") or 0),
        )
        reply = pair.get("student_reply") or ""
        if has_warrant_signal(reply, exp):
            hits += 1
        else:
            bare += 1
    total = len(pairs)
    return {
        "reasoning_press_turns": total,
        "warrant_on_press_rate": round(hits / total, 3) if total else 0.0,
        "bare_claim_on_press": bare,
    }


def score_export(export: dict) -> Dict[str, Any]:
    transcript = export.get("transcript") or []
    display_names = export.get("display_names") or {}
    slot_flags = export.get("slot_flags") or {}
    jordan_flags = slot_flags.get("jordan") or {}
    warrant = reasoning_warrant_metrics(
        transcript,
        task_id=(export.get("task_id") or "phone_plans_linear_01"),
        slot_flags=slot_flags,
    )
    return {
        "peer_reference_rate": round(peer_reference_rate(transcript, display_names), 3),
        "repetition_streak": repetition_streak(transcript),
        "jordan_passive_hits": jordan_passive_hits(transcript),
        "unprompted_crossover": unprompted_crossover(transcript),
        "breakthrough_count": breakthrough_count(export),
        "fight_phase": export.get("fight_phase", "fight"),
        "phase_at_end": export.get("fight_phase", "fight"),
        "jordan_fee_uptake": bool(
            jordan_flags.get("acknowledged_fee")
            or jordan_flags.get("crossover_intuition")
        ),
        "maya_critique": maya_critique_quality(transcript),
        "crossover_unlocked": bool(export.get("crossover_unlocked")),
        **warrant,
    }
