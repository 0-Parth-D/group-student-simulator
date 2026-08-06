"""Probabilistic help-seeking: when a student asks the teacher a clarifying question."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from app.student.models import Student
from app.knowledge.turn_classifier import TurnMode

# Cooldown so students do not pepper the teacher every turn.
MIN_TURNS_BETWEEN_HELP_SEEK = 2

# Soft caps — help-seeking should feel occasional even when confused.
_MAX_P = 0.62

_STYLE_HINTS = {
    "anxious_helpless": (
        "Sound worried and stuck — e.g. 'I don't get what … means' or "
        "'I'm stuck on … can you help?'"
    ),
    "talkative_anxious": (
        "Ask out loud with nervous energy — e.g. 'Wait, how do I even start …?' "
        "or 'What does … mean here?'"
    ),
    "talkative_confident": (
        "Rare for you — if you ask, keep it brief and direct: "
        "'Hold on — what does … mean?'"
    ),
    "passive_minimal": (
        "Keep it short and flat: 'what?', 'how?', or '…what does that mean?'"
    ),
    "one_word_silence": (
        "Barely ask — one short question or trailing off: '…what?' / 'um… how?'"
    ),
    "give_up_quickly": (
        "Ask once then stop: 'can you help with …?' or 'I don't get this part'"
    ),
    "hesitant_uncertain": (
        "Hedge politely: 'I'm not sure about … — could you explain?'"
    ),
}


@dataclass(frozen=True)
class HelpSeekDecision:
    triggered: bool
    probability: float
    roll: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "probability": self.probability,
            "roll": self.roll,
            "reason": self.reason,
        }


def _trait_level(student: Student, name: str) -> str:
    return str(getattr(student, name, "Low") or "Low")


def _base_from_mode(mode: str) -> float:
    if mode == "CONFUSED_HELPSEEKING":
        return 0.38
    if mode == "PARTIAL_ATTEMPT_THEN_STUCK":
        return 0.22
    if mode == "WRONG":
        return 0.06
    if mode == "NORMAL_ERROR_PROFILE":
        return 0.04
    return 0.08


def help_seeking_probability(
    student: Student,
    behavior_profile: Optional[dict],
    *,
    turn_mode: TurnMode = "math_scaffold",
    stall_active: bool = False,
    turns_since_last: Optional[int] = None,
    has_misconceptions: bool = False,
) -> tuple[float, str]:
    """Return (probability, reason) without rolling."""
    if turn_mode in ("social", "off_topic"):
        return 0.0, "social/off-topic turn"
    if stall_active:
        return 0.0, "stall already seeks direction"

    bp = behavior_profile or {}
    mode = bp.get("behavior_mode") or bp.get("mode") or ""
    style = bp.get("help_seek_style") or ""
    level = int(bp.get("student_stack_level", 0) or 0)
    gaps = bp.get("prerequisite_gaps") or []

    # Confident-wrong personas almost never ask — they guess.
    if mode == "WRONG" and style in ("talkative_confident", "passive_minimal"):
        return 0.02, "wrong-mode confident/passive rarely help-seeks"

    p = _base_from_mode(mode)
    reasons = [f"mode {mode or 'unknown'} base {p:.2f}"]

    if level <= 0:
        p += 0.18
        reasons.append("mastery missing")
    elif level == 1:
        p += 0.12
        reasons.append("mastery bad")
    elif level == 2:
        p += 0.05
        reasons.append("mastery partial")
    else:
        p -= 0.12
        reasons.append("mastery good")

    if gaps:
        p += 0.10
        reasons.append("prerequisite gaps")
    if has_misconceptions and mode in (
        "CONFUSED_HELPSEEKING",
        "PARTIAL_ATTEMPT_THEN_STUCK",
        "WRONG",
    ):
        p += 0.06
        reasons.append("active misconception")

    neuro = _trait_level(student, "Neuroticism")
    extra = _trait_level(student, "Extraversion")
    agree = _trait_level(student, "Agreeableness")
    open_ = _trait_level(student, "Openness")
    consc = _trait_level(student, "Conscientiousness")

    if neuro == "High":
        p += 0.16
        reasons.append("high neuroticism")
    elif neuro == "Low":
        p -= 0.08
        reasons.append("low neuroticism")

    if extra == "High" and neuro == "High":
        p += 0.08
        reasons.append("talkative-anxious")
    elif extra == "Low":
        p -= 0.10
        reasons.append("low extraversion")

    if agree == "Low":
        p -= 0.14
        reasons.append("low agreeableness (avoids help)")
    elif agree == "High":
        p += 0.04
        reasons.append("high agreeableness")

    if open_ == "High":
        p += 0.07
        reasons.append("curious (high openness)")
    elif open_ == "Low":
        p -= 0.04
        reasons.append("low openness")

    if consc == "High":
        p += 0.04
        reasons.append("careful clarifier")
    elif consc == "Low":
        p -= 0.05
        reasons.append("gives up without asking")

    if turns_since_last is not None and turns_since_last < MIN_TURNS_BETWEEN_HELP_SEEK:
        p *= 0.12
        reasons.append(f"cooldown ({turns_since_last} turns since last)")

    p = max(0.0, min(_MAX_P, p))
    return round(p, 3), " + ".join(reasons)


def decide_help_seeking(
    student: Student,
    behavior_profile: Optional[dict],
    *,
    turn_mode: TurnMode = "math_scaffold",
    stall_active: bool = False,
    turns_since_last: Optional[int] = None,
    has_misconceptions: bool = False,
    allow: bool = True,
    rng: Optional[random.Random] = None,
) -> HelpSeekDecision:
    """Roll whether this turn should steer the student toward asking the teacher."""
    if not allow:
        return HelpSeekDecision(False, 0.0, 1.0, "help-seeking not allowed this speaker")

    p, reason = help_seeking_probability(
        student,
        behavior_profile,
        turn_mode=turn_mode,
        stall_active=stall_active,
        turns_since_last=turns_since_last,
        has_misconceptions=has_misconceptions,
    )
    if p <= 0:
        return HelpSeekDecision(False, p, 1.0, reason)

    roller = rng.random if rng is not None else random.random
    roll = roller()
    triggered = roll < p
    detail = f"{reason}; roll {roll:.3f} {'<' if triggered else '>='} p {p:.3f}"
    return HelpSeekDecision(triggered, p, round(roll, 3), detail)


def _gap_hint(behavior_profile: Optional[dict]) -> str:
    bp = behavior_profile or {}
    gaps = bp.get("prerequisite_gaps") or []
    for gap in gaps[:1]:
        desc = gap.get("gap_description") or gap.get("construct") or ""
        if desc:
            return str(desc)[:120]
    errors = bp.get("error_types") or bp.get("error_labels") or []
    if errors:
        return str(errors[0])[:120]
    primary = bp.get("primary_construct") or "this step"
    return f"the part about {primary}"


def build_help_seeking_prompt_block(
    student: Student,
    behavior_profile: Optional[dict],
    *,
    group_context: bool = False,
) -> str:
    """System-prompt addendum when help-seeking is triggered for this turn."""
    bp = behavior_profile or {}
    style = bp.get("help_seek_style") or "hesitant_uncertain"
    hint = _STYLE_HINTS.get(style, _STYLE_HINTS["hesitant_uncertain"])
    gap = _gap_hint(bp)
    name = student.student_id or "Student"

    lines = [
        "[HELP-SEEKING THIS TURN — PRIORITY]",
        f"You ({name}) feel overwhelmed, stuck, or unclear on this step.",
        "Ask the teacher ONE specific clarifying question or express a concrete doubt "
        "(a word, a step, or what to try next).",
        "Do NOT solve the step or state a final answer.",
        "Sound like a real middle-schooler — avoid robotic lines like "
        "'I am confused' or 'I lack knowledge'.",
        f"Style: {hint}",
        f"Ground your question in: {gap}",
    ]
    if group_context:
        lines.append(
            "Ask the teacher (not a classmate). Keep it to one short question so others can still talk."
        )
    lines.append("This overrides 'never ask for help' / WRONG-mode guessing for this turn only.")
    return "\n".join(lines)


def turns_since_help_seek(
    turn_history: Sequence[dict],
    current_turn: int,
) -> Optional[int]:
    """Turns since last help-seeking reply; None if never."""
    last = None
    for entry in turn_history or []:
        if entry.get("help_seeking"):
            last = entry.get("turn")
    if last is None:
        return None
    return max(0, int(current_turn) - int(last))


def reply_looks_like_question(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    lower = str(text).lower()
    if "?" in text:
        return True
    cues = (
        "what does",
        "what do i",
        "how do i",
        "can you",
        "could you",
        "i don't get",
        "i dont get",
        "i'm stuck",
        "im stuck",
        "what should i",
        "which part",
        "what means",
    )
    return any(c in lower for c in cues)
