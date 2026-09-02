"""Teacher-triggered reasoning warrants shaped by LPF stack, OCEAN, and misconceptions.

Students justify claims only when the teacher asks for reasoning (narrow detector).
Phone-plans v1; API structured for future tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Pattern, Sequence, Tuple, Union

from app.construct_text import mastery_descriptor
from app.fight_stance import is_phone_plans_task
from app.models import Student

WarrantTier = Literal["minimal", "partial", "rich"]
WarrantTone = Literal["assertive", "hedged", "brief"]

# --- Teacher reasoning-press detection (narrower than math_press) ---

_REASONING_PRESS_RE = re.compile(
    r"(?:"
    r"\bexplain\s+(?:your\s+)?reasoning|"
    r"\b(?:what'?s|what\s+is)\s+your\s+reasoning|"
    r"why\s*do\s*you\s*think|"
    r"whydo\s*you\s*think|"
    r"\bhow\s+do\s+you\s+know|"
    r"why\s+(?:would|won'?t|wont|will)\b|"
    r"\bexplain\s+why\b|"
    r"why\s+that\s+would\s+be\b|"
    r"why\s+.{0,40}\b(?:matter|case|cheaper|better)\b"
    r")",
    re.I,
)

_REASONING_EXCLUDE_RE = re.compile(
    r"\b(?:"
    r"just\s+say\s+it|"
    r"did\s+you\s+get\s*it|"
    r"get\s*it\b|"
    r"what'?s\s+the\s+answer|"
    r"try\s+x\s*=|"
    r"pick\s+a\s+number|"
    r"find\s+what\s+(?:are\s+)?the\s+wrong|"
    r"help\b.+?\bunderstand\s+why\b.+?\bwrong|"
    r"using\s+the\s+table\b.+?\bwrong"
    r")\b",
    re.I,
)

_GROUP_ADDRESS_RE = re.compile(
    r"\b(?:guys|everyone|both\s+of\s+you|canyon|class|group)\b",
    re.I,
)

_PROFILE_NAMES = {
    "jordan": re.compile(r"\bjordan\b", re.I),
    "maya": re.compile(r"\bmaya\b", re.I),
    "alex": re.compile(r"\balex\b", re.I),
    "riley": re.compile(r"\briley\b", re.I),
}

# --- Warrant signal patterns (phone plans) ---

_RATE_SIGNAL_RE = re.compile(
    r"\b0\.?10\b|\b0\.1\b|\b0\.?30\b|\b0\.3\b|\brate\b|\bper\s+text\b|\bcents?\b",
    re.I,
)
_FEE_SIGNAL_RE = re.compile(
    r"\$?\s*20\b|\bfee\b|starting\s+(?:fee|cost)|monthly\s+fee|at\s+the\s+start",
    re.I,
)
_TABLE_SIGNAL_RE = re.compile(
    r"\btable\b|\b50\b|\b15\b|\b25\b|\bplan\s+b\b.{0,30}\b(?:cheaper|less|better)\b",
    re.I,
)
_HEDGE_SIGNAL_RE = re.compile(
    r"\b(?:not\s+sure|i\s+think|kinda|maybe|i\s+guess|hard\s+to\s+say|don'?t\s+really\s+know)\b",
    re.I,
)
_RANGE_SIGNAL_RE = re.compile(
    r"\b(?:cross(?:over)?|break[- ]?even|100\s*texts?|x\s*=\s*100|"
    r"below\s+100|above\s+100|low\s+usage|more\s+texts?)\b",
    re.I,
)

_OVER_WARRANT_RE = [
    re.compile(r"\bset\s+(?:them|the\s+equations?)\s+equal\b", re.I),
    re.compile(r"\bx\s*=\s*100\b", re.I),
    re.compile(
        r"\bplan\s+b\b.{0,40}\b(?:under|below)\s*100\b.{0,60}\bplan\s+a\b",
        re.I,
    ),
]


def _named_profiles_in_msg(msg: str) -> List[str]:
    found: List[str] = []
    for pid, pat in _PROFILE_NAMES.items():
        if pat.search(msg or ""):
            found.append(pid)
    return found


def detect_teacher_reasoning_press(
    teacher_msg: str,
    *,
    addressed_profile: str = "",
    task_id: str = "",
) -> bool:
    """Return True when teacher asks for justification (phone-plans v1)."""
    if not is_phone_plans_task(task_id):
        return False
    msg = (teacher_msg or "").strip()
    if not msg:
        return False
    if _REASONING_EXCLUDE_RE.search(msg):
        return False
    if not _REASONING_PRESS_RE.search(msg):
        return False
    pid = (addressed_profile or "").strip().lower()
    if _GROUP_ADDRESS_RE.search(msg):
        return True
    named = _named_profiles_in_msg(msg)
    if not named:
        return True
    if pid and pid in named:
        return True
    if len(named) == 1 and not pid:
        return True
    return False


@dataclass(frozen=True)
class WarrantExpectation:
    tier: WarrantTier
    tone: WarrantTone
    required_patterns: Tuple[Pattern[str], ...]
    allow_hedge_pass: bool
    forbidden_patterns: Tuple[Pattern[str], ...]
    lp_brief: str
    rewrite_tone_hint: str
    construct_id: str
    mastery_level: int

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "tone": self.tone,
            "lp_brief": self.lp_brief,
            "rewrite_tone_hint": self.rewrite_tone_hint,
            "construct_id": self.construct_id,
            "mastery_level": self.mastery_level,
            "allow_hedge_pass": self.allow_hedge_pass,
        }


def _misconception_id(misconception: Union[dict, str, None]) -> str:
    if not misconception:
        return ""
    if isinstance(misconception, str):
        return misconception.strip()
    return str(misconception.get("id") or "").strip()


def _tone_for_student(
    behavior_mode: str,
    student: Optional[Student],
) -> WarrantTone:
    mode = (behavior_mode or "").upper()
    extraversion = getattr(student, "Extraversion", "Medium") if student else "Medium"
    if mode == "WRONG":
        return "brief" if extraversion == "Low" else "assertive"
    if mode in ("CONFUSED_HELPSEEKING", "PARTIAL_ATTEMPT_THEN_STUCK"):
        return "hedged"
    if extraversion == "Low":
        return "brief"
    if extraversion == "High" and mode != "NORMAL_ERROR_PROFILE":
        return "assertive"
    return "hedged"


def _tier_for_level(
    mastery_level: int,
    fight_phase: str,
    crossover_unlocked: bool,
) -> WarrantTier:
    level = max(0, min(3, int(mastery_level or 0)))
    phase = (fight_phase or "fight").strip().lower()
    if phase in ("nuanced", "resolved") or crossover_unlocked:
        return "rich" if level >= 3 else "partial"
    if level >= 3:
        return "partial"
    return "partial" if level >= 2 else "minimal"


def warrant_expectation(
    *,
    profile_id: str,
    behavior_profile: dict,
    student: Optional[Student] = None,
    misconception: Union[dict, str, None] = None,
    fight_phase: str = "fight",
    crossover_unlocked: bool = False,
    acknowledged_fee: bool = False,
    fee_press_count: int = 0,
) -> WarrantExpectation:
    """Build LP- and persona-shaped warrant expectation for a reasoning-press turn."""
    bp = behavior_profile or {}
    pid = (profile_id or bp.get("profile_id") or "").strip().lower()
    mode = (bp.get("behavior_mode") or "").upper()
    construct_id = (
        bp.get("primary_construct") or bp.get("construct_id") or "rate_comparison"
    )
    mastery_level = int(bp.get("student_stack_level") or 0)
    misc_id = _misconception_id(misconception)
    phase = (fight_phase or "fight").strip().lower()
    tier = _tier_for_level(mastery_level, phase, crossover_unlocked)
    tone = _tone_for_student(mode, student)
    lp_brief = mastery_descriptor(construct_id, mastery_level)

    required: List[Pattern[str]] = []
    forbidden: List[Pattern[str]] = list(_OVER_WARRANT_RE)
    allow_hedge = False
    rewrite_tone = "Sound like a real kid explaining your thinking."

    fee_uptake = acknowledged_fee or fee_press_count >= 2

    if pid == "jordan":
        required.append(_RATE_SIGNAL_RE)
        if fee_uptake or mode == "PARTIAL_ATTEMPT_THEN_STUCK":
            required.append(_FEE_SIGNAL_RE)
            rewrite_tone = (
                "Mention the $20/fee at low usage but still lean Plan A from rate — "
                "kid voice, not a tutor."
            )
        elif mode == "WRONG":
            rewrite_tone = (
                "Give a confident wrong rate-only reason (0.10 vs 0.30) — "
                "do NOT bring in the fee yet."
            )
        else:
            rewrite_tone = "Explain why you picked Plan A in kid words — stay imperfect."
    elif pid == "maya":
        required.append(_TABLE_SIGNAL_RE)
        allow_hedge = True
        if misc_id == "pr_table_missing_warrants" and phase == "fight":
            tier = "partial"
            rewrite_tone = (
                "Use your table numbers and hedge — you see Plan B cheaper at low "
                "texts but struggle to say why it stays that way."
            )
        elif phase in ("nuanced", "resolved"):
            required.append(_RANGE_SIGNAL_RE)
            rewrite_tone = (
                "Connect table to when each plan wins — still hedged, not a lecture."
            )
        else:
            rewrite_tone = (
                "Give table-grounded numbers with a brief hedge if unsure why."
            )
    else:
        required.append(_RATE_SIGNAL_RE)
        required.append(_TABLE_SIGNAL_RE)

    if not crossover_unlocked and phase == "fight":
        forbidden.extend(_OVER_WARRANT_RE)

    if tier == "minimal":
        rewrite_tone += f" Keep it minimal — {lp_brief}"
    elif tier == "partial":
        rewrite_tone += f" Partial warrant only — {lp_brief}"
    else:
        rewrite_tone += f" Richer but still kid-level — {lp_brief}"

    if tone == "assertive":
        rewrite_tone += " Sound confident."
    elif tone == "brief":
        rewrite_tone += " Keep it to one or two short sentences."
    elif tone == "hedged":
        rewrite_tone += " Hedge naturally (I think, not sure)."

    return WarrantExpectation(
        tier=tier,
        tone=tone,
        required_patterns=tuple(required),
        allow_hedge_pass=allow_hedge,
        forbidden_patterns=tuple(forbidden),
        lp_brief=lp_brief,
        rewrite_tone_hint=rewrite_tone,
        construct_id=construct_id,
        mastery_level=mastery_level,
    )


def has_warrant_signal(text: str, expectation: WarrantExpectation) -> bool:
    """True when draft satisfies at least one required signal (or hedge pass)."""
    body = text or ""
    if not body.strip():
        return False
    for pat in expectation.required_patterns:
        if pat.search(body):
            return True
    if expectation.allow_hedge_pass:
        if _HEDGE_SIGNAL_RE.search(body) and (
            _TABLE_SIGNAL_RE.search(body) or _RATE_SIGNAL_RE.search(body)
        ):
            return True
    return False


def is_over_warrant(
    text: str,
    expectation: WarrantExpectation,
    *,
    crossover_unlocked: bool = False,
    fight_phase: str = "fight",
) -> bool:
    phase = (fight_phase or "fight").strip().lower()
    if crossover_unlocked or phase in ("nuanced", "resolved"):
        if expectation.tier == "rich":
            return False
    body = text or ""
    for pat in expectation.forbidden_patterns:
        if pat.search(body):
            return True
    return False
