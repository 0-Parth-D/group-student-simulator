"""LLM-critique response refinement (discourse/LP gates, then personality/register).

Flow:
  draft → semantic repetition → reasoning warrant → discourse progress
       → maya critique rich → jordan passive → crossover gate
       → maya overcomplete → claim lock → adult register → critic LLM → rewrite
       → voice pass (kid register; preserve peer references when discourse passed)
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Union

from app.config import (
    REFINE_MAX_REVISIONS,
    REFINE_MODE,
    STUDENT_MAX_TOKENS_MATH,
    STUDENT_MAX_TOKENS_SOCIAL,
)
from app.expected_behavior import expected_behavior, role_for_turn
from app.fight_progress import check_semantic_repetition, maya_critique_is_rich
from app.llm import complete, complete_chat
from app.llm_config import LLMRole
from app.models import Student
from app.personality import response_length_hint
from app.turn_classifier import TurnMode

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Student reply generation outcome with draft/refine metadata."""

    final_reply: str
    draft: str
    revisions: int
    critic: Optional[dict] = None

    def __iter__(self) -> Iterator:
        """Unpack as ``(final_reply, revisions)`` for call-site compatibility."""
        yield self.final_reply
        yield self.revisions


def critic_for_log(critique: Optional[dict]) -> Optional[dict]:
    """Copy critic JSON without the internal ``pack`` (targets logged separately)."""
    if not critique:
        return None
    out = deepcopy(critique)
    out.pop("pack", None)
    return out


# Adult / tutor register markers (deterministic gate before LLM critic).
ADULT_REGISTER_MARKERS = (
    "therefore",
    "essentially",
    "in order to",
    "we can see",
    "as we can",
    "one approach",
    "let us",
    "consider that",
    "furthermore",
    "consequently",
    "it is important",
    "what you need to",
    "what we need to",
    "obviously superior",
    "for any large volume",
    "unit rate of change",
    "linear function",
    "represent each plan as",
    "compare the rates",
    "in conclusion",
    "to summarize",
)

HEDGE_FILLER_MARKERS = (
    "i think",
    "kinda",
    "kind of",
    "wait",
    "um",
    "uh",
    "idk",
    "i dunno",
    "maybe",
    "sorta",
    "i mean",
    "okay so",
    "ok so",
    "like,",
    "hold on",
)

CRITIC_SYSTEM = """You critique simulated middle-school student replies for a tutoring sim.
Return JSON only. Deterministic gates already checked discourse progress, crossover,
claim lock, and Maya over-complete — do NOT fail those again.

Check in this order:
(1) middle-school REGISTER (not adult tutor/textbook),
(2) personality / OCEAN fit,
(3) learning progression / behavior mode,
(4) misconception enactment.

Do NOT invent math the student must say. Judge whether THIS draft matches the targets.
If implied_plan is A or B, fail when the draft switches plans, agrees with a peer who
chose the opposite plan, or says "yeah same" / "I agree" on a peer turn.

REGISTER fail (tutor_tone / too_polished) when the draft:
- sounds like a tutor, textbook, or solution key (therefore, essentially, "we can see",
  "first… second…", lecturing a peer), OR
- is longer than ~2 sentences (3 if High Extraversion) with no hedging/fillers
  (I think, wait, kinda, um, like, okay so).

JSON schema:
{
  "ok": boolean,
  "claim": {"pass": boolean, "detail": string},
  "personality": {"pass": boolean, "detail": string},
  "learning": {"pass": boolean, "detail": string},
  "misconception": {"pass": boolean, "detail": string},
  "issues": [string],
  "brief": string
}

ok is true ONLY if claim, personality, learning, AND misconception all pass.
Treat tutor_tone / too_polished as personality failures (set personality.pass=false).
brief: one short sentence telling the rewriter what to fix (empty if ok).
issues: short machine tags e.g. claim_flip, yeah_same, personality_overtalk, tutor_tone,
too_polished, lp_overcomplete, misconception_missing, too_eager, too_quiet_when_high_E.
"""

VOICE_REWRITE_SYSTEM = """You rewrite ONE middle-school student's spoken classroom reply.
Change REGISTER / voice only. Keep the exact same math claim, numbers, plan stance, and errors.
Do not add new math, fix mistakes, or teach. Output only the spoken reply — no quotes or labels."""


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", (text or "").strip())
    return max(1, len([p for p in parts if p.strip()])) if (text or "").strip() else 0


def _has_hedge_or_filler(text: str) -> bool:
    lower = (text or "").lower()
    return any(h in lower for h in HEDGE_FILLER_MARKERS)


def ocean_filler_guidance(student: Optional[Student]) -> str:
    """How many / which fillers fit this student's OCEAN — not random slang spam."""
    if student is None:
        return (
            "Use at most 2 natural fillers (wait, okay so, I think, like, kinda, um, I mean) "
            "only at the start or when hedging/repairing. Never inside calculations."
        )
    lines = [
        "At most 2 fillers total. Only at turn-start or hedging/repair — never inside equations/numbers.",
    ]
    if student.Neuroticism == "High" or student.Extraversion == "Low":
        lines.append(
            "Prefer: wait, um, I think, kinda, idk — soft uncertainty fits this student."
        )
    elif student.Extraversion == "High":
        lines.append(
            "Prefer: okay so, wait (when correcting). Avoid stacking um/uh; stay eager but imperfect."
        )
    else:
        lines.append("Prefer: I think, like, okay so, wait.")
    if student.Agreeableness == "High":
        lines.append("Soft peer tone OK (yeah, I mean) without tutoring them.")
    if student.Conscientiousness == "High" and student.Extraversion == "High":
        lines.append(
            "If stating a short equation, one opener (okay so) is enough — don't hedge every clause."
        )
    return "\n".join(lines)


def check_adult_register(
    draft: str,
    student: Optional[Student] = None,
) -> Optional[dict]:
    """Deterministic adult/tutor register gate. Returns failing critic dict or None."""
    text = (draft or "").strip()
    if not text:
        return None
    lower = text.lower()
    issues: List[str] = []

    if any(m in lower for m in ADULT_REGISTER_MARKERS):
        issues.append("tutor_tone")
    if any(m in lower for m in (
        "divide both sides",
        "subtract from both sides",
        "move the x",
        "combine like terms",
        "you need to divide",
        "what you do is",
        "step one is",
        "first you divide",
    )):
        issues.append("tutor_tone")
    if re.search(r"\b(first[,:]|second[,:]|third[,:]|step\s*\d)\b", lower):
        issues.append("tutor_tone")
    if re.search(
        r"\b(what you need to (understand|do)|let me explain|the correct (way|approach))\b",
        lower,
    ):
        issues.append("tutor_tone")

    max_sent = 2
    if student is not None and student.Extraversion == "High":
        max_sent = 3
    if student is not None and student.Extraversion == "Low":
        max_sent = 2
    sc = _sentence_count(text)
    hedged = _has_hedge_or_filler(text)
    if sc > max_sent and not hedged:
        issues.append("too_polished")
    if len(text) > 220 and not hedged:
        issues.append("too_polished")

    # de-dupe preserving order
    seen = set()
    uniq = []
    for i in issues:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    if not uniq:
        return None

    return {
        "ok": False,
        "claim": {"pass": True, "detail": "deferred"},
        "personality": {
            "pass": False,
            "detail": "adult/tutor register: " + ",".join(uniq),
        },
        "learning": {"pass": True, "detail": "deferred"},
        "misconception": {"pass": True, "detail": "deferred"},
        "issues": uniq,
        "brief": (
            "Shorter middle-school voice. More 'I think / wait / kinda' if unsure. "
            "No teaching or textbook wording."
        ),
    }


_CONCESSIVE_MARKERS = (
    " but ",
    " though ",
    " still ",
    "i get ",
    "i see ",
    "yeah,",
    "yeah ",
    "i mean ",
    "even if ",
    "even though ",
)


def _still_defends_plan(text: str, plan: str) -> bool:
    """True when draft still argues for the assigned plan despite conceding a detail."""
    if plan == "A":
        return bool(
            re.search(r"\bplan\s*a\b", text)
            or re.search(r"\b0\.?10\b|\b0\.1\b|\b10\s*cents?\b", text)
            or re.search(r"\bcoefficient\b", text)
            or re.search(r"\brate\b.{0,30}\b(cheaper|less|smaller)\b", text)
            or re.search(r"\b(cheaper|less|smaller)\b.{0,30}\brate\b", text)
            or re.search(r"\bplan\s*a\b.{0,40}\b(better|wins?)\b", text)
        )
    return bool(
        re.search(r"\bplan\s*b\b", text)
        or re.search(r"\btable\b", text)
        or re.search(r"\b0\.?30\b|\b0\.3\b|\b30\s*cents?\b", text)
        or re.search(r"\bplan\s*b\b.{0,40}\b(cheaper|less|better)\b", text)
        or re.search(r"\b(cheaper|less)\b.{0,30}\bplan\s*b\b", text)
    )


def _is_concessive_opposition(text: str, plan: str) -> bool:
    """Allow 'I see X but still Plan Y' without treating it as a claim flip."""
    lower = (text or "").lower()
    if not any(m in lower for m in _CONCESSIVE_MARKERS):
        return False
    return _still_defends_plan(lower, plan)


JORDAN_PASSIVE_PHRASES = (
    "not sure what i'm supposed",
    "not sure what im supposed",
    "write it down or just say",
    "what am i supposed to do",
    "what i'm supposed to do",
    "what im supposed to do",
)


def check_jordan_passive(
    draft: str,
    student: Optional[Student],
    behavior_mode: str,
    *,
    profile_id: str = "",
) -> Optional[dict]:
    """Block passive/confused Jordan when WRONG mode expects confident pushback."""
    pid = (profile_id or "").strip().lower()
    if not pid and student is not None:
        pid = (student.student_id or "").strip().lower()
    if pid != "jordan":
        return None
    mode = (behavior_mode or "").strip().upper()
    if mode != "WRONG":
        return None
    lower = (draft or "").lower()
    if not any(p in lower for p in JORDAN_PASSIVE_PHRASES):
        return None
    return {
        "ok": False,
        "claim": {"pass": True, "detail": "deferred"},
        "personality": {
            "pass": False,
            "detail": "passive_when_should_be_confident",
        },
        "learning": {"pass": True, "detail": "deferred"},
        "misconception": {"pass": True, "detail": "deferred"},
        "issues": ["too_passive_jordan"],
        "brief": (
            "Stay confident in your rate argument; push back, don't go blank or ask "
            "what you're supposed to do."
        ),
    }


def check_crossover_gate(
    draft: str,
    *,
    crossover_unlocked: bool,
) -> Optional[dict]:
    """Block unprompted crossover solve (x=100) until teacher cues equate."""
    if crossover_unlocked:
        return None
    from app.fight_progress import detect_crossover_in_text

    if not detect_crossover_in_text(draft):
        return None
    return {
        "ok": False,
        "claim": {"pass": True, "detail": "deferred"},
        "personality": {"pass": True, "detail": "deferred"},
        "learning": {
            "pass": False,
            "detail": "crossover_before_teacher_cue",
        },
        "misconception": {"pass": True, "detail": "deferred"},
        "issues": ["lp_overcomplete"],
        "brief": (
            "Do not solve crossover or state x=100 yet — hint that rates differ "
            "or the fee matters, but no full equate/solve unless teacher asked."
        ),
    }


_DISCOURSE_TOKENS = (
    "15",
    "25",
    "20",
    "0.10",
    "0.1",
    "0.30",
    "0.3",
    "table",
    "plan a",
    "plan b",
    "fee",
    "rate",
)


def _norm_discourse(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _references_prior_speaker(
    draft: str,
    prior_peer_name: str,
    prior_peer_text: str,
) -> bool:
    lower = (draft or "").lower()
    name = (prior_peer_name or "").strip().lower()
    if name and name in lower:
        return True
    prior_lower = (prior_peer_text or "").lower()
    for tok in _DISCOURSE_TOKENS:
        if tok in prior_lower and tok in lower:
            return True
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", prior_lower):
        if m.group(0) in lower:
            return True
    return False


def check_discourse_progress(
    draft: str,
    *,
    prior_peer_name: str = "",
    prior_peer_text: str = "",
    expected_move: str = "",
    turn_role: str = "",
    is_peer_turn: bool = False,
    prior_replies: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Fail when peer/critique turn repeats without referencing prior speaker."""
    role = (turn_role or "").strip().lower()
    if role == "reasoning_press":
        return None
    if not is_peer_turn and role != "peer_critique":
        return None
    if not (prior_peer_text or prior_peer_name):
        return None
    norm = _norm_discourse(draft)
    if len(norm) < 12:
        return None
    if _references_prior_speaker(draft, prior_peer_name, prior_peer_text):
        return None
    recent = [_norm_discourse(p) for p in (prior_replies or ())[-2:] if p]
    if any(norm in p or p in norm for p in recent if len(p) >= 20):
        move_hint = ""
        move = (expected_move or "").strip().lower()
        if move == "evidence" or role == "peer_critique":
            move_hint = (
                " Add one number, table row, fee, or equation fragment reacting to "
                f"{prior_peer_name or 'your classmate'}."
            )
        elif move == "relating":
            move_hint = (
                f" Name what {prior_peer_name or 'your classmate'} just said; agree or "
                "disagree with one reason."
            )
        return {
            "ok": False,
            "claim": {"pass": True, "detail": "deferred"},
            "personality": {"pass": False, "detail": "no_discourse_progress"},
            "learning": {"pass": False, "detail": "no_peer_reference"},
            "misconception": {"pass": True, "detail": "deferred"},
            "issues": ["no_discourse_progress"],
            "brief": (
                "React to what your classmate just said — use their name or a number "
                "from their message; do not repeat your last line."
                f"{move_hint}"
            ),
        }
    return None


_MAYA_OVERCOMPLETE_RE = [
    re.compile(r"\bset\s+(?:them|the\s+equations?)\s+equal\b", re.I),
    re.compile(r"\bx\s*=\s*100\b", re.I),
    re.compile(r"\bcrossover\s+point\b", re.I),
    re.compile(
        r"\bplan\s+b\b.{0,40}\b(?:under|below)\s*100\b.{0,60}\bplan\s+a\b",
        re.I,
    ),
]


def check_maya_critique_rich(
    draft: str,
    *,
    profile_id: str = "",
    turn_role: str = "",
    prior_replies: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Block thin Maya critique that only repeats her opening table line."""
    if (turn_role or "").strip().lower() != "peer_critique":
        return None
    if (profile_id or "").strip().lower() != "maya":
        return None
    if maya_critique_is_rich(draft):
        return None
    norm = re.sub(r"\s+", " ", (draft or "").strip().lower())
    if len(norm) < 12:
        return None
    for prev in (prior_replies or ()):
        p = re.sub(r"\s+", " ", (prev or "").strip().lower())
        if len(p) < 20:
            continue
        if norm == p or norm in p or p in norm:
            return {
                "ok": False,
                "claim": {"pass": True, "detail": "deferred"},
                "personality": {"pass": True, "detail": "deferred"},
                "learning": {"pass": False, "detail": "maya_critique_thin"},
                "misconception": {"pass": True, "detail": "deferred"},
                "issues": ["maya_critique_thin"],
                "brief": (
                    "Name one specific flaw in Jordan's claim — the $20 starting fee, "
                    "starting cost, or why rate-only reasoning misses low usage. "
                    "Do not repeat your opening table line."
                ),
            }
    return None


def check_reasoning_warrant(
    draft: str,
    *,
    turn_role: str = "",
    behavior_profile: Optional[dict] = None,
    fight_phase: str = "fight",
) -> Optional[dict]:
    """Fail bare or over-complete warrants on teacher reasoning-press turns."""
    from app.reasoning_warrant import (
        WarrantExpectation,
        has_warrant_signal,
        is_over_warrant,
    )

    role = (turn_role or "").strip().lower()
    if role != "reasoning_press":
        return None
    bp = behavior_profile or {}
    expectation = bp.get("_warrant_expectation")
    if not isinstance(expectation, WarrantExpectation):
        return None
    crossover_unlocked = bool(bp.get("crossover_unlocked"))
    if is_over_warrant(
        draft,
        expectation,
        crossover_unlocked=crossover_unlocked,
        fight_phase=fight_phase,
    ):
        return {
            "ok": False,
            "claim": {"pass": True, "detail": "deferred"},
            "personality": {"pass": True, "detail": "deferred"},
            "learning": {"pass": False, "detail": "over_warrant"},
            "misconception": {"pass": True, "detail": "deferred"},
            "issues": ["over_warrant"],
            "brief": (
                f"Too much for your level — {expectation.rewrite_tone_hint} "
                "Do not tutor or solve crossover unprompted."
            ),
        }
    if not has_warrant_signal(draft, expectation):
        return {
            "ok": False,
            "claim": {"pass": True, "detail": "deferred"},
            "personality": {"pass": False, "detail": "bare_claim"},
            "learning": {"pass": False, "detail": "bare_claim"},
            "misconception": {"pass": True, "detail": "deferred"},
            "issues": ["bare_claim"],
            "brief": expectation.rewrite_tone_hint,
        }
    return None


def check_maya_overcomplete(
    draft: str,
    *,
    profile_id: str = "",
    fight_phase: str = "fight",
) -> Optional[dict]:
    """Block Maya from tutoring crossover before resolved phase."""
    pid = (profile_id or "").strip().lower()
    if pid != "maya":
        return None
    phase = (fight_phase or "fight").strip().lower()
    if phase == "resolved":
        return None
    text = draft or ""
    for pat in _MAYA_OVERCOMPLETE_RE:
        if pat.search(text):
            return {
                "ok": False,
                "claim": {"pass": True, "detail": "deferred"},
                "personality": {"pass": True, "detail": "deferred"},
                "learning": {"pass": False, "detail": "lp_overcomplete"},
                "misconception": {"pass": True, "detail": "deferred"},
                "issues": ["lp_overcomplete"],
                "brief": (
                    "Stay table-grounded and hedged — no set-them-equal, x=100, or "
                    "full piecewise summary yet."
                ),
            }
    return None


def check_claim_consistency(
    draft: str,
    implied_plan: str = "",
    *,
    is_peer_turn: bool = False,
    fight_phase: str = "fight",
) -> Optional[dict]:
    """Deterministic claim lock. Returns a failing critic dict or None if OK."""
    plan = (implied_plan or "").strip().upper()
    if plan not in ("A", "B"):
        return None
    text = (draft or "").lower()
    issues: List[str] = []
    phase = (fight_phase or "fight").strip().lower()
    concessive = phase == "fight" and _is_concessive_opposition(text, plan)
    if is_peer_turn and (
        "yeah same" in text
        or "same here" in text
        or "i agree" in text
        or "you're right" in text
        or "youre right" in text
    ):
        issues.append("yeah_same")
    if phase in ("nuanced", "resolved"):
        if plan == "A" and re.search(
            r"\bplan\s*a\b.{0,50}\b(always|every|no matter|obviously)\b", text
        ):
            issues.append("rate_always_regression")
        if plan == "A" and re.search(
            r"\b(fee|20)\b.{0,40}\b(doesn'?t|does not) matter\b", text
        ):
            issues.append("fee_blind_regression")
    else:
        if not concessive:
            if plan == "A":
                if re.search(
                    r"\bplan\s*b\b.{0,40}\b(better|cheaper|wins?|always)\b", text
                ) or re.search(
                    r"\b(better|cheaper|always)\b.{0,40}\bplan\s*b\b", text
                ):
                    issues.append("claim_flip")
            if plan == "B":
                if re.search(
                    r"\bplan\s*a\b.{0,40}\b(better|always|wins?)\b", text
                ) or re.search(r"\b(better|always)\b.{0,40}\bplan\s*a\b", text):
                    issues.append("claim_flip")
    if not issues:
        return None
    return {
        "ok": False,
        "claim": {"pass": False, "detail": ",".join(issues)},
        "personality": {"pass": True, "detail": "deferred"},
        "learning": {"pass": True, "detail": "deferred"},
        "misconception": {"pass": True, "detail": "deferred"},
        "issues": issues,
        "brief": (
            f"Keep arguing for Plan {plan}; do not agree with the other plan "
            "or say yeah same."
            if phase == "fight"
            else (
                f"Stay nuanced for Plan {plan}: acknowledge crossover/ranges "
                "you already found; do not revert to rate-only always-wins talk."
            )
        ),
    }


def _misconception_cue(misconception: Union[dict, str, None]) -> str:
    if not misconception:
        return ""
    if isinstance(misconception, str):
        return misconception.strip()
    return str(
        misconception.get("prompt_cue")
        or misconception.get("description")
        or misconception.get("id")
        or ""
    ).strip()


def _personality_target_block(student: Optional[Student], turn_mode: TurnMode) -> str:
    if student is None:
        return (
            "Sound like a real middle-schooler: short, casual, hedging when unsure. "
            "Not a tutor or textbook. Light fillers (wait, I think, kinda) OK when natural."
        )
    traits = student.personality_dict()
    lines = [
        f"Student: {student.student_id}",
        "OCEAN: " + ", ".join(f"{k}={v}" for k, v in traits.items()),
        response_length_hint(student, turn_mode),
        "Filler guidance:",
        ocean_filler_guidance(student),
    ]
    if student.Extraversion == "Low":
        lines.append(
            "Low Extraversion: quiet, brief, not eager to dominate; speak mainly when "
            "addressed; avoid long polished explanations."
        )
    else:
        lines.append(
            "High Extraversion: can jump in and sound eager/confident, but still age-appropriate "
            "and imperfect — not a teacher."
        )
    if student.Neuroticism == "High":
        lines.append(
            "High Neuroticism: stress-sensitive; hedges (I think, kinda, wait, um); "
            "not calmly lecturing."
        )
    else:
        lines.append(
            "Low Neuroticism: steadier tone; avoid excessive helpless panic unless confused mode."
        )
    if student.Agreeableness == "High":
        lines.append("High Agreeableness: cooperative, not harsh with peers.")
    else:
        lines.append("Lower Agreeableness: more direct; still not rude or tutoring.")
    lines.append(
        "REGISTER floor (all students): never sound like ChatGPT/tutor/textbook even when "
        "math is right or wrong."
    )
    return "\n".join(lines)


def build_expected_pack(
    behavior_profile: dict,
    *,
    student: Optional[Student] = None,
    misconception: Union[dict, str, None] = None,
    turn_mode: TurnMode = "math_scaffold",
    is_peer_turn: bool = False,
    turn_role: str = "",
) -> dict[str, str]:
    """Fetch personality + LP + misconception targets for critique/rewrite."""
    construct_id = (
        behavior_profile.get("primary_construct")
        or behavior_profile.get("construct_id")
        or ""
    )
    mastery = int(
        behavior_profile.get("student_stack_level")
        or (student.construct_mastery.get(construct_id, 0) if student else 0)
        or 0
    )
    turn_role_resolved = role_for_turn(
        turn_mode=turn_mode or "",
        is_peer_turn=is_peer_turn,
        is_first_turn=False,
        turn_role=turn_role or "",
    )
    misc = misconception
    if misc is None:
        misc = behavior_profile.get("active_misconception") or behavior_profile.get(
            "misconception_id"
        )
    lp_text = expected_behavior(
        construct_id,
        mastery,
        misconception=misc,
        turn_role=turn_role_resolved,
        behavior_mode=behavior_profile.get("behavior_mode") or "",
        stall_active=bool(behavior_profile.get("stall_active")),
    )
    misc_cue = _misconception_cue(misc)
    mode = (behavior_profile.get("behavior_mode") or "").upper()
    if turn_mode in ("social", "off_topic"):
        misc_note = "No misconception demand on social/off-topic turns."
        lp_text = (
            "Social/off-topic turn: brief friendly classroom reply only. "
            "No math, no Plan A/B claim, no misconception."
        )
    elif mode in ("", "NORMAL_ERROR_PROFILE") and not behavior_profile.get("stall_active"):
        misc_note = (
            misc_cue
            and f"Standing disposition (show only if this step applies): {misc_cue}"
            or "No hard misconception demand this turn."
        )
    elif behavior_profile.get("stall_active"):
        misc_note = "Stall turn: do not enact new math errors or advance the solution."
    else:
        misc_note = (
            f"Should surface this error when attempting the relevant step: {misc_cue}"
            if misc_cue
            else "Mode expects imperfect/wrong attempt — do not give a clean expert solution."
        )

    return {
        "personality": _personality_target_block(student, turn_mode),
        "learning": lp_text,
        "misconception": misc_note,
        "mode": mode or "UNKNOWN",
        "likely_correctness": str(behavior_profile.get("likely_correctness") or ""),
        "help_seek_style": str(behavior_profile.get("help_seek_style") or ""),
        "stack": f"{mastery} / target {behavior_profile.get('target_stack_level', 3)}",
    }


def critique_student_reply(
    draft: str,
    behavior_profile: dict,
    *,
    student: Optional[Student] = None,
    misconception: Union[dict, str, None] = None,
    turn_mode: TurnMode = "math_scaffold",
    is_peer_turn: bool = False,
    implied_plan: str = "",
    prior_replies: Optional[Sequence[str]] = None,
    fight_phase: str = "fight",
    expected_move: str = "",
    turn_role: str = "",
    prior_peer_name: str = "",
    prior_peer_text: str = "",
) -> dict[str, Any]:
    """LLM critic: discourse/LP gates → claim → register → personality."""
    pack = build_expected_pack(
        behavior_profile,
        student=student,
        misconception=misconception,
        turn_mode=turn_mode,
        is_peer_turn=is_peer_turn,
        turn_role=turn_role,
    )
    plan = (implied_plan or behavior_profile.get("implied_plan") or "").strip()
    profile_id = str(behavior_profile.get("profile_id") or "")
    # Social/off-topic: never force Plan A/B or misconception onto a greeting reply.
    non_math = turn_mode in ("social", "off_topic")
    if non_math:
        plan = ""
    rep_fail = None if non_math else check_semantic_repetition(
        draft, prior_replies or (), expected_move=expected_move
    )
    if rep_fail is not None:
        rep_fail["pack"] = pack
        return rep_fail

    warrant_fail = None if non_math else check_reasoning_warrant(
        draft,
        turn_role=turn_role,
        behavior_profile=behavior_profile,
        fight_phase=fight_phase,
    )
    if warrant_fail is not None:
        warrant_fail["pack"] = pack
        return warrant_fail

    discourse_fail = None if non_math else check_discourse_progress(
        draft,
        prior_peer_name=prior_peer_name,
        prior_peer_text=prior_peer_text,
        expected_move=expected_move,
        turn_role=turn_role,
        is_peer_turn=is_peer_turn,
        prior_replies=prior_replies,
    )
    if discourse_fail is not None:
        discourse_fail["pack"] = pack
        return discourse_fail

    maya_critique_fail = None if non_math else check_maya_critique_rich(
        draft,
        profile_id=profile_id,
        turn_role=turn_role,
        prior_replies=prior_replies,
    )
    if maya_critique_fail is not None:
        maya_critique_fail["pack"] = pack
        return maya_critique_fail

    jordan_fail = check_jordan_passive(
        draft,
        student,
        str(behavior_profile.get("behavior_mode") or ""),
        profile_id=profile_id,
    )
    if jordan_fail is not None:
        jordan_fail["pack"] = pack
        return jordan_fail

    cross_fail = None if non_math else check_crossover_gate(
        draft,
        crossover_unlocked=bool(behavior_profile.get("crossover_unlocked")),
    )
    if cross_fail is not None:
        cross_fail["pack"] = pack
        return cross_fail

    maya_fail = None if non_math else check_maya_overcomplete(
        draft,
        profile_id=profile_id,
        fight_phase=fight_phase,
    )
    if maya_fail is not None:
        maya_fail["pack"] = pack
        return maya_fail

    claim_fail = None if non_math else check_claim_consistency(
        draft, plan, is_peer_turn=is_peer_turn, fight_phase=fight_phase
    )
    if claim_fail is not None:
        claim_fail["pack"] = pack
        return claim_fail

    register_fail = check_adult_register(draft, student=student)
    if register_fail is not None:
        register_fail["pack"] = pack
        return register_fail

    claim_lock_block = (
        "## Claim lock\nimplied_plan=none (social/off-topic — do NOT demand a plan claim)\n\n"
        if non_math
        else f"## Claim lock\nimplied_plan={plan or 'none'}\n\n"
    )
    user = (
        f"## Draft reply\n{draft}\n\n"
        f"{claim_lock_block}"
        f"## Personality target\n{pack['personality']}\n\n"
        f"## Learning progression target\n{pack['learning']}\n\n"
        f"## Misconception target\n{pack['misconception']}\n\n"
        f"## Behavior snapshot\n"
        f"mode={pack['mode']}; likely_correctness={pack['likely_correctness']}; "
        f"help_seek_style={pack['help_seek_style']}; stack={pack['stack']}; "
        f"turn_mode={turn_mode}; peer={is_peer_turn}\n"
    )
    if non_math:
        user += (
            "\nIMPORTANT: This is a social/off-topic turn. Pass claim and misconception. "
            "Only fail for personality/register (adult tutor tone). Do NOT require Plan A/B "
            "or math error patterns.\n"
        )
    try:
        raw = complete(LLMRole.REPLY_CRITIC, CRITIC_SYSTEM, user)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("critic returned non-object")
        claim = data.get("claim") or {"pass": True, "detail": "ok"}
        personality = data.get("personality") or {}
        learning = data.get("learning") or {}
        misconception_b = data.get("misconception") or {}
        c_ok = bool(claim.get("pass", True))
        p_ok = bool(personality.get("pass", True))
        l_ok = bool(learning.get("pass", True))
        m_ok = bool(misconception_b.get("pass", True))
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        issue_strs = [str(i) for i in issues]
        if non_math:
            # Strip claim/misc failures that would rewrite greetings into Plan A/B.
            c_ok = True
            m_ok = True
            claim = {"pass": True, "detail": "n/a_social"}
            misconception_b = {"pass": True, "detail": "n/a_social"}
            issue_strs = [
                i
                for i in issue_strs
                if i
                not in (
                    "claim_flip",
                    "yeah_same",
                    "claim_missing",
                    "misconception_missing",
                    "lp_overcomplete",
                )
            ]
        # Force personality fail when critic tagged adult register but left pass=true.
        if any(t in issue_strs for t in ("tutor_tone", "too_polished")) and p_ok:
            p_ok = False
            personality = {
                "pass": False,
                "detail": personality.get("detail")
                or "adult/tutor register (critic issues)",
            }
        ok = (
            bool(data.get("ok", c_ok and p_ok and l_ok and m_ok))
            and c_ok
            and p_ok
            and l_ok
            and m_ok
        )
        if non_math:
            ok = p_ok and l_ok
        return {
            "ok": ok,
            "claim": claim,
            "personality": personality,
            "learning": learning,
            "misconception": misconception_b,
            "issues": issue_strs,
            "brief": str(data.get("brief") or "").strip(),
            "pack": pack,
        }
    except Exception as exc:
        logger.warning("LLM reply critique failed — accepting draft: %s", exc)
        return {
            "ok": True,
            "claim": {"pass": True, "detail": "critic_unavailable"},
            "personality": {"pass": True, "detail": "critic_unavailable"},
            "learning": {"pass": True, "detail": "critic_unavailable"},
            "misconception": {"pass": True, "detail": "critic_unavailable"},
            "issues": [],
            "brief": "",
            "pack": pack,
        }


def build_llm_refinement_message(critique: dict[str, Any]) -> str:
    pack = critique.get("pack") or {}
    issues = ", ".join(critique.get("issues") or []) or "persona/LP/misc mismatch"
    brief = critique.get("brief") or "Rewrite to match the targets."
    learning = str(pack.get("learning") or "")
    social = "social/off-topic" in learning.lower() or "no math, no plan" in learning.lower()
    claim_rule = (
        "Do NOT introduce Plan A/B, math claims, or misconception patterns. "
        "Keep it a brief friendly social reply."
        if social
        else (
            "Keep the same core math claim / plan stance; do NOT erase the classroom "
            "disagreement or switch to the peer's plan. Change voice, length, "
            "eagerness, and completeness only."
        )
    )
    move_hint = ""
    issue_set = set(critique.get("issues") or [])
    if "no_discourse_progress" in issue_set or "semantic_repetition" in issue_set:
        move_hint = (
            "\n\nMOVE HINT: React to your classmate — name them or cite a number "
            "from their last message; add one new detail (fee, table row, or rate contrast)."
        )
    if "maya_critique_thin" in issue_set:
        move_hint = (
            "\n\nMOVE HINT: Name the $20 fee, starting cost, or a rate-only blind spot — "
            "do not repeat your opening table-at-50 line."
        )
    return (
        "Your last reply failed a simulation quality check.\n"
        f"Critic brief: {brief}\n"
        f"Issues: {issues}\n\n"
        f"PERSONALITY TARGET:\n{pack.get('personality', '')}\n\n"
        f"LEARNING TARGET:\n{pack.get('learning', '')}\n\n"
        f"MISCONCEPTION TARGET:\n{pack.get('misconception', '')}\n\n"
        "Rewrite ONE middle-school student reply that meets the targets. "
        f"{claim_rule} No tutor/textbook tone. "
        "Use at most two natural fillers (wait, I think, kinda, okay so, um) only at "
        f"starts or hedges — never inside calculations. Output only the reply.{move_hint}"
    )


def _discourse_reference_token(
    draft: str,
    prior_peer_name: str,
    prior_peer_text: str,
) -> str:
    """Return a substring from draft that references the prior speaker (for voice guard)."""
    lower = (draft or "").lower()
    name = (prior_peer_name or "").strip().lower()
    if name and name in lower:
        return name
    prior_lower = (prior_peer_text or "").lower()
    for tok in _DISCOURSE_TOKENS:
        if tok in prior_lower and tok in lower:
            return tok
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", prior_lower):
        if m.group(0) in lower:
            return m.group(0)
    return ""


def rewrite_kid_voice(
    draft: str,
    *,
    student: Optional[Student] = None,
    turn_mode: TurnMode = "math_scaffold",
) -> str:
    """Second pass: lock math claim, rewrite surface register + OCEAN-aware fillers."""
    text = (draft or "").strip()
    if not text:
        return text
    # Stall / ultra-short: leave alone
    if len(text) < 12:
        return text

    length_hint = (
        response_length_hint(student, turn_mode)
        if student is not None
        else "1-2 short sentences."
    )
    claim_rule = (
        "- Keep it social: no Plan A/B, no new math claims.\n"
        if turn_mode in ("social", "off_topic")
        else "- Keep the EXACT same math claim, numbers, plan stance, and mistakes.\n"
        "- Do NOT add new steps or correct the student.\n"
    )
    user = (
        f"LENGTH: {length_hint}\n\n"
        f"OCEAN FILLER GUIDANCE:\n{ocean_filler_guidance(student)}\n\n"
        "Rules:\n"
        f"{claim_rule}"
        "- Sound like a real 6–8th grader talking out loud (not a tutor).\n"
        "- At most two fillers, only at start or when hedging/repairing.\n"
        "- No textbook words (therefore, essentially, we can see, first/second…).\n"
        "- Output ONLY the spoken reply.\n\n"
        f"DRAFT:\n{text}"
    )
    try:
        out = complete(LLMRole.STUDENT_REPLY, VOICE_REWRITE_SYSTEM, user)
        cleaned = (out or "").strip().strip('"').strip("'")
        return cleaned or text
    except Exception as exc:
        logger.warning("Voice rewrite failed — keeping prior reply: %s", exc)
        return text


def generate_with_refinement(
    system: str,
    history: list,
    behavior_profile: dict,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_revisions: int | None = None,
    turn_mode: TurnMode = "math_scaffold",
    scaffold_boost: Optional[dict] = None,
    student: Optional[Student] = None,
    misconception: Union[dict, str, None] = None,
    is_peer_turn: bool = False,
    implied_plan: str = "",
    prior_replies: Optional[Sequence[str]] = None,
    fight_phase: str = "fight",
    expected_move: str = "",
    turn_role: str = "",
    prior_peer_name: str = "",
    prior_peer_text: str = "",
    **_kwargs,
) -> GenerationResult:
    """Generate student reply; critique/rewrite; then kid-voice pass (#5/#6 + fillers).

    ``scaffold_boost`` retained for call-site compatibility (unused by LLM critic).
    Unpacks as ``(final_reply, revisions)`` via ``GenerationResult.__iter__``.
    """
    _ = scaffold_boost  # API compat
    if max_tokens is None:
        max_tokens = (
            STUDENT_MAX_TOKENS_SOCIAL
            if turn_mode in ("social", "off_topic")
            else STUDENT_MAX_TOKENS_MATH
        )
    elif turn_mode in ("social", "off_topic"):
        max_tokens = max(max_tokens, STUDENT_MAX_TOKENS_SOCIAL)

    mode = (REFINE_MODE or "llm").lower()
    revisions_cap = (
        REFINE_MAX_REVISIONS if max_revisions is None else max(0, int(max_revisions))
    )
    refine_off = mode in ("off", "none", "0", "false")
    if refine_off:
        revisions_cap = 0

    chat_kw: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        chat_kw["temperature"] = temperature

    draft = complete_chat(LLMRole.STUDENT_REPLY, system, history, **chat_kw)
    response = draft
    last_critique: Optional[dict] = None

    if revisions_cap <= 0:
        # Still run voice pass unless refine fully off (voice is the register floor).
        if not refine_off:
            response = rewrite_kid_voice(
                response, student=student, turn_mode=turn_mode
            )
        return GenerationResult(
            final_reply=response,
            draft=draft,
            revisions=0,
            critic=None,
        )

    revisions = 0
    discourse_ref_token = ""
    while revisions < revisions_cap:
        critique = critique_student_reply(
            response,
            behavior_profile,
            student=student,
            misconception=misconception,
            turn_mode=turn_mode,
            is_peer_turn=is_peer_turn,
            implied_plan=implied_plan
            or str((behavior_profile or {}).get("implied_plan") or ""),
            prior_replies=prior_replies,
            fight_phase=fight_phase,
            expected_move=expected_move,
            turn_role=turn_role,
            prior_peer_name=prior_peer_name,
            prior_peer_text=prior_peer_text,
        )
        last_critique = critique
        if critique.get("ok"):
            if not discourse_ref_token and (
                is_peer_turn or turn_role == "peer_critique"
            ):
                discourse_ref_token = _discourse_reference_token(
                    response, prior_peer_name, prior_peer_text
                )
            break

        logger.info(
            "LLM critique failed — refining (pass %d): personality=%s learning=%s "
            "misconception=%s issues=%s",
            revisions + 1,
            (critique.get("personality") or {}).get("pass"),
            (critique.get("learning") or {}).get("pass"),
            (critique.get("misconception") or {}).get("pass"),
            critique.get("issues"),
        )
        refine_msg = build_llm_refinement_message(critique)
        refine_hist = history + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": refine_msg},
        ]
        response = complete_chat(LLMRole.STUDENT_REPLY, system, refine_hist, **chat_kw)
        revisions += 1

    # #6: math claim locked above; surface register + OCEAN fillers in a dedicated pass.
    pre_voice = response
    response = rewrite_kid_voice(response, student=student, turn_mode=turn_mode)
    if discourse_ref_token and discourse_ref_token.lower() not in response.lower():
        response = pre_voice

    return GenerationResult(
        final_reply=response,
        draft=draft,
        revisions=revisions,
        critic=critic_for_log(last_critique),
    )
