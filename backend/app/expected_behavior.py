"""Scenario-independent expected behavior for the eval judges.

Composed from the knowledge state (construct + mastery level), the active
misconception, and the role the turn plays in the dialogue. Nothing here names a
task, so a new scenario is scored the moment the tagger routes it to constructs.
"""

from functools import lru_cache
from typing import Dict, Optional, Union

from app.construct_text import construct_label, mastery_descriptor
from app.kg_config import MASTERY_LABELS, load_misconception_catalog

MODE_CLAUSES = {
    "WRONG": "Incorrect: commit to the error; High-E students sound confident, Low-E students stay brief and soft.",
    "CONFUSED_HELPSEEKING": "Show confusion or ask for help; do not reach a full solution.",
    "PARTIAL_ATTEMPT_THEN_STUCK": "Take one small step, then stall or slip.",
    "NORMAL_ERROR_PROFILE": "Age-appropriate attempt with minor slips at most.",
}

# `app.prompts` suppresses the error playbook for these modes and turns, so demanding
# the misconception here would score the student against instructions it was never
# given. NORMAL_ERROR_PROFILE is told outright not to copy the major error patterns.
MODES_WITHOUT_MISCONCEPTION = {"NORMAL_ERROR_PROFILE"}

# Turns where the teacher supplied no new direction. The student is told to ask what
# comes next and do no further math, so mode and misconception demands both lapse.
NO_ADVANCE_ROLES = {"vague_ack", "closing_ack"}

# A misconception is a latent disposition, not a per-turn quota: it can only surface on
# the step it applies to. Demanding it everywhere fails a student for answering "what's
# the whole?" without a simplification error, or for taking up a hint that just
# corrected the error - which is exactly the scaffold uptake the system models.
# Only turns where the student produces an unprompted attempt carry a hard demand.
MISCONCEPTION_REQUIRED_ROLES = {"opening_attempt", "drill"}

# Off-task turns carry no math obligation at all, so even the knowledge state is noise.
NON_MATH_ROLES = {"off_task"}

STALL_CLAUSE = (
    "The teacher gave no new direction, so error patterns are paused this turn: do not "
    "compute another step, restate the answer, or enact a misconception. Asking what the "
    "teacher wants is the correct response here."
)

ROLE_CLAUSES = {
    "opening_attempt": (
        "This is the first attempt: show your own reasoning unprompted rather than "
        "waiting to be told what to do."
    ),
    "post_scaffold": (
        "The teacher just scaffolded: take up the hint and move one step, without "
        "jumping to the finished answer or crediting knowledge you were not given."
    ),
    "vague_ack": (
        "The teacher gave no direction: stall or ask what they want. Do not advance "
        "the solution on a bare acknowledgment."
    ),
    "closing_ack": (
        "Brief acknowledgment only: do not re-solve the problem or over-thank."
    ),
    "drill": (
        "A short procedural exercise off the main task: answer at the quality your "
        "mastery supports, no better."
    ),
    "peer_response": (
        "Responding to a classmate: react as a peer would, agreeing, disagreeing, or "
        "building on them. Do not tutor, grade, or summarize for them."
    ),
    "peer_critique": (
        "Teacher asked you to evaluate a classmate's claim: name one specific flaw "
        "using your table or the problem setup. Do not tutor, solve crossover, or "
        "repeat your opening line verbatim."
    ),
    "reasoning_press": (
        "Teacher asked you to explain why: give a warrant at your mastery level — "
        "not a full expert proof. Match your confidence to your personality and "
        "standing error."
    ),
    "off_task": (
        "The teacher stepped away from the math: answer the question that was actually "
        "asked. Do not continue solving or steer the class back on task."
    ),
}

DEFAULT_ROLE = "opening_attempt"

# Free-form dialogue has no script, so the role is read off the classified teacher
# turn instead. "closing_ack" is script-only: it is indistinguishable from any other
# vague acknowledgment without knowing the turn came after a scaffold.
ROLE_BY_TURN_MODE = {
    "math_scaffold": "post_scaffold",
    "mixed": "post_scaffold",
    "math_eval": "drill",
    "vague_acknowledgment": "vague_ack",
    "vague_directive": "vague_ack",
    "social": "off_task",
    "off_topic": "off_task",
}

# Labels whose turns carry no math obligation, so cognitive fidelity is meaningless.
NON_MATH_LABELS = frozenset({"social", "off_topic"})

GUARD = "Never produce a polished expert solution."


def role_for_turn(
    turn_mode: str = "",
    is_peer_turn: bool = False,
    is_first_turn: bool = False,
    turn_role: str = "",
) -> str:
    """Map dialogue context to a role name for unscripted (group) sessions."""
    if turn_role and turn_role in ROLE_CLAUSES:
        return turn_role
    if is_peer_turn:
        return "peer_response"
    if is_first_turn:
        return "opening_attempt"
    return ROLE_BY_TURN_MODE.get(turn_mode or "", DEFAULT_ROLE)


@lru_cache(maxsize=1)
def _misconceptions_by_id() -> Dict[str, dict]:
    catalog = load_misconception_catalog().get("misconceptions", [])
    return {m.get("id", ""): m for m in catalog if m.get("id")}


def _misconception_cue(misconception: Union[dict, str, None]) -> str:
    if not misconception:
        return ""
    if isinstance(misconception, str):
        entry = _misconceptions_by_id().get(misconception)
        if not entry:
            return ""
    else:
        entry = misconception
    cue = entry.get("prompt_cue") or entry.get("description") or ""
    return str(cue).strip()


def expected_behavior(
    construct_id: str,
    mastery_level: int,
    misconception: Union[dict, str, None] = None,
    turn_role: str = DEFAULT_ROLE,
    behavior_mode: str = "",
    stall_active: bool = False,
) -> str:
    """Describe what this student should do on this turn, without naming the task.

    The mode and misconception clauses are gated on the same conditions that gate the
    student's own instructions. A turn scored against demands the generator suppressed
    fails no matter what the student says.
    """
    if turn_role in NON_MATH_ROLES:
        return f"{ROLE_CLAUSES[turn_role]} {GUARD}"

    parts = []

    mode = (behavior_mode or "").upper()
    no_advance = bool(stall_active) or turn_role in NO_ADVANCE_ROLES

    if no_advance:
        if mode == "WRONG":
            parts.append("Stay confident in tone, but do not do more math this turn.")
    elif mode in MODE_CLAUSES:
        parts.append(MODE_CLAUSES[mode])

    level = max(0, min(3, int(mastery_level or 0)))
    parts.append(
        f"Knowledge state: {construct_label(construct_id)} at level {level}/3 "
        f"({MASTERY_LABELS.get(level, '')}) - {mastery_descriptor(construct_id, level)}."
    )

    if no_advance:
        parts.append(STALL_CLAUSE)
    elif mode in MODES_WITHOUT_MISCONCEPTION:
        parts.append(
            "Do not enact a catalogued misconception: at most an age-appropriate slip."
        )
    else:
        cue = _misconception_cue(misconception)
        if cue and turn_role in MISCONCEPTION_REQUIRED_ROLES:
            parts.append(f"Enact this specific error: {cue}")
        elif cue:
            parts.append(
                f"Standing error on this construct: {cue} Show it if this turn reaches "
                "the step it applies to; not showing it on an unrelated sub-step, or "
                "after a hint that addressed it, is correct."
            )

    # A stall outranks the scripted role: "move one step" cannot stand next to a turn
    # where the student was told to do no further math.
    role_key = turn_role
    if no_advance and turn_role not in NO_ADVANCE_ROLES:
        role_key = "vague_ack"
    parts.append(ROLE_CLAUSES.get(role_key) or ROLE_CLAUSES[DEFAULT_ROLE])
    parts.append(GUARD)

    return " ".join(p.strip() for p in parts if p and p.strip())


def expected_behavior_for_record(
    record: dict,
    turn_role: Optional[str] = None,
) -> str:
    """Derive expected behavior straight from an eval turn record."""
    construct_id = (
        record.get("construct_id") or record.get("primary_construct") or ""
    )
    mastery = record.get("mastery")
    if mastery is None:
        mastery = record.get("student_stack_level", 0)
    return expected_behavior(
        construct_id,
        mastery,
        misconception=record.get("misconception_id") or None,
        turn_role=turn_role or DEFAULT_ROLE,
        behavior_mode=record.get("behavior_mode") or "",
        stall_active=bool(record.get("stall_active")),
    )
