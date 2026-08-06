"""Track B: session-fixed scaffold receptivity + pedagogical move gate.

Peer-group path only. Direct answers never raise mastery; scaffolds may tick
mastery for listeners with probability ρ (Big Five, narrow band).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, TYPE_CHECKING

from app.student.demo_students import DEMO_STUDENTS
from app.knowledge.scaffold_detector import (
    EXPLANATION_MARKERS,
    scaffold_is_worked_example,
)

if TYPE_CHECKING:
    from app.group.sessions import GroupSession, StudentSlot

MoveKind = Literal["direct_answer", "scaffold", "worked_example", "other"]

# Narrow band so personality tilts uptake modestly (PNAS regularity-aligned).
RHO_MIN = 0.38
RHO_MAX = 0.58
RHO_BASE = 0.45

DIRECT_ANSWER_PATTERNS = (
    re.compile(r"\bthe\s+answer\s+is\b", re.I),
    re.compile(r"\bit'?s\s+\d+\s*/\s*\d+\b", re.I),
    re.compile(r"\bis\s+\d+\s*/\s*\d+\s*\.?\s*$", re.I),
    re.compile(r"\bequals?\s+\d+\s*/\s*\d+\b", re.I),
    re.compile(r"\bso\s+(?:the\s+)?(?:answer|left|remaining)\s+is\b", re.I),
    re.compile(r"\bjust\s+(?:write|put|say)\b.{0,40}\d+\s*/\s*\d+", re.I),
    re.compile(r"\bi'?ll\s+tell\s+you\b", re.I),
    re.compile(r"\bcorrect\s+answer\b", re.I),
)

SCAFFOLD_CUE_PATTERNS = (
    re.compile(r"\bwhat\s+if\b", re.I),
    re.compile(r"\bhow\s+(?:might|could|would|can)\b", re.I),
    re.compile(r"\bwhy\s+(?:do|does|is|might)\b", re.I),
    re.compile(r"\bthink\s+about\b", re.I),
    re.compile(r"\bcan\s+you\s+(?:show|explain|try)\b", re.I),
    re.compile(r"\bwhat\s+do\s+you\s+(?:notice|think)\b", re.I),
    re.compile(r"\bbreak\s+it\s+(?:into|down)\b", re.I),
    re.compile(r"\bcompare\b", re.I),
    re.compile(r"\bdraw\b", re.I),
    re.compile(r"\bhint\b", re.I),
)

TEACHING_WARNING_MSG = (
    "Giving the answer doesn't raise student mastery — try a scaffold or question instead."
)


@dataclass(frozen=True)
class LearningGain:
    profile_id: str
    construct_id: str
    from_level: int
    to_level: int
    receptivity: float

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "construct_id": self.construct_id,
            "from_level": self.from_level,
            "to_level": self.to_level,
            "receptivity": round(self.receptivity, 3),
        }


@dataclass
class OpportunityResult:
    move: MoveKind
    teaching_warning: bool
    gains: List[LearningGain]
    applied: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "move": self.move,
            "teaching_warning": self.teaching_warning,
            "applied": self.applied,
            "reason": self.reason,
            "gains": [g.to_dict() for g in self.gains],
        }


def compute_receptivity(
    openness: float,
    conscientiousness: float,
    neuroticism: float,
) -> float:
    """ρ from Big Five: ↑C, ↑O, ↓N; clamped to a narrow band."""
    raw = (
        RHO_BASE
        + 0.12 * float(conscientiousness)
        + 0.10 * float(openness)
        - 0.10 * float(neuroticism)
    )
    return round(max(RHO_MIN, min(RHO_MAX, raw)), 3)


def receptivity_for_profile(profile_id: str) -> float:
    raw = DEMO_STUDENTS.get(str(profile_id).lower(), {})
    bf = raw.get("big_five") or {}
    return compute_receptivity(
        bf.get("openness", 0.5),
        bf.get("conscientiousness", 0.5),
        bf.get("neuroticism", 0.5),
    )


# Documented session bands for demo roster (computed from DEMO_STUDENTS).
PROFILE_RECEPTIVITY = {
    pid: receptivity_for_profile(pid) for pid in ("alex", "jordan", "sam", "morgan")
}


def classify_pedagogical_move(
    text: str,
    *,
    source: str = "teacher",
) -> MoveKind:
    """Classify an utterance as DIRECT_ANSWER | SCAFFOLD | WORKED_EXAMPLE | OTHER."""
    msg = (text or "").strip()
    if not msg:
        return "other"
    lower = msg.lower()

    if any(p.search(msg) for p in DIRECT_ANSWER_PATTERNS):
        # Pure tell: answer dump with little scaffolding language.
        scaffoldish = any(p.search(msg) for p in SCAFFOLD_CUE_PATTERNS) or any(
            m in lower for m in EXPLANATION_MARKERS
        )
        if not scaffoldish or len(msg.split()) < 18:
            return "direct_answer"

    if scaffold_is_worked_example(msg):
        return "worked_example"

    if any(p.search(msg) for p in SCAFFOLD_CUE_PATTERNS):
        return "scaffold"

    explanation_hits = sum(1 for m in EXPLANATION_MARKERS if m in lower)
    if explanation_hits >= 2 and len(msg.split()) >= 8:
        return "scaffold"
    if source == "peer" and explanation_hits >= 1 and len(msg.split()) >= 6:
        return "scaffold"

    return "other"


def worked_example_is_opportunity(text: str) -> bool:
    """Selective gate: worked examples count only when they explain, not just dump steps."""
    lower = (text or "").lower()
    if not scaffold_is_worked_example(text):
        return False
    # Require a reasoning cue so "watch me… answer is 3/8" stays gated out upstream.
    if any(p.search(text) for p in DIRECT_ANSWER_PATTERNS) and "because" not in lower:
        return False
    return any(
        cue in lower
        for cue in ("because", "so we", "first", "means", "denominator", "equivalent")
    )


def move_creates_opportunity(move: MoveKind, text: str) -> bool:
    if move == "scaffold":
        return True
    if move == "worked_example":
        return worked_example_is_opportunity(text)
    return False


def _target_construct(slot: "StudentSlot", group: "GroupSession") -> Optional[str]:
    primary = (slot.behavior_profile or {}).get("primary_construct")
    if primary:
        return primary
    required = (group.task_metadata or {}).get("required_constructs") or []
    return required[0] if required else None


def apply_scaffold_opportunity(
    group: "GroupSession",
    text: str,
    *,
    speaker_id: Optional[str] = None,
    source: str = "teacher",
    rng: Optional[random.Random] = None,
) -> OpportunityResult:
    """Apply Track B gate: listeners may gain +1 mastery with probability ρ_i.

    - Teacher utterance: all students are listeners.
    - Peer utterance: everyone except the speaker (no learning-by-teaching).
    - DIRECT_ANSWER: no gains + teaching_warning.
    """
    move = classify_pedagogical_move(text, source=source)
    if move == "direct_answer":
        return OpportunityResult(
            move=move,
            teaching_warning=True,
            gains=[],
            applied=False,
            reason="direct_answer_blocked",
        )

    if not move_creates_opportunity(move, text):
        return OpportunityResult(
            move=move,
            teaching_warning=False,
            gains=[],
            applied=False,
            reason="no_learning_opportunity",
        )

    rand = rng or random
    gains: List[LearningGain] = []
    for pid in group.profile_ids:
        if speaker_id and pid == speaker_id:
            continue  # listener-only for peer explanations
        slot = group.students[pid]
        rho = float(getattr(slot, "receptivity", None) or receptivity_for_profile(pid))
        if rand.random() >= rho:
            continue
        construct = _target_construct(slot, group)
        if not construct:
            continue
        current = int(slot.mastery_state.get(construct, 0))
        if current >= 3:
            continue
        new_level = current + 1
        slot.mastery_state[construct] = new_level
        slot.student.construct_mastery = dict(slot.mastery_state)
        gains.append(
            LearningGain(
                profile_id=pid,
                construct_id=construct,
                from_level=current,
                to_level=new_level,
                receptivity=rho,
            )
        )

    return OpportunityResult(
        move=move,
        teaching_warning=False,
        gains=gains,
        applied=True,
        reason="scaffold_opportunity",
    )


def summarize_receptivity(profile_ids: Sequence[str]) -> Dict[str, float]:
    return {pid: receptivity_for_profile(pid) for pid in profile_ids}
