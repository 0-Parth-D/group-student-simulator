"""Deterministic personality, mastery, and behavior-based speaking policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING

from app.student.demo_students import DEMO_STUDENTS

if TYPE_CHECKING:
    from app.group.orchestrator import SpeakConstraints
    from app.group.sessions import GroupSession, StudentSlot


@dataclass(frozen=True)
class SpeakDecision:
    profile_id: str
    will_speak: bool
    score: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "will_speak": self.will_speak,
            "score": self.score,
            "reason": self.reason,
        }


def _trait(profile_id: str, slot: "StudentSlot", name: str) -> float:
    numeric = (DEMO_STUDENTS.get(profile_id, {}).get("big_five") or {}).get(
        name.lower()
    )
    if numeric is not None:
        return float(numeric)
    label = getattr(slot.student, name.capitalize(), "Low")
    return {"low": 0.25, "medium": 0.5, "med": 0.5, "high": 0.75}.get(
        str(label).lower(), 0.5
    )


def _primary_mastery(slot: "StudentSlot", group: "GroupSession") -> int:
    primary = (slot.behavior_profile or {}).get("primary_construct")
    required = (group.task_metadata or {}).get("required_constructs") or []
    construct = primary or (required[0] if required else None)
    if construct:
        return int(slot.mastery_state.get(construct, 0))
    return 0


def should_i_speak(
    slot: "StudentSlot",
    group: "GroupSession",
    constraints: "SpeakConstraints",
    last_peer_reply: Optional[str] = None,
    *,
    teacher_message: str = "",
    threshold: Optional[float] = None,
) -> SpeakDecision:
    """Score whether one student elects to speak on this teacher turn."""
    pid = slot.profile_id
    if pid in constraints.must_not_speak:
        return SpeakDecision(pid, False, 0.0, "must_not_speak (watch/listen)")
    if pid in constraints.must_speak:
        return SpeakDecision(pid, True, 1.0, "must_speak (directly addressed)")
    if pid not in constraints.may_speak:
        return SpeakDecision(pid, False, 0.0, "not eligible this turn")

    extraversion = _trait(pid, slot, "extraversion")
    neuroticism = _trait(pid, slot, "neuroticism")
    agreeableness = _trait(pid, slot, "agreeableness")
    mode = (slot.behavior_profile or {}).get("behavior_mode", "")
    mastery = _primary_mastery(slot, group)
    score = 0.15 + (0.35 * extraversion) + (0.15 * (1 - neuroticism))
    reasons = [f"{constraints.mode} floor", f"extraversion {extraversion:.1f}"]

    if mode == "WRONG":
        score += 0.25
        reasons.append("confident misconception")
    elif mode == "CONFUSED_HELPSEEKING":
        score -= 0.05
        reasons.append("confusion lowers volunteering")
    elif mode == "PARTIAL_ATTEMPT_THEN_STUCK":
        score += 0.10
        reasons.append("partial idea to share")
    elif mode == "NORMAL_ERROR_PROFILE":
        score += 0.05
        reasons.append("ready to contribute")

    if mastery <= 1:
        score += 0.10
        reasons.append("low mastery")
    elif mastery >= 3 and extraversion < 0.4:
        score -= 0.15
        reasons.append("quiet despite high mastery")

    lower_message = teacher_message.lower()
    if "disagree" in lower_message and mode == "WRONG":
        score += 0.20
        reasons.append("disagreement cue")
    if constraints.mode == "critique" and agreeableness < 0.5:
        score += 0.10
        reasons.append("low agreeableness")
    if last_peer_reply and mode == "WRONG":
        score += 0.05
        reasons.append("likely to challenge peer")

    prior_student = next(
        (
            entry.get("speaker_id")
            for entry in reversed(group.transcript[:-1])
            if entry.get("speaker_type") == "student"
        ),
        None,
    )
    if prior_student == pid:
        # Soft penalty so the same student can continue a thread, but others
        # still get a fair shot. Critique keeps a lighter touch.
        score -= 0.15 if constraints.mode == "critique" else 0.35
        reasons.append("spoke last round")

    score = round(max(0.0, min(1.0, score)), 2)
    cutoff = threshold
    if cutoff is None:
        cutoff = 0.45 if constraints.mode == "discuss" else 0.50 if constraints.mode == "critique" else 0.50
    return SpeakDecision(pid, score >= cutoff, score, " + ".join(reasons))


def self_select(
    candidates: Sequence[str],
    group: "GroupSession",
    constraints: "SpeakConstraints",
    teacher_message: str,
    *,
    last_peer_reply: Optional[str] = None,
) -> tuple[List[str], List[dict]]:
    """Evaluate all students, then order volunteers by score and roster order."""
    candidate_set = set(candidates)
    by_id: dict[str, SpeakDecision] = {}
    for pid in group.profile_ids:
        if pid not in candidate_set and pid not in constraints.must_not_speak:
            decision = SpeakDecision(pid, False, 0.0, "not eligible this turn")
        else:
            decision = should_i_speak(
                group.students[pid],
                group,
                constraints,
                last_peer_reply,
                teacher_message=teacher_message,
            )
        by_id[pid] = decision

    roster_index = {pid: index for index, pid in enumerate(group.profile_ids)}
    ranked = sorted(
        (decision for decision in by_id.values() if decision.will_speak),
        key=lambda decision: (-decision.score, roster_index[decision.profile_id]),
    )
    max_speakers = 3 if constraints.mode == "discuss" else 2
    selected = ranked[:max_speakers]

    minimum = 2 if constraints.mode == "discuss" and len(candidates) >= 2 else 1
    if len(selected) < minimum:
        remaining = sorted(
            (
                by_id[pid]
                for pid in candidates
                if pid not in {item.profile_id for item in selected}
                and pid not in constraints.must_not_speak
            ),
            key=lambda decision: (-decision.score, roster_index[decision.profile_id]),
        )
        selected.extend(remaining[: minimum - len(selected)])

    selected_ids = [decision.profile_id for decision in selected]
    selected_set = set(selected_ids)
    decisions = []
    for pid in group.profile_ids:
        decision = by_id[pid]
        reason = decision.reason
        if pid in selected_set and not decision.will_speak:
            reason += "; discuss participation floor"
        decisions.append(
            SpeakDecision(
                pid,
                pid in selected_set,
                decision.score,
                reason,
            ).to_dict()
        )
    return selected_ids, decisions


def volunteer_select(
    candidates: Sequence[str],
    group: "GroupSession",
    constraints: "SpeakConstraints",
    teacher_message: str,
    *,
    last_peer_reply: Optional[str] = None,
    last_speaker: Optional[str] = None,
) -> tuple[List[str], List[dict]]:
    """Peer-continuation select: at most one true volunteer; no participation floor.

    Returns an empty speaker list when nobody clears the speak threshold so the
    discussion can end naturally instead of forcing another turn.
    """
    candidate_set = set(candidates)
    cutoff = (
        0.45
        if constraints.mode == "discuss"
        else 0.50
        if constraints.mode == "critique"
        else 0.50
    )
    by_id: dict[str, SpeakDecision] = {}
    for pid in group.profile_ids:
        if pid not in candidate_set or pid in constraints.must_not_speak:
            by_id[pid] = SpeakDecision(
                pid,
                False,
                0.0,
                (
                    "must_not_speak (watch/listen)"
                    if pid in constraints.must_not_speak
                    else "not eligible this turn"
                ),
            )
            continue
        decision = should_i_speak(
            group.students[pid],
            group,
            constraints,
            last_peer_reply,
            teacher_message=teacher_message,
            threshold=cutoff,
        )
        if last_speaker and pid == last_speaker:
            score = round(max(0.0, decision.score - 0.1), 2)
            by_id[pid] = SpeakDecision(
                pid,
                score >= cutoff,
                score,
                decision.reason + " + just spoke",
            )
        else:
            by_id[pid] = decision

    roster_index = {pid: index for index, pid in enumerate(group.profile_ids)}
    ranked = sorted(
        (decision for decision in by_id.values() if decision.will_speak),
        key=lambda decision: (-decision.score, roster_index[decision.profile_id]),
    )
    if last_speaker and len(ranked) > 1:
        ranked = [d for d in ranked if d.profile_id != last_speaker] + [
            d for d in ranked if d.profile_id == last_speaker
        ]
    selected_ids = [ranked[0].profile_id] if ranked else []
    selected_set = set(selected_ids)
    decisions = [
        SpeakDecision(
            pid,
            pid in selected_set,
            by_id[pid].score,
            by_id[pid].reason,
        ).to_dict()
        for pid in group.profile_ids
    ]
    return selected_ids, decisions
