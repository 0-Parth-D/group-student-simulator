"""Track B: receptivity bands, move gate, listener-only peer gains."""

from __future__ import annotations

import random

from app.group.sessions import GroupSession, StudentSlot
from app.student.learning_receptivity import (
    PROFILE_RECEPTIVITY,
    RHO_MAX,
    RHO_MIN,
    apply_scaffold_opportunity,
    classify_pedagogical_move,
    compute_receptivity,
    receptivity_for_profile,
)
from app.student.models import Student


def _student(name: str) -> Student:
    return Student(
        student_id=name,
        Openness="High",
        Conscientiousness="High",
        Extraversion="Low",
        Agreeableness="High",
        Neuroticism="Low",
    )


def _group(mastery: dict | None = None) -> GroupSession:
    mastery = mastery or {"part_whole": 1}
    slots = {}
    for pid, name in (("jordan", "Jordan"), ("sam", "Sam"), ("alex", "Alex")):
        slots[pid] = StudentSlot(
            pid,
            _student(name),
            mastery_state=dict(mastery),
            behavior_profile={
                "behavior_mode": "WRONG",
                "primary_construct": "part_whole",
            },
            receptivity=receptivity_for_profile(pid),
        )
    return GroupSession(
        session_id="track-b",
        task_text="Maria ate pizza. How much is left?",
        task_metadata={"required_constructs": ["part_whole"]},
        students=slots,
        profile_ids=["jordan", "sam", "alex"],
        display_names={"jordan": "Jordan", "sam": "Sam", "alex": "Alex"},
        started=True,
    )


def test_receptivity_bands_are_narrow_and_ordered():
    for pid, rho in PROFILE_RECEPTIVITY.items():
        assert RHO_MIN <= rho <= RHO_MAX, pid
    # Morgan (high C/O, low N) should be at least as receptive as Jordan.
    assert PROFILE_RECEPTIVITY["morgan"] >= PROFILE_RECEPTIVITY["jordan"]
    assert PROFILE_RECEPTIVITY["alex"] >= PROFILE_RECEPTIVITY["jordan"]


def test_compute_receptivity_weights():
    high = compute_receptivity(0.8, 0.9, 0.1)
    low = compute_receptivity(0.2, 0.2, 0.9)
    assert high > low
    assert RHO_MIN <= low <= RHO_MAX
    assert RHO_MIN <= high <= RHO_MAX


def test_classify_direct_answer():
    assert (
        classify_pedagogical_move("The answer is 3/8.") == "direct_answer"
    )
    assert classify_pedagogical_move("It's 5/8 left.") == "direct_answer"


def test_classify_scaffold():
    move = classify_pedagogical_move(
        "What if you draw the whole pizza and shade the parts Maria ate?"
    )
    assert move == "scaffold"


def test_direct_answer_never_raises_mastery():
    group = _group({"part_whole": 1})
    before = {pid: dict(group.students[pid].mastery_state) for pid in group.profile_ids}
    result = apply_scaffold_opportunity(
        group, "The answer is 3/8.", source="teacher", rng=random.Random(0)
    )
    assert result.teaching_warning is True
    assert result.gains == []
    assert result.move == "direct_answer"
    for pid in group.profile_ids:
        assert group.students[pid].mastery_state == before[pid]


def test_scaffold_can_raise_listener_mastery_with_forced_rng():
    group = _group({"part_whole": 1})

    class Always(random.Random):
        def random(self):
            return 0.0  # always < ρ

    result = apply_scaffold_opportunity(
        group,
        "What if you compare the pieces that are left after she shares?",
        source="teacher",
        rng=Always(),
    )
    assert result.teaching_warning is False
    assert result.move == "scaffold"
    assert result.applied is True
    assert len(result.gains) == 3
    for gain in result.gains:
        assert gain.to_level == gain.from_level + 1
        assert group.students[gain.profile_id].mastery_state["part_whole"] == 2


def test_peer_explanation_listener_only():
    group = _group({"part_whole": 1})

    class Always(random.Random):
        def random(self):
            return 0.0

    result = apply_scaffold_opportunity(
        group,
        "Because the pieces have to be equal parts of the whole pizza first.",
        speaker_id="jordan",
        source="peer",
        rng=Always(),
    )
    assert result.move == "scaffold"
    gainer_ids = {g.profile_id for g in result.gains}
    assert "jordan" not in gainer_ids
    assert gainer_ids <= {"sam", "alex"}
    assert group.students["jordan"].mastery_state["part_whole"] == 1
