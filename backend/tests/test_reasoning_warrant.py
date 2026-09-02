"""Tests for teacher-triggered LP reasoning warrants."""

from unittest.mock import patch

from app.models import Student
from app.reasoning_warrant import (
    WarrantExpectation,
    detect_teacher_reasoning_press,
    has_warrant_signal,
    is_over_warrant,
    warrant_expectation,
)
from app.response_refinement import (
    check_discourse_progress,
    check_reasoning_warrant,
    critique_student_reply,
)

TASK = "phone_plans_linear_01"


def _maya() -> Student:
    return Student(
        student_id="Maya",
        Openness="Low",
        Conscientiousness="High",
        Extraversion="Low",
        Agreeableness="High",
        Neuroticism="High",
        construct_mastery={"rate_comparison": 2},
    )


def _jordan() -> Student:
    return Student(
        student_id="Jordan",
        Openness="Medium",
        Conscientiousness="High",
        Extraversion="High",
        Agreeableness="Low",
        Neuroticism="Low",
        construct_mastery={"rate_comparison": 2},
    )


# --- Detector ---


def test_detect_reasoning_press_pst_script_positive():
    assert detect_teacher_reasoning_press(
        "canyon guys explain your reasoning for the claims",
        task_id=TASK,
    )
    assert detect_teacher_reasoning_press(
        "Jordan what's your reasoning? maya can you check him?",
        addressed_profile="jordan",
        task_id=TASK,
    )
    assert detect_teacher_reasoning_press(
        "Jordan, okaybutwhydo you think 20 won't matter",
        addressed_profile="jordan",
        task_id=TASK,
    )
    assert detect_teacher_reasoning_press(
        "okay can you explain maya why that would be thecase",
        addressed_profile="maya",
        task_id=TASK,
    )


def test_detect_reasoning_press_excludes_comprehension_and_critique():
    assert not detect_teacher_reasoning_press(
        "Jordan did you getit",
        addressed_profile="jordan",
        task_id=TASK,
    )
    assert not detect_teacher_reasoning_press(
        "Just say it",
        addressed_profile="jordan",
        task_id=TASK,
    )
    assert not detect_teacher_reasoning_press(
        "yes but maya find what are the wrong things in Jordan's claim",
        addressed_profile="maya",
        task_id=TASK,
    )
    assert not detect_teacher_reasoning_press(
        "maya help Jordan understand why heis wrong using the table",
        addressed_profile="maya",
        task_id=TASK,
    )


def test_detect_reasoning_press_wrong_student_not_addressed():
    assert not detect_teacher_reasoning_press(
        "Jordan what's your reasoning?",
        addressed_profile="maya",
        task_id=TASK,
    )


def test_detect_reasoning_press_non_phone_plans():
    assert not detect_teacher_reasoning_press(
        "explain your reasoning",
        task_id="frac_word_maria_pizza_01",
    )


# --- Expectation ---


def test_warrant_expectation_jordan_wrong_assertive_rate():
    exp = warrant_expectation(
        profile_id="jordan",
        behavior_profile={
            "behavior_mode": "WRONG",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
        },
        student=_jordan(),
        misconception={"id": "pr_rate_always_wins"},
    )
    assert exp.tone == "assertive"
    assert exp.tier in ("partial", "minimal")
    assert "rate" in exp.rewrite_tone_hint.lower() or "0.10" in exp.rewrite_tone_hint


def test_warrant_expectation_jordan_after_fee_press():
    exp = warrant_expectation(
        profile_id="jordan",
        behavior_profile={
            "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
        },
        student=_jordan(),
        fee_press_count=2,
    )
    assert "$20" in exp.rewrite_tone_hint or "fee" in exp.rewrite_tone_hint.lower()


def test_warrant_expectation_maya_hedged_table():
    exp = warrant_expectation(
        profile_id="maya",
        behavior_profile={
            "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
        },
        student=_maya(),
        misconception={"id": "pr_table_missing_warrants"},
    )
    assert exp.allow_hedge_pass is True
    assert exp.tone in ("hedged", "brief")


# --- Signal / gate ---


def test_has_warrant_signal_jordan_rate():
    exp = warrant_expectation(
        profile_id="jordan",
        behavior_profile={"behavior_mode": "WRONG", "student_stack_level": 2},
        student=_jordan(),
    )
    assert has_warrant_signal(
        "Plan A is cheaper because 0.10 per text is less than 0.30.",
        exp,
    )
    assert not has_warrant_signal("Plan A is always better.", exp)


def test_has_warrant_signal_maya_table_hedge():
    exp = warrant_expectation(
        profile_id="maya",
        behavior_profile={"behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK", "student_stack_level": 2},
        student=_maya(),
        misconception={"id": "pr_table_missing_warrants"},
    )
    assert has_warrant_signal(
        "On my table at 50 texts Plan B is $15 vs $25 — I'm not sure why though.",
        exp,
    )


def test_is_over_warrant_blocks_crossover_before_unlock():
    exp = warrant_expectation(
        profile_id="maya",
        behavior_profile={"behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK", "student_stack_level": 2},
        student=_maya(),
        fight_phase="fight",
        crossover_unlocked=False,
    )
    assert is_over_warrant(
        "Set them equal and x = 100 so Plan B is cheaper below 100.",
        exp,
        crossover_unlocked=False,
        fight_phase="fight",
    )


def test_check_reasoning_warrant_bare_claim_fails():
    exp = warrant_expectation(
        profile_id="jordan",
        behavior_profile={"behavior_mode": "WRONG", "student_stack_level": 2},
        student=_jordan(),
    )
    fail = check_reasoning_warrant(
        "Plan A is better, that's my answer.",
        turn_role="reasoning_press",
        behavior_profile={"_warrant_expectation": exp},
    )
    assert fail is not None
    assert "bare_claim" in fail["issues"]


def test_check_reasoning_warrant_rate_passes():
    exp = warrant_expectation(
        profile_id="jordan",
        behavior_profile={"behavior_mode": "WRONG", "student_stack_level": 2},
        student=_jordan(),
    )
    assert check_reasoning_warrant(
        "Because 0.10 per text beats 0.30 — Plan A wins.",
        turn_role="reasoning_press",
        behavior_profile={"_warrant_expectation": exp},
    ) is None


def test_discourse_skipped_on_reasoning_press():
    assert check_discourse_progress(
        "Plan A is better.",
        turn_role="reasoning_press",
        is_peer_turn=False,
        prior_peer_name="Jordan",
        prior_peer_text="0.10 is cheaper",
    ) is None


@patch("app.response_refinement.complete_chat")
def test_critique_pipeline_reasoning_warrant_before_discourse(mock_chat):
    exp = warrant_expectation(
        profile_id="jordan",
        behavior_profile={
            "behavior_mode": "WRONG",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
            "profile_id": "jordan",
        },
        student=_jordan(),
    )
    result = critique_student_reply(
        "Plan A is better.",
        {
            "behavior_mode": "WRONG",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
            "profile_id": "jordan",
            "_warrant_expectation": exp,
        },
        student=_jordan(),
        turn_role="reasoning_press",
    )
    assert result["ok"] is False
    assert "bare_claim" in result["issues"]
    mock_chat.assert_not_called()
