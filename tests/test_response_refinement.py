"""Unit tests for response-refinement heuristics (no LLM calls)."""

from app.student.response_refinement import (
    is_advancing_on_stall,
    is_meta_process_question,
    is_too_anxious,
    is_too_expert,
    missing_error_enactment,
)


def _wrong_profile(**overrides):
    base = {
        "behavior_mode": "WRONG",
        "likely_correctness": "incorrect",
        "student_stack_level": 1,
        "target_stack_level": 3,
        "error_types": ["adds rates instead of comparing"],
        "error_labels": ["adds rates instead of comparing"],
        "wrong_answers": ["just add 20 + 0.1"],
        "avoid_answers": ["100", "x = 100"],
        "stall_active": False,
        "help_seeking": False,
    }
    base.update(overrides)
    return base


def _partial_profile(**overrides):
    base = {
        "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
        "likely_correctness": "partially correct",
        "student_stack_level": 2,
        "target_stack_level": 3,
        "error_types": ["stops after writing one equation"],
        "avoid_answers": ["100", "x = 100"],
        "stall_active": False,
        "help_seeking": False,
    }
    base.update(overrides)
    return base


def test_meta_process_banned_on_active_math():
    profile = _wrong_profile()
    assert is_meta_process_question(
        "Okay — what do you want me to do with it?", profile
    )
    assert is_meta_process_question(
        "Do you want me to show the actual math or explain it like in words?",
        profile,
    )


def test_meta_process_allowed_on_stall():
    profile = _wrong_profile(stall_active=True)
    assert not is_meta_process_question(
        "Okay — what do you want me to do next?", profile
    )


def test_mastery_gate_blocks_full_solution_leap():
    profile = _partial_profile()
    assert is_too_expert(
        "Just set them equal: 20 + 0.1x = 0.3x, so x = 100.",
        profile,
    )


def test_mastery_gate_allows_when_at_target():
    profile = _partial_profile(
        student_stack_level=3,
        target_stack_level=3,
        likely_correctness="mostly correct",
    )
    assert not is_too_expert(
        "Set them equal and get x = 100.",
        profile,
    )


def test_wrong_mode_anxious_hedge_flagged():
    profile = _wrong_profile()
    assert is_too_anxious(
        "I'm not sure... what do you think? Maybe 50?",
        profile,
    )


def test_wrong_confident_mistake_ok():
    profile = _wrong_profile()
    assert not is_too_anxious(
        "Easy! Just add 20 + 0.1 and you're done.",
        profile,
    )


def test_missing_error_enactment_on_empty_hedge():
    profile = _wrong_profile()
    assert missing_error_enactment(
        "Hmm I'm kinda confused about this one.",
        profile,
    )


def test_error_enactment_present():
    profile = _wrong_profile()
    assert not missing_error_enactment(
        "Easy! Just adds rates instead of comparing — so 20 + 0.1.",
        profile,
    )


def test_stall_advancing_flagged():
    profile = _wrong_profile(stall_active=True)
    assert is_advancing_on_stall(
        "So then 20 + 0.1x = 30, which equals the answer.",
        profile,
    )


def test_stall_direction_question_ok():
    profile = _wrong_profile(stall_active=True)
    assert not is_advancing_on_stall(
        "Okay — what should I do next?",
        profile,
    )
