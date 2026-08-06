"""Unit tests for probabilistic help-seeking decisions."""

import random

from app.student.help_seeking import (
    MIN_TURNS_BETWEEN_HELP_SEEK,
    build_help_seeking_prompt_block,
    decide_help_seeking,
    help_seeking_probability,
    reply_looks_like_question,
    turns_since_help_seek,
)
from app.student.models import Student


def _student(**kwargs) -> Student:
    defaults = dict(
        student_id="Sam",
        Openness="Medium",
        Conscientiousness="Medium",
        Extraversion="Medium",
        Agreeableness="High",
        Neuroticism="High",
    )
    defaults.update(kwargs)
    return Student(**defaults)


def test_confused_anxious_more_likely_than_mastered_calm():
    confused = _student(Neuroticism="High", Extraversion="High")
    calm = _student(Neuroticism="Low", Extraversion="Low", Agreeableness="Low")
    p_confused, _ = help_seeking_probability(
        confused,
        {
            "behavior_mode": "CONFUSED_HELPSEEKING",
            "student_stack_level": 0,
            "help_seek_style": "talkative_anxious",
            "prerequisite_gaps": [{"gap_description": "common denominators"}],
        },
    )
    p_calm, _ = help_seeking_probability(
        calm,
        {
            "behavior_mode": "NORMAL_ERROR_PROFILE",
            "student_stack_level": 3,
            "help_seek_style": "passive_minimal",
        },
    )
    assert p_confused > p_calm
    assert p_confused >= 0.35
    assert p_calm < 0.15


def test_wrong_confident_rarely_help_seeks():
    jordan = _student(Extraversion="High", Neuroticism="Low")
    p, reason = help_seeking_probability(
        jordan,
        {
            "behavior_mode": "WRONG",
            "student_stack_level": 1,
            "help_seek_style": "talkative_confident",
        },
    )
    assert p <= 0.05
    assert "rarely" in reason


def test_social_and_stall_block_help_seeking():
    student = _student()
    bp = {"behavior_mode": "CONFUSED_HELPSEEKING", "student_stack_level": 0}
    p_social, _ = help_seeking_probability(
        student, bp, turn_mode="social"
    )
    p_stall, _ = help_seeking_probability(
        student, bp, stall_active=True
    )
    assert p_social == 0.0
    assert p_stall == 0.0


def test_cooldown_reduces_probability():
    student = _student(Neuroticism="High")
    bp = {
        "behavior_mode": "CONFUSED_HELPSEEKING",
        "student_stack_level": 0,
        "help_seek_style": "anxious_helpless",
    }
    p_fresh, _ = help_seeking_probability(student, bp, turns_since_last=None)
    p_cool, _ = help_seeking_probability(
        student, bp, turns_since_last=MIN_TURNS_BETWEEN_HELP_SEEK - 1
    )
    assert p_cool < p_fresh * 0.5


def test_decide_respects_rng_and_allow_flag():
    student = _student(Neuroticism="High")
    bp = {
        "behavior_mode": "CONFUSED_HELPSEEKING",
        "student_stack_level": 0,
        "help_seek_style": "anxious_helpless",
        "prerequisite_gaps": [{"gap_description": "unit fraction"}],
    }
    # Force trigger: roll 0.0 always below any positive p
    yes = decide_help_seeking(
        student, bp, rng=random.Random(0)
    )
    # With seeded Random(0), first random() is deterministic; also test allow=False
    blocked = decide_help_seeking(student, bp, allow=False, rng=random.Random(0))
    assert blocked.triggered is False
    assert blocked.reason.startswith("help-seeking not allowed")

    # High-p student with forced low roll via a stub rng
    class AlwaysLow:
        def random(self):
            return 0.0

    class AlwaysHigh:
        def random(self):
            return 0.99

    assert decide_help_seeking(student, bp, rng=AlwaysLow()).triggered is True
    assert decide_help_seeking(student, bp, rng=AlwaysHigh()).triggered is False
    assert yes.probability > 0


def test_turns_since_help_seek():
    history = [
        {"turn": 1, "help_seeking": False},
        {"turn": 2, "help_seeking": True},
        {"turn": 3, "help_seeking": False},
    ]
    assert turns_since_help_seek(history, 5) == 3
    assert turns_since_help_seek([], 1) is None


def test_prompt_block_mentions_question_and_style():
    student = _student()
    block = build_help_seeking_prompt_block(
        student,
        {
            "help_seek_style": "anxious_helpless",
            "prerequisite_gaps": [{"gap_description": "finding a common denominator"}],
            "primary_construct": "equiv_fractions",
        },
        group_context=True,
    )
    assert "HELP-SEEKING" in block
    assert "finding a common denominator" in block
    assert "Ask the teacher" in block
    assert "classmate" in block.lower() or "others" in block.lower()


def test_reply_looks_like_question():
    assert reply_looks_like_question("What does LCD mean?")
    assert reply_looks_like_question("I'm stuck on the first step")
    assert not reply_looks_like_question("I think it's 3/4")
