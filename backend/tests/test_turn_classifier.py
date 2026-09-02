"""Turn classifier + stall coupling: prefer scaffold over false vague/stall."""

from app.turn_classifier import (
    classify_teacher_turn,
    discourse_continues_prior_student,
    is_actionable_scaffold,
    is_high_precision_vague,
    semantic_prefers_scaffold,
    strip_addressing_noise,
    _heuristic_classify,
    _has_math_content,
    _is_vague_directive,
)
from app.turn_logic import (
    is_math_press_turn,
    prepare_student_user_message,
    should_student_stall,
    STALL_INJECT,
)


def test_jordan_numbers_followup_is_scaffold_not_vague():
    msg = "Jordan, okay what numbers would you select here?"
    context = [
        {
            "role": "student",
            "content": (
                "Okay so first step is probably just figuring out what each plan costs — "
                "like pick some number of texts and see which one's cheaper."
            ),
        }
    ]
    assert is_actionable_scaffold(msg, context=context)
    assert classify_teacher_turn(msg, context=context) == "math_scaffold"
    assert _heuristic_classify(msg, context=context) == "math_scaffold"
    assert not should_student_stall(
        [{"teacher_label": "math_scaffold"}], "vague_directive", msg
    )


def test_stall_inject_only_when_stall_active():
    msg = "Jordan, okay what numbers would you select here?"
    out, warn = prepare_student_user_message(msg, "vague_directive", stall_active=False)
    assert out == msg
    assert STALL_INJECT.strip() not in out

    stalled, _ = prepare_student_user_message("okay", "vague_acknowledgment", stall_active=True)
    assert STALL_INJECT.strip() in stalled


def test_genuine_go_on_still_stalls():
    hist = [{"teacher_label": "math_scaffold"}]
    assert should_student_stall(hist, "vague_directive", "go on")
    assert _heuristic_classify("go on") == "vague_directive"
    assert is_high_precision_vague("go on", "vague_directive")


def test_ack_only_okay_is_vague():
    assert _heuristic_classify("okay") == "vague_acknowledgment"
    assert is_high_precision_vague("okay", "vague_acknowledgment")
    assert should_student_stall([{"teacher_label": "math_scaffold"}], "vague_acknowledgment", "okay")


def test_okay_with_scaffold_ask_is_not_ack_only():
    msg = "okay what numbers would you try?"
    assert not is_high_precision_vague(msg, "vague_acknowledgment")
    assert classify_teacher_turn(msg) == "math_scaffold"


def test_strip_addressing_noise():
    assert strip_addressing_noise("[Addressing Jordan] okay") == "okay"
    assert "what numbers" in strip_addressing_noise("Jordan, okay what numbers?").lower()


def test_discourse_continuation_from_prior_student():
    context = [
        {"role": "student", "content": "I'd just try a few numbers and see which plan wins."}
    ]
    assert discourse_continues_prior_student(
        "Which numbers would you pick?", context
    )


def test_what_do_you_think_is_scaffold_not_vague_directive():
    # Old marker "what do you" must not force vague_directive.
    assert not _is_vague_directive("what do you think Plan A costs at 50 texts?")
    assert classify_teacher_turn("Maya, what do you think?") == "math_scaffold"


def test_heuristic_mode_still_classifies_jordan_numbers():
    import app.turn_classifier as tc

    prev = tc.TURN_CLASSIFY_MODE
    tc.TURN_CLASSIFY_MODE = "heuristic"
    try:
        msg = "Jordan, okay what numbers would you select here?"
        context = [
            {
                "role": "student",
                "content": "I'd just try a few numbers and see which plan wins.",
            }
        ]
        assert tc.classify_teacher_turn(msg, context=context) == "math_scaffold"
    finally:
        tc.TURN_CLASSIFY_MODE = prev


def test_semantic_prefers_scaffold_with_prior_student():
    msg = "okay what numbers would you select here?"
    context = [
        {
            "role": "student",
            "content": "I'd pick some number of texts and compare which plan is cheaper.",
        }
    ]
    # May be True or None depending on embedder availability; must not be False.
    vote = semantic_prefers_scaffold(msg, context)
    assert vote is not False


def test_math_detection_still_works():
    assert _has_math_content("set them equal and solve for x")
    assert _has_math_content("which plan is the better deal")
    assert not _has_math_content("how are you feeling today")


def test_greeting_hard_gates_as_social():
    from app.turn_classifier import (
        _hard_gate_label,
        _heuristic_classify,
        claim_lock_active,
    )

    for msg in (
        "how are you guys",
        "whats up ky childeren",
        "hey kids how are you all doing today",
        "Good morning everyone",
    ):
        assert _hard_gate_label(msg) == "social", msg
        assert _heuristic_classify(msg) == "social", msg
        assert classify_teacher_turn(msg, context=[]) == "social", msg
        assert not claim_lock_active("social")

    assert claim_lock_active("math_scaffold")
    assert not claim_lock_active("math_scaffold", stall_active=True)
    assert not claim_lock_active("off_topic")
    # Mixed social + math must not hard-gate as pure social.
    mixed = _hard_gate_label("how are you — which plan is cheaper?")
    assert mixed != "social"


def test_math_press_turns_do_not_stall():
    history = [{"teacher_label": "vague_acknowledgment"}]
    for msg in (
        "Jordan did you get it?",
        "okay but why do you think 20 won't matter",
        "Maya find what are the wrong things in Jordan's claim",
        "Did you understand what Maya pointed out?",
    ):
        assert is_math_press_turn(msg)
        assert not should_student_stall(history, "vague_acknowledgment", msg)


def test_facilitation_script_lines_hard_gate_math_scaffold():
    from unittest.mock import patch

    from app.turn_classifier import _hard_gate_label

    lines = (
        "canyon guys explain your reasoning for the claims",
        "yes but maya find what are the wrong things in Jordan's claim",
        "Jordan, okaybutwhydo you think 20 won't matter",
        "socan you answer the question then? when is each plan better. maya?",
        "what's the answer both of you say",
    )
    for msg in lines:
        assert _hard_gate_label(msg) == "math_scaffold", msg

    with patch("app.turn_classifier.complete", side_effect=RuntimeError("no llm")):
        for msg in lines:
            assert classify_teacher_turn(msg) == "math_scaffold", msg
