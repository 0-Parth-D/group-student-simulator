"""Tests for layered student prompt assembly (Part B)."""

from app.models import Student
from app.prompts import student_prompt
from app.student_prompt_layers import (
    VOICE_FOOTER,
    build_voice_card,
    critique_move_block,
    peer_move_block,
)


def _maya() -> Student:
    return Student(
        student_id="Maya",
        Openness="Low",
        Conscientiousness="High",
        Extraversion="Low",
        Agreeableness="High",
        Neuroticism="High",
        construct_mastery={"linear_relationship": 2, "unit_rate": 2},
        pisa_profile={"mathematical_reasoning": "med"},
    )


def _partial_profile():
    return {
        "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
        "likely_correctness": "partially correct",
        "student_stack_level": 2,
        "target_stack_level": 3,
        "primary_construct": "linear_relationship",
        "help_seek_style": "anxious_helpless",
    }


def test_voice_card_after_cognitive_state():
    prompt = student_prompt(
        _maya(),
        "Maya tries a table but stalls.",
        ["linear_relationship"],
        task_text="Compare phone plans.",
        behavior_profile=_partial_profile(),
        turn_mode="math_scaffold",
    )
    cog_idx = prompt.index("[COGNITIVE STATE")
    voice_idx = prompt.index("[VOICE — apply to every reply]")
    assert voice_idx > cog_idx
    assert VOICE_FOOTER in prompt
    assert "[KG-Based Cognitive Profile" not in prompt
    assert "[Predicted Behavior for this Task]" not in prompt


def test_prompt_diet_soft_cap_and_single_playbook_pattern():
    from app.config import GROUP_MATH_PROMPT_CHAR_SOFT_CAP

    playbook = {
        "patterns": [
            {
                "concept": "rate_comparison",
                "type": "missing_warrants",
                "example": "table Plan B",
                "prompt_cue": "table cue",
                "stack_level": 2,
            },
            {
                "concept": "unit_rate",
                "type": "other",
                "example": "should not appear",
                "prompt_cue": "skip me",
                "stack_level": 1,
            },
        ]
    }
    prompt = student_prompt(
        _maya(),
        "predicted",
        ["linear_relationship", "rate_comparison"],
        task_text="Phone plans compare A and B.",
        behavior_profile=_partial_profile(),
        error_playbook=playbook,
        turn_mode="math_scaffold",
    )
    assert len(prompt) <= GROUP_MATH_PROMPT_CHAR_SOFT_CAP
    assert "table Plan B" in prompt or "table cue" in prompt
    assert "should not appear" not in prompt
    assert "skip me" not in prompt


def test_no_personality_override_wording():
    prompt = student_prompt(
        _maya(),
        "partial",
        ["linear_relationship"],
        behavior_profile=_partial_profile(),
        turn_mode="math_scaffold",
    )
    assert "overrides conflicting personality" not in prompt.lower()
    assert "Your OCEAN voice (below) sets HOW" in prompt


def test_response_length_hint_once_on_math_turn():
    prompt = student_prompt(
        _maya(),
        "partial",
        ["linear_relationship"],
        behavior_profile=_partial_profile(),
        turn_mode="math_scaffold",
    )
    needle = "Reply in 1 short sentence for math."
    assert prompt.count(needle) == 1


def test_pisa_omitted_when_no_task_weak_attributes():
    student = _maya()
    prompt = student_prompt(
        student,
        "partial",
        ["linear_relationship"],
        behavior_profile=_partial_profile(),
        turn_mode="math_scaffold",
    )
    assert "[PISA Mathematical Competencies]" not in prompt


def test_pisa_included_when_weak_attributes_present():
    student = _maya()
    profile = {
        **_partial_profile(),
        "weak_attributes": ["mathematical_reasoning"],
    }
    prompt = student_prompt(
        student,
        "partial",
        ["linear_relationship"],
        behavior_profile=profile,
        turn_mode="math_scaffold",
    )
    assert "[PISA Mathematical Competencies]" in prompt


def test_maya_partial_voice_mentions_brevity():
    card = build_voice_card(
        _maya(),
        _partial_profile(),
        "math_scaffold",
        profile_id="maya",
    )
    lower = card.lower()
    assert "short" in lower or "brief" in lower or "one" in lower
    assert "maya" in lower or "In group work" in card


def test_peer_move_block_evidence():
    block = peer_move_block("evidence", prev_name="Jordan")
    assert "PEER MOVE" in block
    assert "Jordan" in block
    assert "ONE new detail" in block or "one new detail" in block.lower()


def test_critique_move_block_teacher_direct():
    block = critique_move_block(
        peer_name="Jordan",
        peer_text="Plan A is better because 0.10 is cheaper per text.",
    )
    assert "PEER CRITIQUE" in block
    assert "Jordan" in block
    assert "0.10" in block
    assert "Do NOT restate" in block or "do not restate" in block.lower()
