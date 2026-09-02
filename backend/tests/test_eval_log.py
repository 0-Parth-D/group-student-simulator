"""Tests for eval_log generation_debug helpers."""

from app.eval_log import (
    build_eval_turn_record,
    build_generation_debug,
    learning_expected_behavior_text,
)
from app.models import Student


def _maya() -> Student:
    return Student(
        student_id="Maya",
        Openness="Low",
        Conscientiousness="High",
        Extraversion="Low",
        Agreeableness="High",
        Neuroticism="High",
        construct_mastery={"linear_relationship": 2},
    )


def _profile(**overrides):
    base = {
        "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
        "likely_correctness": "partially correct",
        "student_stack_level": 2,
        "target_stack_level": 3,
        "primary_construct": "linear_relationship",
        "help_seek_style": "anxious_helpless",
        "stall_active": False,
        "help_seeking": False,
    }
    base.update(overrides)
    return base


def test_build_generation_debug_shape_and_history_cap():
    hist = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    targets = {
        "personality": "Low E",
        "learning": "Stay partial",
        "misconception": "table cue",
        "mode": "PARTIAL_ATTEMPT_THEN_STUCK",
        "stack": "2 / target 3",
    }
    debug = build_generation_debug(
        system_prompt="You are Maya.",
        history=hist,
        turn_mode="math_scaffold",
        is_peer_turn=False,
        behavior_profile=_profile(),
        targets=targets,
        personality_export={"Extraversion": "Low"},
        misconception={
            "id": "pr_compare_one_x_only",
            "prompt_cue": "few numbers",
            "description": "desc",
        },
        draft="draft reply",
        final_reply="final reply",
        revisions=1,
        critic={"ok": False, "issues": ["tutor_tone"]},
        hist_max=5,
    )
    assert debug["system_prompt"] == "You are Maya."
    assert len(debug["history"]) == 5
    assert debug["history"][0]["content"] == "m15"
    assert debug["targets"]["learning"] == "Stay partial"
    assert debug["turn_state"]["behavior_mode"] == "PARTIAL_ATTEMPT_THEN_STUCK"
    assert debug["misconception"]["id"] == "pr_compare_one_x_only"
    assert debug["draft"] == "draft reply"
    assert debug["final_reply"] == "final reply"
    assert debug["revisions"] == 1
    assert debug["critic"]["issues"] == ["tutor_tone"]
    assert "refine_mode" in debug
    assert debug["prompt_chars"] == len("You are Maya.")
    debug_r = build_generation_debug(
        system_prompt="x",
        history=[],
        turn_mode="math_scaffold",
        is_peer_turn=False,
        behavior_profile=_profile(),
        targets=targets,
        retrieval={"active_misc_id": "pr_table_missing_warrants"},
    )
    assert debug_r["retrieval"]["active_misc_id"] == "pr_table_missing_warrants"


def test_build_eval_turn_record_includes_generation_debug():
    debug = build_generation_debug(
        system_prompt="sys",
        history=[{"role": "user", "content": "hi"}],
        turn_mode="math_scaffold",
        is_peer_turn=False,
        behavior_profile=_profile(),
        targets={"learning": "Stay partial"},
        draft="d",
        final_reply="f",
        revisions=0,
    )
    record = build_eval_turn_record(
        turn=1,
        teacher_message="Try again",
        reply="f",
        teacher_label="math_scaffold",
        stall_active=False,
        profile_id="maya",
        task_id="phone_plans_linear_01",
        behavior_profile=_profile(),
        task_metadata={"task_id": "phone_plans_linear_01"},
        expected_behavior="Stay partial",
        generation_debug=debug,
    )
    assert record["expected_behavior"] == "Stay partial"
    assert record["generation_debug"]["system_prompt"] == "sys"
    assert record["generation_debug"]["targets"]["learning"] == "Stay partial"


def test_build_eval_turn_record_omits_debug_when_none():
    record = build_eval_turn_record(
        turn=1,
        teacher_message="hi",
        reply="ok",
        teacher_label="math_scaffold",
        stall_active=False,
        profile_id="maya",
        task_id="",
        behavior_profile=_profile(),
        task_metadata={},
        expected_behavior="Stay partial",
    )
    assert "generation_debug" not in record
    assert record["expected_behavior"] == "Stay partial"


def test_build_generation_debug_includes_llm_params():
    llm_params = {
        "model": "protected.Claude Sonnet 4.6",
        "temperature": 1.0,
        "max_tokens": 384,
        "requested_max_tokens": 200,
        "thinking_model": True,
        "seed_mode": "random",
    }
    debug = build_generation_debug(
        system_prompt="sys",
        history=[{"role": "user", "content": "hi"}],
        turn_mode="math_scaffold",
        is_peer_turn=False,
        behavior_profile=_profile(),
        targets={"learning": "Stay partial"},
        draft="d",
        final_reply="f",
        revisions=0,
        llm_params=llm_params,
    )
    assert debug["llm_params"]["max_tokens"] == 384
    assert debug["llm_params"]["requested_max_tokens"] == 200
    assert debug["llm_params"]["thinking_model"] is True


def test_learning_expected_behavior_text_nonempty():
    text = learning_expected_behavior_text(
        _profile(),
        misconception={"id": "x", "prompt_cue": "pick a few numbers"},
        turn_mode="math_scaffold",
        student=_maya(),
    )
    assert isinstance(text, str)
    assert text.strip()
