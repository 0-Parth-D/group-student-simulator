"""Tests for student sampling / thinking-model constraints (Part C)."""

from app.config import (
    CLAUDE_THINKING_MODELS,
    STUDENT_MAX_TOKENS_MATH,
    STUDENT_MAX_TOKENS_SOCIAL,
    STUDENT_THINKING_MIN_MAX_TOKENS,
)
from app.llm_config import (
    THINKING_MIN_MAX_TOKENS,
    LLMRole,
    apply_thinking_constraints,
    resolve_profile,
    resolved_student_chat_params,
    student_max_tokens_for_turn,
)


def test_student_max_tokens_for_turn():
    assert student_max_tokens_for_turn("math_scaffold") == STUDENT_MAX_TOKENS_MATH
    assert student_max_tokens_for_turn("math_eval") == STUDENT_MAX_TOKENS_MATH
    assert student_max_tokens_for_turn("social") == STUDENT_MAX_TOKENS_SOCIAL
    assert student_max_tokens_for_turn("off_topic") == STUDENT_MAX_TOKENS_SOCIAL


def test_student_reply_thinking_floor_not_2000():
    model = next(iter(CLAUDE_THINKING_MODELS))
    temp, tokens = apply_thinking_constraints(
        model, 0.85, 200, role=LLMRole.STUDENT_REPLY
    )
    assert temp == 1.0
    assert tokens == max(200, STUDENT_THINKING_MIN_MAX_TOKENS)
    assert tokens < THINKING_MIN_MAX_TOKENS
    assert tokens != 2000 or STUDENT_THINKING_MIN_MAX_TOKENS >= 2000


def test_non_student_thinking_floor_still_2000():
    model = next(iter(CLAUDE_THINKING_MODELS))
    temp, tokens = apply_thinking_constraints(
        model, 0.0, 120, role=LLMRole.REPLY_CRITIC
    )
    assert temp == 1.0
    assert tokens == max(120, THINKING_MIN_MAX_TOKENS)


def test_resolve_profile_student_uses_student_floor():
    model = next(iter(CLAUDE_THINKING_MODELS))
    params = resolve_profile(
        LLMRole.STUDENT_REPLY, model=model, max_tokens=200
    )
    assert params["temperature"] == 1.0
    assert params["max_tokens"] == max(200, STUDENT_THINKING_MIN_MAX_TOKENS)


def test_resolved_student_chat_params_shape():
    snap = resolved_student_chat_params(200)
    assert snap["requested_max_tokens"] == 200
    assert "model" in snap
    assert "temperature" in snap
    assert "max_tokens" in snap
    assert snap["seed_mode"] in ("random", "fixed")
    assert isinstance(snap["thinking_model"], bool)
    if snap["thinking_model"]:
        assert snap["max_tokens"] == max(200, STUDENT_THINKING_MIN_MAX_TOKENS)
        assert snap["temperature"] == 1.0
