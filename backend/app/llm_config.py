"""Central LLM call profiles — models, temperatures, and token limits per role."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.config import (
    CLAUDE_THINKING_MODELS,
    MODEL,
    STUDENT_LLM_SEED,
    STUDENT_MAX_TOKENS_MATH,
    STUDENT_MAX_TOKENS_SOCIAL,
    STUDENT_OPENAI_MODEL,
    STUDENT_THINKING_MIN_MAX_TOKENS,
    TEMPERATURE,
    TURN_CLASSIFIER_MODEL,
)

THINKING_MIN_MAX_TOKENS = int(os.getenv("THINKING_MIN_MAX_TOKENS", "2000"))
JUDGE_TEMPERATURE = float(os.getenv("LLM_JUDGE_TEMPERATURE", str(TEMPERATURE)))
STUDENT_TEMPERATURE = float(
    os.getenv("LLM_STUDENT_TEMPERATURE", os.getenv("REFINE_TEMPERATURE", "0.85"))
)


class LLMRole(str, Enum):
    STUDENT_REPLY = "student_reply"
    TURN_CLASSIFIER = "turn_classifier"
    TASK_TAGGER = "task_tagger"
    KG_GRADING = "kg_grading"
    GROUP_CONSTRAINT = "group_constraint"
    EVAL_BEHAVIORAL = "eval_behavioral"
    EVAL_COGNITIVE = "eval_cognitive"
    EVAL_PERSONA = "eval_persona"
    EVAL_GROUP_DISCOURSE = "eval_group_discourse"
    EVAL_BELIEVABILITY = "eval_believability"
    GAME_FLOW_ORCHESTRATOR = "game_flow_orchestrator"
    REPLY_CRITIC = "reply_critic"


@dataclass(frozen=True)
class LLMProfile:
    temperature: float
    max_tokens: int
    json_mode: bool = False


PROFILES: dict[LLMRole, LLMProfile] = {
    LLMRole.STUDENT_REPLY: LLMProfile(
        temperature=STUDENT_TEMPERATURE, max_tokens=STUDENT_MAX_TOKENS_MATH
    ),
    LLMRole.TURN_CLASSIFIER: LLMProfile(temperature=0.0, max_tokens=160, json_mode=True),
    LLMRole.TASK_TAGGER: LLMProfile(temperature=JUDGE_TEMPERATURE, max_tokens=500, json_mode=True),
    LLMRole.KG_GRADING: LLMProfile(temperature=JUDGE_TEMPERATURE, max_tokens=80, json_mode=True),
    LLMRole.GROUP_CONSTRAINT: LLMProfile(temperature=0.0, max_tokens=250, json_mode=True),
    LLMRole.EVAL_BEHAVIORAL: LLMProfile(temperature=JUDGE_TEMPERATURE, max_tokens=120, json_mode=True),
    LLMRole.EVAL_COGNITIVE: LLMProfile(temperature=JUDGE_TEMPERATURE, max_tokens=120, json_mode=True),
    LLMRole.EVAL_PERSONA: LLMProfile(temperature=JUDGE_TEMPERATURE, max_tokens=200, json_mode=True),
    LLMRole.EVAL_GROUP_DISCOURSE: LLMProfile(
        temperature=JUDGE_TEMPERATURE, max_tokens=180, json_mode=True
    ),
    LLMRole.EVAL_BELIEVABILITY: LLMProfile(
        temperature=JUDGE_TEMPERATURE, max_tokens=120, json_mode=True
    ),
    LLMRole.GAME_FLOW_ORCHESTRATOR: LLMProfile(
        temperature=0.3, max_tokens=500, json_mode=True
    ),
    # Personality-primary + LP + misconception gate for student drafts
    LLMRole.REPLY_CRITIC: LLMProfile(
        temperature=JUDGE_TEMPERATURE, max_tokens=280, json_mode=True
    ),
}


def student_max_tokens_for_turn(turn_mode: str) -> int:
    """Centralized student output cap by turn type."""
    if turn_mode in ("social", "off_topic"):
        return STUDENT_MAX_TOKENS_SOCIAL
    return STUDENT_MAX_TOKENS_MATH


def model_for_role(role: LLMRole, model: str | None = None) -> str:
    if model:
        return model
    if role == LLMRole.STUDENT_REPLY:
        return STUDENT_OPENAI_MODEL
    if role == LLMRole.TURN_CLASSIFIER:
        return TURN_CLASSIFIER_MODEL
    return MODEL


def apply_thinking_constraints(
    model: str,
    temperature: float,
    max_tokens: int,
    role: Optional[LLMRole] = None,
) -> tuple[float, int]:
    """Thinking models: temperature is always 1.0.

    ``max_tokens`` must be greater than the model's thinking budget. Student
    replies use ``STUDENT_THINKING_MIN_MAX_TOKENS`` (set >= 2000 for Claude).
    Judges use ``THINKING_MIN_MAX_TOKENS`` (default 2000).
    """
    if model not in CLAUDE_THINKING_MODELS:
        return temperature, max_tokens
    if role == LLMRole.STUDENT_REPLY:
        return 1.0, max(max_tokens, STUDENT_THINKING_MIN_MAX_TOKENS)
    return 1.0, max(max_tokens, THINKING_MIN_MAX_TOKENS)


def resolve_profile(role: LLMRole, **overrides) -> dict:
    """Merge role defaults, overrides, and thinking-model constraints."""
    profile = PROFILES[role]
    model = model_for_role(role, overrides.pop("model", None))
    temperature = overrides.pop("temperature", profile.temperature)
    max_tokens = overrides.pop("max_tokens", profile.max_tokens)
    json_mode = overrides.pop("json_mode", profile.json_mode)
    temperature, max_tokens = apply_thinking_constraints(
        model, temperature, max_tokens, role=role
    )
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        **overrides,
    }


def resolved_student_chat_params(
    max_tokens: int,
    **overrides,
) -> dict:
    """Snapshot of resolved STUDENT_REPLY params for generation_debug / tests."""
    requested = int(max_tokens)
    params = resolve_profile(LLMRole.STUDENT_REPLY, max_tokens=requested, **overrides)
    model = params["model"]
    return {
        "model": model,
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "requested_max_tokens": requested,
        "thinking_model": model in CLAUDE_THINKING_MODELS,
        "seed_mode": "fixed" if STUDENT_LLM_SEED else "random",
    }
