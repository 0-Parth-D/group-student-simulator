"""Central LLM call profiles — models, temperatures, and token limits per role."""

import os
from dataclasses import dataclass
from enum import Enum

from app.config import CLAUDE_THINKING_MODELS, MODEL, STUDENT_OPENAI_MODEL, TEMPERATURE

THINKING_MIN_MAX_TOKENS = int(os.getenv("THINKING_MIN_MAX_TOKENS", "2000"))
JUDGE_TEMPERATURE = float(os.getenv("LLM_JUDGE_TEMPERATURE", str(TEMPERATURE)))
STUDENT_TEMPERATURE = float(os.getenv("LLM_STUDENT_TEMPERATURE", os.getenv("REFINE_TEMPERATURE", "0.85")))


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


@dataclass(frozen=True)
class LLMProfile:
    temperature: float
    max_tokens: int
    json_mode: bool = False


PROFILES: dict[LLMRole, LLMProfile] = {
    LLMRole.STUDENT_REPLY: LLMProfile(temperature=STUDENT_TEMPERATURE, max_tokens=200),
    LLMRole.TURN_CLASSIFIER: LLMProfile(temperature=JUDGE_TEMPERATURE, max_tokens=120, json_mode=True),
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
}


def model_for_role(role: LLMRole, model: str | None = None) -> str:
    if model:
        return model
    if role == LLMRole.STUDENT_REPLY:
        return STUDENT_OPENAI_MODEL
    return MODEL


def apply_thinking_constraints(model: str, temperature: float, max_tokens: int) -> tuple[float, int]:
    if model in CLAUDE_THINKING_MODELS:
        return 1.0, max(max_tokens, THINKING_MIN_MAX_TOKENS)
    return temperature, max_tokens


def resolve_profile(role: LLMRole, **overrides) -> dict:
    """Merge role defaults, overrides, and thinking-model constraints."""
    profile = PROFILES[role]
    model = model_for_role(role, overrides.pop("model", None))
    temperature = overrides.pop("temperature", profile.temperature)
    max_tokens = overrides.pop("max_tokens", profile.max_tokens)
    json_mode = overrides.pop("json_mode", profile.json_mode)
    temperature, max_tokens = apply_thinking_constraints(model, temperature, max_tokens)
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        **overrides,
    }
