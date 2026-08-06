"""TAMU Chat client and per-role LLM profiles."""

from app.llm.client import (
    chat_completion,
    complete,
    complete_chat,
    get_client,
    llm,
    llm1,
    tamu_chat,
)
from app.llm.roles import (
    LLMRole,
    apply_thinking_constraints,
    model_for_role,
    resolve_profile,
)

__all__ = [
    "LLMRole",
    "apply_thinking_constraints",
    "chat_completion",
    "complete",
    "complete_chat",
    "get_client",
    "llm",
    "llm1",
    "model_for_role",
    "resolve_profile",
    "tamu_chat",
]
