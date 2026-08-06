import random

from openai import OpenAI

from app.config import MODEL, TEMPERATURE, TAMU_CHAT_API_KEY, TAMU_CHAT_BASE_URL
from app.llm.roles import LLMRole, apply_thinking_constraints, resolve_profile

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=TAMU_CHAT_API_KEY, base_url=TAMU_CHAT_BASE_URL)
    return _client


def _check_response(resp):
    if resp.choices:
        return
    err = getattr(resp, "error", None)
    if err:
        msg = err.get("message", err) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"TAMU Chat API error: {msg}")
    raise RuntimeError("TAMU Chat API returned no choices")


def tamu_chat(**kwargs):
    """Low-level TAMU Chat API call (stream=False, random seed)."""
    resp = get_client().chat.completions.create(
        stream=False,
        seed=random.randint(0, 1_000_000),
        **kwargs,
    )
    _check_response(resp)
    return resp


def complete(role: LLMRole, system: str, user: str, **overrides) -> str:
    """Single system+user completion for judges, classifiers, and taggers."""
    params = resolve_profile(role, **overrides)
    json_mode = params.pop("json_mode")
    kwargs = dict(
        model=params.pop("model"),
        temperature=params.pop("temperature"),
        max_tokens=params.pop("max_tokens"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **params,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return tamu_chat(**kwargs).choices[0].message.content.strip()


def complete_chat(role: LLMRole, system: str, history: list, **overrides) -> str:
    """Multi-turn completion (student reply generation)."""
    params = resolve_profile(role, **overrides)
    params.pop("json_mode", None)
    msgs = [{"role": "system", "content": system}] + history
    return tamu_chat(
        model=params.pop("model"),
        messages=msgs,
        temperature=params.pop("temperature"),
        max_tokens=params.pop("max_tokens"),
        **params,
    ).choices[0].message.content.strip()


def llm(system, user, model=None, temperature=None, max_tokens=400, n=1, json_mode=False):
    """Generic aux LLM helper (prefer complete() with an explicit role)."""
    model = model or MODEL
    temperature = TEMPERATURE if temperature is None else temperature
    temperature, max_tokens = apply_thinking_constraints(model, temperature, max_tokens)
    kwargs = dict(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = tamu_chat(**kwargs)
    return [c.message.content.strip() for c in resp.choices]


def llm1(system, user, **kw):
    return llm(system, user, n=1, **kw)[0]


def chat_completion(system: str, history: list, temperature: float | None = None, max_tokens: int = 200) -> str:
    """Student reply generation via centralized profile."""
    overrides = {"max_tokens": max_tokens}
    if temperature is not None:
        overrides["temperature"] = temperature
    return complete_chat(LLMRole.STUDENT_REPLY, system, history, **overrides)
