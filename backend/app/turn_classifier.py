"""Context-aware teacher-turn classification (LLM + heuristic + semantic fallback)."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import List, Literal, Optional, Sequence, Tuple

from app.construct_text import math_cue_phrases
from app.llm import complete
from app.llm_config import LLMRole

try:
    from app.config import TURN_CLASSIFY_MODE
except Exception:  # pragma: no cover
    TURN_CLASSIFY_MODE = "semantic_llm"

logger = logging.getLogger(__name__)

TurnMode = Literal[
    "social",
    "math_scaffold",
    "math_eval",
    "vague_acknowledgment",
    "vague_directive",
    "mixed",
    "off_topic",
]

VALID_LABELS = frozenset(
    {
        "math_eval",
        "math_scaffold",
        "vague_acknowledgment",
        "vague_directive",
        "social",
        "off_topic",
        "mixed",
    }
)

VAGUE_ACK_TOKENS = {
    "yes",
    "ok",
    "okay",
    "correct",
    "right",
    "good",
    "yep",
    "sure",
    "yeah",
    "yup",
    "alright",
    "all right",
}

# Phrase-anchored only — never bare "what do you" (false-positives on real scaffolds).
VAGUE_DIRECTIVE_PATTERNS = (
    re.compile(r"\bgo on\b", re.I),
    re.compile(r"\bkeep going\b", re.I),
    re.compile(r"\bwhat next\b", re.I),
    re.compile(r"\bwhat(?:'s| is) next\b", re.I),
    re.compile(r"\bcontinue\b(?!\s+with\b)", re.I),
    re.compile(r"\btry again\b", re.I),
    re.compile(r"\byour turn\b", re.I),
    re.compile(r"\bgo ahead\b", re.I),
)

# Pedagogical next-step asks — force math_scaffold even without numerals/KG cues.
SCAFFOLD_ACTION_CUES = (
    "what number",
    "which number",
    "what numbers",
    "which numbers",
    "how many",
    "which plan",
    "what does",
    "what do you think",
    "how do you know",
    "why does",
    "why is",
    "why would",
    "can you explain",
    "explain why",
    "show me how",
    "walk me through",
    "try a",
    "try some",
    "pick a",
    "pick some",
    "select",
    "choose",
    "compare",
    "set them equal",
    "make a table",
    "in your table",
    "on the graph",
    "what if you",
    "for example if",
)

ROSTER_NAME_RE = re.compile(
    r"\b(?:alex|maya|jordan|riley)\b",
    re.I,
)
ADDRESSING_TAG_RE = re.compile(
    r"^\[(?:addressing|to)\s+[^\]]+\]\s*",
    re.I,
)
LEADING_NAME_RE = re.compile(
    r"^(?:alex|maya|jordan|riley)\s*[,:\-–—]?\s+",
    re.I,
)

SOCIAL_PATTERNS = (
    "how are you",
    "how're you",
    "hows your day",
    "how's your day",
    "how have you been",
    "how you doing",
    "how yall",
    "how y'all",
    "how are u",
    "how are ya",
    "good morning",
    "good afternoon",
    "hello",
    "hi ",
    "hey ",
    "hey guys",
    "hey kids",
    "what's up",
    "whats up",
    "what up",
    "sup ",
    "wassup",
    "nice to see",
    "how was your",
    "how do you feel",
    "feeling today",
)

# Turns with no math obligation — claim lock / fight restatement must not apply.
NON_MATH_TURN_MODES = frozenset({"social", "off_topic"})

MATH_STRUCTURE = re.compile(
    r"\d+\s*/\s*\d+"
    r"|\d+\s*:\s*\d+"
    r"|\d+\s*[-+*/=]\s*\d+"
    r"|\$\s*\d"
    r"|\bx\s*="
    r"|\\frac"
)

GENERIC_MATH_VERBS = (
    "solve",
    "calculate",
    "simplify",
    "convert",
    "equation",
    "expression",
    "answer",
    "step",
    "problem",
    "number",
    "numbers",
    "texts",
    "plan a",
    "plan b",
    "table",
    "graph",
    "rate",
    "cost",
)

EVAL_PATTERNS = (
    "good job",
    "well done",
    "not quite",
    "almost",
    "that's right",
    "thats right",
    "you got it",
    "close",
    "try again",
    "think about",
    "remember",
)

VAGUE_PROTOTYPES = (
    "okay",
    "yeah",
    "go on",
    "keep going",
    "what next?",
    "continue",
    "your turn",
    "alright go ahead",
)

SCAFFOLD_PROTOTYPES = (
    "what numbers would you try?",
    "which numbers should we pick?",
    "what does 0.1 mean for ten texts?",
    "why is plan B cheaper at first?",
    "can you explain your table?",
    "how do you know the plans are equal?",
    "walk me through that step",
    "try fifty texts and compare the costs",
)

TURN_CLASSIFY_PROMPT = """You tag ONE teacher turn in a middle-school math group discussion.
Be conservative about vague labels — when unsure, prefer math_scaffold.

## Task
{task_text}

## Recent dialogue
{context_window}

## Last student claim (most important discourse cue)
{last_student_claim}

## Teacher's latest message
{teacher_turn}

## Labels (pick exactly one)
- math_scaffold: hint, sub-question, asks for numbers/reason/method/representation, or continues the student's last idea
- math_eval: judges correctness (correct / not quite / close)
- vague_acknowledgment: ack ONLY with no new ask ("okay", "yeah", "good" alone)
- vague_directive: empty continue ONLY ("go on", "keep going", "what next") with NO new math target
- social: personal/off-math chat
- off_topic: unrelated non-social
- mixed: social + math in one message

## Decision rules
1. If the teacher asks for a concrete action (pick numbers, explain why, compare plans, try a value) → math_scaffold.
2. If the last student proposed an approach and the teacher asks a follow-up slot about it → math_scaffold.
3. Mid-sentence "okay, …" with a real question is NOT vague_acknowledgment.
4. Vague labels require zero new mathematical target.

Return JSON only:
{{"label":"math_scaffold","confidence":0.0,"has_concrete_ask":true,"continues_prior_student":false,"reason":"..."}}"""

TURN_CLASSIFY_SEMANTIC = os.getenv("TURN_CLASSIFY_SEMANTIC", "true").lower() in (
    "1",
    "true",
    "yes",
)
TURN_CLASSIFY_SEMANTIC_MARGIN = float(
    os.getenv("TURN_CLASSIFY_SEMANTIC_MARGIN", "0.05")
)
LLM_VAGUE_MIN_CONFIDENCE = float(os.getenv("LLM_VAGUE_MIN_CONFIDENCE", "0.70"))

_prototype_cache: dict[str, list[float]] = {}


def format_context_window(context: Optional[List[dict]], limit: int = 4) -> str:
    if not context:
        return "(no prior turns)"
    lines = []
    for entry in context[-limit:]:
        role = entry.get("role", "unknown")
        content = (entry.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:220]}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(".!?")


def strip_addressing_noise(text: str) -> str:
    """Remove addressee tags / leading roster names before vague checks."""
    cleaned = (text or "").strip()
    cleaned = ADDRESSING_TAG_RE.sub("", cleaned).strip()
    cleaned = LEADING_NAME_RE.sub("", cleaned).strip()
    return cleaned or (text or "").strip()


def _content_tokens(text: str) -> set[str]:
    stop = {
        "okay",
        "ok",
        "yeah",
        "just",
        "like",
        "this",
        "that",
        "here",
        "what",
        "which",
        "would",
        "could",
        "should",
        "have",
        "with",
        "from",
        "your",
        "you",
        "the",
        "and",
        "for",
        "are",
    }
    return {
        t
        for t in re.findall(r"[a-z]{4,}", (text or "").lower())
        if t not in stop and not ROSTER_NAME_RE.fullmatch(t)
    }


def _task_content_overlap(text: str, task_text: str) -> bool:
    if not task_text:
        return False
    return len(_content_tokens(text) & _content_tokens(task_text)) >= 2


def _has_math_content(
    text: str,
    task_text: str = "",
    construct_cues: Optional[Sequence[str]] = None,
) -> bool:
    lower = text.lower()
    if MATH_STRUCTURE.search(text):
        return True
    if any(v in lower for v in GENERIC_MATH_VERBS):
        return True
    cues = construct_cues if construct_cues is not None else math_cue_phrases()
    if any(cue in lower for cue in cues if len(cue) >= 3):
        return True
    return _task_content_overlap(text, task_text)


def _has_social_content(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in SOCIAL_PATTERNS)


def _has_scaffold_action(text: str) -> bool:
    lower = text.lower()
    return any(cue in lower for cue in SCAFFOLD_ACTION_CUES)


def _is_vague_ack_only(text: str) -> bool:
    cleaned = strip_addressing_noise(text)
    cleaned = ROSTER_NAME_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _normalize(cleaned) in VAGUE_ACK_TOKENS


def _is_narrow_vague_directive(text: str) -> bool:
    cleaned = strip_addressing_noise(text)
    if _is_vague_ack_only(cleaned):
        return False
    if _has_scaffold_action(cleaned) or _has_math_content(cleaned):
        return False
    if not any(p.search(cleaned) for p in VAGUE_DIRECTIVE_PATTERNS):
        return False
    if re.search(
        r"\b(?:what|which|why|how)\b.+\b(?:number|plan|mean|table|graph|cost|rate|text)",
        cleaned,
        re.I,
    ):
        return False
    return True


def _is_vague_directive(text: str) -> bool:
    """Backward-compatible name used by tests; now high-precision only."""
    return _is_narrow_vague_directive(text)


def _last_student_content(context: Optional[List[dict]]) -> str:
    if not context:
        return ""
    for entry in reversed(context):
        role = (entry.get("role") or "").lower()
        if role in ("student", "assistant") or role.startswith("student"):
            return (entry.get("content") or "").strip()
    return ""


def discourse_continues_prior_student(
    teacher_msg: str,
    context: Optional[List[dict]] = None,
) -> bool:
    """True when teacher follow-up slots into the student's last claim."""
    prior = _last_student_content(context)
    if not prior:
        return False
    cleaned = strip_addressing_noise(teacher_msg)
    if not cleaned:
        return False
    if not (
        _has_scaffold_action(cleaned)
        or re.search(r"\b(?:what|which|why|how|can you|could you)\b", cleaned, re.I)
    ):
        return False
    overlap = _content_tokens(cleaned) & _content_tokens(prior)
    if len(overlap) >= 1:
        return True
    bridges = (
        "number",
        "numbers",
        "text",
        "texts",
        "plan",
        "table",
        "graph",
        "equation",
        "cost",
        "cheaper",
        "equal",
    )
    prior_l = prior.lower()
    msg_l = cleaned.lower()
    return any(b in prior_l and b in msg_l for b in bridges)


def is_actionable_scaffold(
    teacher_msg: str,
    task_text: str = "",
    context: Optional[List[dict]] = None,
) -> bool:
    cleaned = strip_addressing_noise(teacher_msg)
    if not cleaned:
        return False
    if _is_vague_ack_only(cleaned):
        return False
    if _has_scaffold_action(cleaned):
        return True
    if _has_math_content(cleaned, task_text):
        return True
    if discourse_continues_prior_student(cleaned, context):
        return True
    return False


def is_high_precision_vague(
    teacher_msg: str,
    turn_mode: TurnMode,
) -> bool:
    """True only for ack-only / empty-continue prompts safe to stall on."""
    if turn_mode == "vague_acknowledgment":
        return _is_vague_ack_only(teacher_msg)
    if turn_mode == "vague_directive":
        return _is_narrow_vague_directive(teacher_msg)
    return False


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return dot / (na * nb)


def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    try:
        from app.misconception_store import _get_embedder

        vectors = _get_embedder().encode(texts)
        return [list(map(float, v)) for v in vectors]
    except Exception as exc:
        logger.debug("turn classify embed unavailable: %s", exc)
        return None


def _max_prototype_sim(query: List[float], prototypes: Sequence[str]) -> float:
    global _prototype_cache
    missing = [p for p in prototypes if p not in _prototype_cache]
    if missing:
        embedded = _embed_texts(list(missing))
        if not embedded:
            return 0.0
        for text, vec in zip(missing, embedded):
            _prototype_cache[text] = vec
    best = 0.0
    for p in prototypes:
        vec = _prototype_cache.get(p)
        if vec:
            best = max(best, _cosine(query, vec))
    return best


def semantic_prefers_scaffold(
    teacher_msg: str,
    context: Optional[List[dict]] = None,
) -> Optional[bool]:
    """Local MiniLM vote: scaffold prototypes (+ last student) vs vague prototypes.

    Returns True/False on a clear margin, else None (ambiguous / unavailable).
    """
    if not TURN_CLASSIFY_SEMANTIC:
        return None
    cleaned = strip_addressing_noise(teacher_msg)
    if not cleaned or _is_vague_ack_only(cleaned):
        return False
    if _is_narrow_vague_directive(cleaned) and not _has_scaffold_action(cleaned):
        return False

    texts = [cleaned]
    prior = _last_student_content(context)
    if prior:
        texts.append(prior[:400])
    embedded = _embed_texts(texts)
    if not embedded:
        return None
    query = embedded[0]
    vague_sim = _max_prototype_sim(query, VAGUE_PROTOTYPES)
    scaffold_sim = _max_prototype_sim(query, SCAFFOLD_PROTOTYPES)

    # Boost scaffold when teacher utterance is close to the last student claim
    # and looks like a follow-up question.
    if prior and len(embedded) > 1:
        prior_sim = _cosine(query, embedded[1])
        if prior_sim >= 0.35 and re.search(
            r"\b(?:what|which|why|how|can you|could you|select|try|pick)\b",
            cleaned,
            re.I,
        ):
            scaffold_sim = max(scaffold_sim, prior_sim)

    if scaffold_sim - vague_sim >= TURN_CLASSIFY_SEMANTIC_MARGIN:
        return True
    if vague_sim - scaffold_sim >= TURN_CLASSIFY_SEMANTIC_MARGIN and not _has_scaffold_action(
        cleaned
    ):
        return False
    return None


_MATH_PRESS_PATTERNS = (
    "why ",
    "explain",
    "what's wrong",
    "what is wrong",
    "using the table",
    "did you understand",
    "get it",
    "getit",
    "pointed out",
    "20 won't matter",
    "won't matter",
    "what do you think",
    "find what",
    "help ",
    "check him",
    "check her",
    "when is each plan",
    "which plan is better",
    "what's the answer",
)


def _is_math_press_turn(teacher_msg: str) -> bool:
    cleaned = strip_addressing_noise(teacher_msg or "").strip().lower()
    if not cleaned:
        return False
    if "?" in cleaned:
        return True
    return any(p in cleaned for p in _MATH_PRESS_PATTERNS)


def _extract_json_payload(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.S)
    return match.group(0) if match else text


def _hard_gate_label(
    teacher_msg: str,
    task_text: str = "",
    context: Optional[List[dict]] = None,
) -> Optional[TurnMode]:
    """Only undeniable labels — prefer leaving the rest to semantics + LLM."""
    stripped = (teacher_msg or "").strip()
    if not stripped:
        return "vague_acknowledgment"
    cleaned = strip_addressing_noise(stripped)
    if _is_vague_ack_only(cleaned):
        return "vague_acknowledgment"
    if _is_math_press_turn(stripped):
        return "math_scaffold"
    if is_actionable_scaffold(cleaned, task_text, context):
        return "math_scaffold"
    if _is_narrow_vague_directive(cleaned):
        return "vague_directive"
    if _has_social_content(cleaned) and not _has_math_content(cleaned, task_text):
        # Pure social hard-gated (any length); mixed social+math goes to LLM.
        return "social"
    return None


def _heuristic_classify(
    teacher_msg: str,
    task_text: str = "",
    context: Optional[List[dict]] = None,
) -> TurnMode:
    """Fast path: prefer scaffold over vague when any concrete action is present."""
    stripped = (teacher_msg or "").strip()
    if not stripped:
        return "vague_acknowledgment"

    cleaned = strip_addressing_noise(stripped)

    if is_actionable_scaffold(cleaned, task_text, context):
        return "math_scaffold"

    if _is_vague_ack_only(cleaned):
        return "vague_acknowledgment"

    if _is_narrow_vague_directive(cleaned):
        return "vague_directive"

    has_math = _has_math_content(cleaned, task_text)
    has_social = _has_social_content(cleaned)

    if has_social and has_math:
        return "mixed"
    if has_social and not has_math:
        return "social"

    lower = cleaned.lower()
    if any(p in lower for p in EVAL_PATTERNS) and len(cleaned) < 80:
        return "math_eval"

    if has_math:
        return "math_scaffold"

    if discourse_continues_prior_student(cleaned, context):
        return "math_scaffold"

    semantic = semantic_prefers_scaffold(cleaned, context)
    if semantic is True:
        return "math_scaffold"
    if semantic is False and len(cleaned.split()) <= 6:
        return "vague_directive"

    if len(cleaned) < 20 and not has_math:
        return "vague_acknowledgment"

    return "math_scaffold"


def _llm_classify(
    teacher_msg: str,
    task_text: str = "",
    context: Optional[List[dict]] = None,
) -> Optional[Tuple[TurnMode, float, dict]]:
    last_student = _last_student_content(context) or "(none)"
    prompt = TURN_CLASSIFY_PROMPT.format(
        task_text=(task_text or "middle school math")[:400],
        context_window=format_context_window(context, limit=6),
        last_student_claim=last_student[:400].replace('"', "'"),
        teacher_turn=teacher_msg.replace('"', "'")[:500],
    )
    try:
        raw = complete(
            LLMRole.TURN_CLASSIFIER,
            "You are a precise dialogue-act tagger. Prefer math_scaffold over vague when unsure. JSON only.",
            prompt,
        )
        payload = json.loads(_extract_json_payload(raw))
        label = (payload.get("label") or "").strip().lower()
        label = re.sub(r"[^a-z_]", "", label.replace(" ", "_"))
        if label not in VALID_LABELS:
            logger.warning("LLM turn label invalid: %s", label)
            return None
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return label, confidence, payload  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("LLM turn classification failed: %s", exc)
    return None


def _needs_llm_disambiguation(
    teacher_msg: str,
    heuristic_label: TurnMode,
    context: Optional[List[dict]] = None,
) -> bool:
    """Legacy helper: in semantic_llm mode we call LLM for non-hard-gated turns."""
    if TURN_CLASSIFY_MODE in ("llm", "semantic_llm"):
        return _hard_gate_label(teacher_msg, context=context) is None
    if heuristic_label in ("social", "off_topic", "mixed"):
        return True
    cleaned = strip_addressing_noise(teacher_msg)
    if is_actionable_scaffold(cleaned, context=context):
        return False
    if is_high_precision_vague(cleaned, heuristic_label):
        return False
    if (
        len(cleaned) > 120
        and _has_math_content(cleaned)
        and _has_social_content(cleaned)
    ):
        return True
    if (
        heuristic_label == "math_scaffold"
        and len(cleaned) >= 20
        and not _has_math_content(cleaned)
        and not _has_scaffold_action(cleaned)
        and not discourse_continues_prior_student(cleaned, context)
    ):
        return True
    return False


def _apply_vague_safety(
    label: TurnMode,
    confidence: float,
    teacher_msg: str,
    task_text: str,
    context: Optional[List[dict]],
    fallback: TurnMode,
    payload: Optional[dict] = None,
) -> TurnMode:
    """Never keep a vague label when evidence says the turn is actionable."""
    if label not in ("vague_acknowledgment", "vague_directive"):
        return label
    if is_actionable_scaffold(teacher_msg, task_text, context):
        return "math_scaffold"
    if not is_high_precision_vague(teacher_msg, label):
        return "math_scaffold"
    if confidence < LLM_VAGUE_MIN_CONFIDENCE:
        return fallback if fallback not in ("vague_acknowledgment", "vague_directive") else "math_scaffold"
    if payload:
        if payload.get("has_concrete_ask") is True:
            return "math_scaffold"
        if payload.get("continues_prior_student") is True:
            return "math_scaffold"
    return label


def classify_teacher_turn(
    teacher_msg: str,
    task_text: str = "",
    context: Optional[List[dict]] = None,
) -> TurnMode:
    """Classify teacher message for conversational vs math prompt routing.

    Default mode ``semantic_llm``:
      1) hard gates (ack-only / go-on / clear scaffold cues / discourse)
      2) local MiniLM semantic vote (+ last-student similarity)
      3) cheap/fast TURN_CLASSIFIER model with rich discourse context
      4) vague-safety: prefer scaffold when unsure
    """
    stripped = (teacher_msg or "").strip()
    if not stripped:
        return "vague_acknowledgment"

    mode = (TURN_CLASSIFY_MODE or "semantic_llm").lower().strip()

    if mode == "heuristic":
        return _heuristic_classify(stripped, task_text, context)

    # --- Hard gates (safe either way) ---
    hard = _hard_gate_label(stripped, task_text, context)
    if hard in ("vague_acknowledgment", "vague_directive"):
        return hard
    if hard == "math_scaffold":
        return "math_scaffold"
    if hard == "social":
        return "social"

    # --- Local semantic vote (fast, no API) ---
    semantic = semantic_prefers_scaffold(stripped, context)
    if semantic is True:
        return "math_scaffold"

    if mode == "llm" or mode == "semantic_llm":
        llm = _llm_classify(stripped, task_text, context)
        if llm:
            label, confidence, payload = llm
            fallback = _heuristic_classify(stripped, task_text, context)
            if semantic is False and fallback in (
                "vague_acknowledgment",
                "vague_directive",
            ):
                # Semantic leans vague and heuristic agrees — still apply safety.
                pass
            return _apply_vague_safety(
                label, confidence, stripped, task_text, context, fallback, payload
            )

    # LLM unavailable or mode without LLM
    if semantic is False:
        return "vague_directive"
    return _heuristic_classify(stripped, task_text, context)


def is_vague_acknowledgment(turn_mode: TurnMode) -> bool:
    return turn_mode == "vague_acknowledgment"


def mastery_update_turn(turn_mode: TurnMode) -> bool:
    """Only math_eval and math_scaffold may update mastery (Phase 2)."""
    return turn_mode in ("math_eval", "math_scaffold")


def claim_lock_active(turn_mode: str, *, stall_active: bool = False) -> bool:
    """Whether fight CLAIM LOCK / claim critic should apply this turn.

    Social, off-topic, and stall turns must not force Plan A/B restatement.
    """
    if stall_active:
        return False
    return (turn_mode or "") not in NON_MATH_TURN_MODES
