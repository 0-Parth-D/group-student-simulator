"""Context-aware teacher-turn classification (LLM + heuristic fallback)."""

import json
import logging
import re
from typing import List, Literal, Optional, Sequence

from app.knowledge.construct_text import math_cue_phrases
from app.llm import complete
from app.llm.roles import LLMRole

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
}

VAGUE_DIRECTIVE_MARKERS = (
    "go on",
    "continue",
    "what next",
    "what will you",
    "what do you",
    "keep going",
    "try again",
    "your turn",
    "show me",
)

# Generic social greetings only — never demo roster names.
SOCIAL_PATTERNS = (
    "how are you",
    "how're you",
    "hows your day",
    "how's your day",
    "how have you been",
    "how you doing",
    "how are u",
    "good morning",
    "good afternoon",
    "hello",
    "hi ",
    "hey ",
    "what's up",
    "whats up",
    "nice to see",
    "how was your",
    "how do you feel",
    "feeling today",
)

# Topic-neutral math structure. Domain vocabulary comes from KG cues + task text.
MATH_STRUCTURE = re.compile(
    r"\d+\s*/\s*\d+"          # fractions
    r"|\d+\s*:\s*\d+"          # ratios
    r"|\d+\s*[-+*/=]\s*\d+"    # arithmetic
    r"|\$\s*\d"                # money
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

TURN_CLASSIFY_PROMPT = """Classify the teacher's latest message in a middle-school math tutoring session.
Task context: "{task_text}"

Conversation context (last 3 turns):
{context_window}

Teacher's latest message: "{teacher_turn}"

Choose exactly one label:
- math_eval: explicitly assesses the student's answer (correct/incorrect/close)
- math_scaffold: provides a hint, sub-question, or worked step
- vague_acknowledgment: confirms without new content or next step ("okay", "yeah", "good" alone)
- vague_directive: asks student to continue with a specified direction ("go on", "what next", "try again")
- social: off-topic personal chat or emotional support unrelated to the math task
- off_topic: unrelated to the math task but not personal chat
- mixed: combines social chat with math in one message

Return JSON: {{"label": "...", "confidence": 0.0-1.0, "reason": "..."}}"""


def format_context_window(context: Optional[List[dict]], limit: int = 3) -> str:
    if not context:
        return "(no prior turns)"
    lines = []
    for entry in context[-limit:]:
        role = entry.get("role", "unknown")
        content = (entry.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:200]}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(".!?")


def _task_content_overlap(text: str, task_text: str) -> bool:
    if not task_text:
        return False
    msg_tokens = {t for t in re.findall(r"[a-z]{4,}", text.lower())}
    task_tokens = {t for t in re.findall(r"[a-z]{4,}", task_text.lower())}
    return len(msg_tokens & task_tokens) >= 2


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


def _is_vague_ack_only(text: str) -> bool:
    return _normalize(text) in VAGUE_ACK_TOKENS


def _is_vague_directive(text: str) -> bool:
    lower = text.lower()
    if _is_vague_ack_only(text):
        return False
    return any(m in lower for m in VAGUE_DIRECTIVE_MARKERS)


def _heuristic_classify(teacher_msg: str, task_text: str = "") -> TurnMode:
    """Fast fallback when LLM classification fails."""
    stripped = teacher_msg.strip()
    if not stripped:
        return "vague_acknowledgment"

    if _is_vague_ack_only(stripped):
        return "vague_acknowledgment"

    if _is_vague_directive(stripped):
        return "vague_directive"

    has_math = _has_math_content(stripped, task_text)
    has_social = _has_social_content(stripped)

    if has_social and has_math:
        return "mixed"

    if has_social and not has_math:
        return "social"

    lower = stripped.lower()
    if any(p in lower for p in EVAL_PATTERNS) and len(stripped) < 80:
        return "math_eval"

    if has_math:
        return "math_scaffold"

    if len(stripped) < 20 and not has_math:
        return "vague_acknowledgment"

    return "math_scaffold"


def _llm_classify(
    teacher_msg: str,
    task_text: str = "",
    context: Optional[List[dict]] = None,
) -> Optional[TurnMode]:
    prompt = TURN_CLASSIFY_PROMPT.format(
        task_text=(task_text or "middle school math")[:300],
        context_window=format_context_window(context),
        teacher_turn=teacher_msg.replace('"', "'")[:500],
    )
    try:
        raw = complete(
            LLMRole.TURN_CLASSIFIER,
            "You classify tutoring dialogue acts. Return JSON only.",
            prompt,
        )
        payload = json.loads(raw)
        label = (payload.get("label") or "").strip().lower()
        label = re.sub(r"[^a-z_]", "", label.replace(" ", "_"))
        if label in VALID_LABELS:
            return label  # type: ignore[return-value]
        logger.warning("LLM turn label invalid: %s", label)
    except Exception as exc:
        logger.warning("LLM turn classification failed: %s", exc)
    return None


def _needs_llm_disambiguation(teacher_msg: str, heuristic_label: TurnMode) -> bool:
    """Use LLM only when heuristics cannot confidently route the turn."""
    if heuristic_label in ("social", "off_topic", "mixed"):
        return True
    stripped = teacher_msg.strip()
    if (
        len(stripped) > 120
        and _has_math_content(stripped)
        and _has_social_content(stripped)
    ):
        return True
    # Unknown short prose with no math cues: let the LLM decide social vs scaffold.
    if (
        heuristic_label == "math_scaffold"
        and len(stripped) >= 20
        and not _has_math_content(stripped)
        and not _is_vague_directive(stripped)
    ):
        return True
    return False


def classify_teacher_turn(
    teacher_msg: str,
    task_text: str = "",
    context: Optional[List[dict]] = None,
) -> TurnMode:
    """Classify teacher message for conversational vs math prompt routing."""
    stripped = (teacher_msg or "").strip()
    if not stripped:
        return "vague_acknowledgment"

    heuristic = _heuristic_classify(stripped, task_text)
    if not _needs_llm_disambiguation(stripped, heuristic):
        return heuristic

    label = _llm_classify(stripped, task_text, context)
    return label or heuristic


def is_vague_acknowledgment(turn_mode: TurnMode) -> bool:
    return turn_mode == "vague_acknowledgment"


def mastery_update_turn(turn_mode: TurnMode) -> bool:
    """Only math_eval and math_scaffold may update mastery (Phase 2)."""
    return turn_mode in ("math_eval", "math_scaffold")
