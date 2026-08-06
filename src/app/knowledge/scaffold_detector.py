"""Detect teacher scaffolding and map to LP constructs."""

import logging
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from app.config import SCAFFOLD_SEMANTIC, SCAFFOLD_SEMANTIC_THRESHOLD
from app.knowledge.construct_text import (
    all_construct_cues,
    construct_vocabulary,
    math_cue_phrases,
)
from app.knowledge.turn_classifier import TurnMode

logger = logging.getLogger(__name__)

WORKED_EXAMPLE_MARKERS = (
    "for example",
    "let me show",
    "we do it like this",
    "watch me",
    "here's how",
    "here is how",
    "step by step",
    "first we",
    "so you",
    "that means",
    "you would",
    "multiply",
    "divide both",
)

# A worked step shows the work: numbers being operated on, an equation, or a rewrite
# arrow. Topic-neutral by design - the fraction-only version missed ratio, rate, and
# linear tasks entirely.
NUMERIC_WORK = re.compile(
    r"\d+\s*/\s*\d+"          # fractions
    r"|\d+\s*:\s*\d+"          # ratios
    r"|\d+\s*[-+*/=]\s*\d+"    # arithmetic or equation
    r"|\$\s*\d"                # money
    r"|\d+\s*(?:->|→|becomes|turns into)"
)


def scaffold_is_worked_example(teacher_msg: str) -> bool:
    """True when teacher gives a full worked step, not just a hint question."""
    lower = teacher_msg.lower()
    if len(teacher_msg.split()) < 12:
        return False
    marker_hits = sum(1 for m in WORKED_EXAMPLE_MARKERS if m in lower)
    if not marker_hits:
        return False
    if NUMERIC_WORK.search(teacher_msg):
        return True
    # No numerals is still a worked step when the teacher names the procedure.
    return any(cue in lower for cue in math_cue_phrases())


EXPLANATION_MARKERS = (
    "because",
    "so we",
    "first",
    "then",
    "step",
    "means",
    "remember",
    "let me show",
    "for example",
    "you need to",
    "we can",
    "change",
    "multiply",
    "divide both",
)


@dataclass
class ScaffoldEvent:
    construct_id: str
    quality: float
    snippet: str

    def to_dict(self) -> dict:
        return {
            "construct_id": self.construct_id,
            "quality": round(self.quality, 3),
            "snippet": self.snippet[:120],
        }


PROBE_MARKERS = (
    "what",
    "how",
    "why",
    "which",
    "where",
    "can you",
    "could you",
    "do you",
    "tell me",
    "explain",
    "show me",
    "walk me",
)

# Common words long enough to survive the 4-letter filter in _content_tokens.
_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "your",
        "they",
        "them",
        "then",
        "than",
        "what",
        "when",
        "have",
        "does",
        "from",
        "were",
        "will",
        "would",
        "about",
        "there",
        "their",
        "which",
        "these",
        "those",
        "some",
        "much",
        "many",
        "each",
        "into",
        "just",
        "like",
        "here",
        "over",
        "more",
        "very",
        "know",
        "think",
        "want",
        "need",
        "make",
        "take",
        "give",
        "look",
    }
)


def _content_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]{4,}", text.lower())} - _STOPWORDS


def _explanation_quality(text: str, task_text: str = "") -> float:
    """Topic-neutral scaffold quality: reasoning, probing, length, specificity.

    Deliberately free of domain vocabulary so identical facilitation scores the
    same on a fraction task and a rate task.
    """
    lower = text.lower()
    score = 0.35

    if any(m in lower for m in EXPLANATION_MARKERS):
        score += 0.2

    if "?" in text or any(m in lower for m in PROBE_MARKERS):
        score += 0.2

    if len(text.split()) > 15:
        score += 0.15

    grounded = bool(re.search(r"\d", text))
    if not grounded and task_text:
        grounded = len(_content_tokens(text) & _content_tokens(task_text)) >= 2
    if grounded:
        score += 0.1

    return min(1.0, score)


EmbedFn = Callable[[List[str]], list]

# Vectors for the default embedder only; an injected embed_fn is never cached.
_construct_vectors: Dict[str, List[float]] = {}


def _as_list(vector) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(x) for x in vector]


def _default_embed(texts: List[str]) -> list:
    # Imported lazily: loading the sentence-transformer costs seconds, and callers
    # with keyword hits never need it.
    from app.knowledge.misconception_store import _get_embedder

    return _get_embedder().encode(texts)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _construct_vectors_for(cids: List[str], embed_fn: Optional[EmbedFn]) -> Dict[str, List[float]]:
    if embed_fn is not None:
        vectors = embed_fn([construct_vocabulary(c) for c in cids])
        return {c: _as_list(v) for c, v in zip(cids, vectors)}

    missing = [c for c in cids if c not in _construct_vectors]
    if missing:
        for cid, vector in zip(missing, _default_embed([construct_vocabulary(c) for c in missing])):
            _construct_vectors[cid] = _as_list(vector)
    return {c: _construct_vectors[c] for c in cids if c in _construct_vectors}


def _semantic_events(
    teacher_msg: str,
    required: List[str],
    quality_base: float,
    embed_fn: Optional[EmbedFn] = None,
) -> List[ScaffoldEvent]:
    """Grade the teacher message against each required construct's LPF prose.

    Lets a scenario with no hand-written keywords still earn differentiated
    scaffold credit. Returns [] whenever embedding is unavailable, leaving the
    keyword and blanket paths untouched.
    """
    if not SCAFFOLD_SEMANTIC or not required:
        return []

    cids = sorted(required)
    try:
        vectors = _construct_vectors_for(cids, embed_fn)
        query = _as_list((embed_fn or _default_embed)([teacher_msg])[0])
    except Exception as exc:
        logger.debug("Semantic scaffold matching unavailable: %s", exc)
        return []

    threshold = SCAFFOLD_SEMANTIC_THRESHOLD
    headroom = max(1e-6, 1.0 - threshold)
    events = []
    for cid in cids:
        similarity = _cosine(query, vectors.get(cid) or [])
        if similarity < threshold:
            continue
        # Scale into the upper half of quality_base so a bare threshold hit is
        # still worth clearly less than a keyword match.
        scale = 0.5 + 0.5 * min(1.0, (similarity - threshold) / headroom)
        events.append(
            ScaffoldEvent(
                construct_id=cid,
                quality=quality_base * scale,
                snippet=f"semantic match {similarity:.2f}: {teacher_msg[:60]}",
            )
        )
    return events


def detect_scaffold(
    teacher_msg: str,
    task_meta: dict,
    turn_mode: TurnMode,
    embed_fn: Optional[EmbedFn] = None,
) -> List[ScaffoldEvent]:
    """Return scaffold events when teacher explains math constructs."""
    if turn_mode not in ("math_scaffold", "math_eval", "mixed"):
        return []

    required = set(task_meta.get("required_constructs") or [])
    task_text = task_meta.get("description") or task_meta.get("task_text") or ""
    lower = teacher_msg.lower()
    quality_base = _explanation_quality(teacher_msg, task_text)
    events = []

    for cid, keywords in all_construct_cues().items():
        # A construct the task does not exercise must not absorb scaffold credit,
        # or generic words ("compare") boost unrelated constructs.
        if required and cid not in required:
            continue
        if not any(kw in lower for kw in keywords):
            continue

        hit = next((kw for kw in keywords if kw in lower), "")
        events.append(
            ScaffoldEvent(
                construct_id=cid,
                quality=quality_base,
                snippet=hit or teacher_msg[:80],
            )
        )

    # Constructs the keyword table cannot reach fall through to embedding.
    covered = {e.construct_id for e in events}
    uncovered = [cid for cid in sorted(required) if cid not in covered]
    if uncovered:
        events.extend(
            _semantic_events(teacher_msg, uncovered, quality_base, embed_fn)
        )

    if not events and turn_mode == "math_scaffold" and quality_base >= 0.5:
        for cid in required:
            events.append(
                ScaffoldEvent(
                    construct_id=cid,
                    quality=quality_base * 0.6,
                    snippet=teacher_msg[:80],
                )
            )

    seen = set()
    unique = []
    for e in events:
        if e.construct_id not in seen:
            seen.add(e.construct_id)
            unique.append(e)
    return unique[:4]


def apply_scaffold_boost(
    scaffold_boost: dict,
    events: List[ScaffoldEvent],
    gain_rate: float,
    scaffold_sensitivity: float,
) -> dict:
    """Apply teacher scaffold events to session-local boost map."""
    updated = dict(scaffold_boost)
    for event in events:
        delta = scaffold_sensitivity * event.quality * gain_rate
        current = updated.get(event.construct_id, 0.0)
        updated[event.construct_id] = min(1.0, current + delta)
    return updated


def decay_scaffold_boost(scaffold_boost: dict, forget_rate: float) -> dict:
    if not scaffold_boost:
        return {}
    factor = max(0.0, 1.0 - forget_rate)
    return {cid: boost * factor for cid, boost in scaffold_boost.items() if boost * factor > 0.01}
