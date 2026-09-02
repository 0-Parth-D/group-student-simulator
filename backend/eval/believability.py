"""Believability scoring: does this sound like a middle schooler wrote it?

Distinct from cognitive fidelity, which asks whether a reply matched the intended
knowledge state. A reply can be perfectly aligned to its construct and mastery level
and still read like an adult explaining a misconception. This module scores voice.

`linguistic_proxies` needs no LLM and runs on every export, giving a regression
signal between judge runs and something to calibrate the judge against.
"""

import json
import re
from typing import Dict, Iterable, List

from app.llm import complete
from app.llm_config import LLMRole

BELIEVABILITY_PROMPT = """You are judging whether a simulated student turn reads like a
real middle schooler (grades 6-8) in a math class, NOT whether the math is correct.

Reward: everyday vocabulary, short or run-on sentences, fragments, hedging ("I think",
"kinda", "wait"), self-correction, thinking out loud, mild off-task drift.
Penalize: textbook register, tidy multi-clause explanations, teacherly framing
("Let's consider..."), vocabulary or syntax above grade 8, unnaturally complete answers.

Student turn: "{reply}"

Score 1-5 (5 = a teacher would believe a real student wrote this).
Return JSON: {{"score": int, "tell": str|null}}
where "tell" names the single strongest giveaway if the score is below 4.
"""

HEDGES = (
    "i think",
    "i guess",
    "maybe",
    "kinda",
    "kind of",
    "sorta",
    "sort of",
    "probably",
    "not sure",
    "i dunno",
    "dunno",
    "idk",
    "wait",
    "um",
    "uh",
    "hmm",
    "or something",
    "right?",
    "i mean",
)

# Whole-phrase matching only: "number" must not count as the hedge "um".
_HEDGE_RE = re.compile(
    "|".join(rf"(?<!\w){re.escape(h)}(?!\w)" for h in HEDGES),
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
_WORD = re.compile(r"[a-z']+")
_VOWEL_RUN = re.compile(r"[aeiouy]+")

LONG_TURN_WORDS = 40


def _syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word:
        return 0
    count = len(_VOWEL_RUN.findall(word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def linguistic_proxies(replies: Iterable[str]) -> Dict[str, float]:
    """Voice statistics over a set of student replies. No LLM, no dependencies.

    Returns zeros for an empty input so callers can aggregate without guarding.
    """
    texts = [r.strip() for r in replies if r and r.strip()]
    empty = {
        "n_replies": 0,
        "mean_sentence_words": 0.0,
        "flesch_kincaid_grade": 0.0,
        "type_token_ratio": 0.0,
        "hedge_rate": 0.0,
        "long_turn_rate": 0.0,
        "fragment_rate": 0.0,
    }
    if not texts:
        return empty

    all_words: List[str] = []
    sentence_lengths: List[int] = []
    syllable_total = 0
    fragments = 0
    hedged = 0
    long_turns = 0

    for text in texts:
        lower = text.lower()
        words = _WORD.findall(lower)
        if not words:
            continue
        all_words.extend(words)
        syllable_total += sum(_syllables(w) for w in words)

        sents = _sentences(text)
        for sent in sents:
            n = len(_WORD.findall(sent.lower()))
            if n:
                sentence_lengths.append(n)
        # A turn with no terminal punctuation reads as an unfinished thought.
        if not _SENTENCE_SPLIT.search(text):
            fragments += 1
        if _HEDGE_RE.search(text):
            hedged += 1
        if len(words) > LONG_TURN_WORDS:
            long_turns += 1

    if not all_words or not sentence_lengths:
        return empty

    n_words = len(all_words)
    n_sentences = len(sentence_lengths)
    mean_sentence_words = n_words / n_sentences
    fk = (
        0.39 * mean_sentence_words
        + 11.8 * (syllable_total / n_words)
        - 15.59
    )

    return {
        "n_replies": len(texts),
        "mean_sentence_words": round(mean_sentence_words, 2),
        "flesch_kincaid_grade": round(fk, 2),
        "type_token_ratio": round(len(set(all_words)) / n_words, 3),
        "hedge_rate": round(hedged / len(texts), 3),
        "long_turn_rate": round(long_turns / len(texts), 3),
        "fragment_rate": round(fragments / len(texts), 3),
    }


def believability(turn: dict) -> dict:
    """Score one student turn for middle-school voice."""
    reply = (turn.get("reply") or turn.get("content") or "").replace('"', "'")[:500]
    if not reply.strip():
        return {"score": 0, "tell": "empty reply"}
    try:
        raw = complete(
            LLMRole.EVAL_BELIEVABILITY,
            "You rate whether simulated student dialogue sounds like a real "
            "middle schooler. Return JSON only.",
            BELIEVABILITY_PROMPT.format(reply=reply),
        )
        return json.loads(raw)
    except Exception as exc:
        return {"score": 0, "tell": str(exc)}
