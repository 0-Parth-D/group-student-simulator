"""Deterministic teacher constraints and turn orchestration for group sessions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from app.group_sessions import GroupSession

DISCUSS_PATTERNS = (
    re.compile(r"\bdiscuss(\s+together)?\b", re.IGNORECASE),
    re.compile(r"\btalk\s+together\b", re.IGNORECASE),
    re.compile(r"\btalk\s+in\s+your\s+group\b", re.IGNORECASE),
    re.compile(r"\bamong\s+yourselves\b", re.IGNORECASE),
    re.compile(r"\bwork\s+(it\s+)?out\s+(together|as\s+a\s+group)\b", re.IGNORECASE),
)

CRITIQUE_PATTERNS = (
    re.compile(r"\b(disagree|challenge|critique|different\s+(idea|answer))\b", re.I),
)

EXCLUDE_WORDS = r"(?:watch|listen|observe|stay\s+quiet|be\s+quiet|don'?t\s+(?:answer|speak))"
ACTOR_WORDS = r"(?:what|why|how|do|does|did|can|could|would|will|explain|show|tell|answer|try|share)"


@dataclass(frozen=True)
class SpeakConstraints:
    must_speak: List[str]
    may_speak: List[str]
    must_not_speak: List[str]
    mode: str


@dataclass(frozen=True)
class SpeakerResolution:
    speakers: List[str]
    constraints: SpeakConstraints
    decisions: List[dict]


def _alias_to_profile(
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> Dict[str, str]:
    """Map lowercased profile_id / first name → profile_id."""
    aliases: Dict[str, str] = {}
    for pid in profile_ids:
        aliases[pid.lower()] = pid
        name = (display_names.get(pid) or pid).strip()
        if name:
            aliases[name.lower()] = pid
    return aliases


def _name_hits(
    message: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> List[tuple[int, int, str, str]]:
    """Return unique alias hits as (start, end, profile_id, matched_text)."""
    aliases = _alias_to_profile(profile_ids, display_names)
    hits: List[tuple[int, int, str, str]] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for alias, pid in sorted(aliases.items(), key=lambda item: -len(item[0])):
        for match in re.finditer(rf"(?<!\w)@?{re.escape(alias)}(?!\w)", message, re.I):
            key = (match.start(), match.end(), pid)
            if key not in seen_spans:
                seen_spans.add(key)
                hits.append((match.start(), match.end(), pid, match.group(0)))
    return sorted(hits, key=lambda item: item[0])


def parse_addressed_names(
    message: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> List[str]:
    """Return roster profile_ids addressed in the message, in first-occurrence order.

    Patterns (case-insensitive):
    - @id / @FirstName
    - word-boundary first name or profile_id
    - leading ``Name,`` or ``Name:``
    """
    text = message or ""
    if not text.strip() or not profile_ids:
        return []

    seen: set[str] = set()
    ordered: List[str] = []
    for _, _, pid, _ in _name_hits(text, profile_ids, display_names):
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def has_discuss_cue(message: str) -> bool:
    """True if the teacher message asks the group to discuss among themselves."""
    text = (message or "").strip()
    if not text:
        return False
    return any(p.search(text) for p in DISCUSS_PATTERNS)


def _is_excluded(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 24):start]
    after = text[end:min(len(text), end + 40)]
    return bool(
        re.search(r"\bnot\s*$", before, re.I)
        or re.search(rf"^\s*[,:\-]?\s*(?:please\s+|just\s+)?{EXCLUDE_WORDS}\b", after, re.I)
        or re.search(rf"{EXCLUDE_WORDS}\s*[,:\-]?\s*$", before, re.I)
    )


def _is_topic_only(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 24):start]
    after = text[end:min(len(text), end + 24)]
    return bool(
        re.match(r"['’]s\b", after, re.I)
        or re.search(r"\b(?:explain|show|tell)\s+to\s*$", before, re.I)
        or re.search(r"\b(?:agree|disagree)\s+with\s*$", before, re.I)
        or re.search(r"\b(?:method|idea|answer)\s+(?:from|by)\s*$", before, re.I)
    )


def _is_actor(text: str, start: int, end: int, matched: str) -> bool:
    if matched.startswith("@"):
        return True
    before = text[max(0, start - 24):start]
    after = text[end:min(len(text), end + 48)]
    if re.search(r"\bask\s*$", before, re.I):
        return True
    if re.match(rf"\s*[,:\-]?\s*{ACTOR_WORDS}\b", after, re.I):
        return True
    # Name starts a clause, including "Alex and Sam discuss".
    clause_prefix = text[:start]
    starts_clause = not clause_prefix.strip() or bool(
        re.search(r"(?:[.;!?]\s*|\bthen\s+)$", clause_prefix, re.I)
    )
    if starts_clause and re.match(r"\s*(?:,|and\b|\bdiscuss\b)", after, re.I):
        return True
    # A name joined to another actor in the same leading noun phrase.
    if re.search(r"\band\s+$", before, re.I) and re.search(
        r"\b(?:discuss|talk|what|do|can|explain)\b", after, re.I
    ):
        return True
    return False


def parse_speak_constraints(
    message: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> SpeakConstraints:
    """Regex/deterministic constraint parse.

    Production group turns use ``hybrid`` (LLM + this function as must_not veto).
    Call this directly only for offline tests, gold ``--parser regex``, or veto merge.
    """
    text = message or ""
    discuss = has_discuss_cue(text)
    critique = any(pattern.search(text) for pattern in CRITIQUE_PATTERNS)
    mode = "discuss" if discuss else "critique" if critique else "open"

    actors: List[str] = []
    excluded: List[str] = []
    for start, end, pid, matched in _name_hits(text, profile_ids, display_names):
        if _is_excluded(text, start, end):
            if pid not in excluded:
                excluded.append(pid)
            continue
        if _is_topic_only(text, start, end):
            continue
        if _is_actor(text, start, end, matched) and pid not in actors:
            actors.append(pid)

    if actors and not discuss:
        mode = "direct"
    must_speak = [] if discuss else [pid for pid in actors if pid not in excluded]
    if discuss:
        may_speak = [pid for pid in actors if pid not in excluded]
        if not may_speak:
            may_speak = [pid for pid in profile_ids if pid not in excluded]
    elif mode in ("open", "critique"):
        may_speak = [pid for pid in profile_ids if pid not in excluded]
    else:
        may_speak = []

    return SpeakConstraints(
        must_speak=must_speak,
        may_speak=may_speak,
        must_not_speak=excluded,
        mode=mode,
    )


def resolve_speakers_with_meta(
    message: str,
    group: "GroupSession",
    *,
    enable_self_select: bool = True,
    last_peer_reply: Optional[str] = None,
    constraint_parser: Optional[str] = None,
) -> SpeakerResolution:
    """Resolve ordered speakers and transparent per-student decisions.

    Uses ``GROUP_CONSTRAINT_PARSER`` (default ``hybrid``). Pass
    ``constraint_parser="regex"`` only for offline unit tests.
    """
    from app.group_constraint_llm import resolve_constraint_parser

    parse_fn = resolve_constraint_parser(constraint_parser)
    constraints = parse_fn(message, group.profile_ids, group.display_names)
    excluded = set(constraints.must_not_speak)

    if constraints.must_speak:
        speakers = [pid for pid in constraints.must_speak if pid not in excluded]
        decisions = [
            {
                "profile_id": pid,
                "will_speak": pid in speakers,
                "score": 1.0 if pid in speakers else 0.0,
                "reason": (
                    "must_speak (directly addressed)"
                    if pid in speakers
                    else "not eligible this turn"
                ),
            }
            for pid in group.profile_ids
        ]
    else:
        candidates = constraints.may_speak or [
            pid for pid in group.profile_ids if pid not in excluded
        ]
        if enable_self_select:
            from app.group_speak_policy import self_select

            speakers, decisions = self_select(
                candidates,
                group,
                constraints,
                message,
                last_peer_reply=last_peer_reply,
            )
        else:
            speakers = (
                list(candidates)
                if constraints.mode == "discuss"
                else list(candidates[:1])
            )
            decisions = [
                {
                    "profile_id": pid,
                    "will_speak": pid in speakers,
                    "score": 1.0 if pid in speakers else 0.0,
                    "reason": (
                        "eligible (legacy teacher-directed order)"
                        if pid in speakers
                        else "not eligible this turn"
                    ),
                }
                for pid in group.profile_ids
            ]

    if not speakers:
        fallback = next((pid for pid in group.profile_ids if pid not in excluded), None)
        if fallback:
            speakers = [fallback]
            for decision in decisions:
                if decision["profile_id"] == fallback:
                    decision.update(
                        will_speak=True,
                        reason=f"{decision['reason']}; fallback speaker",
                    )

    return SpeakerResolution(
        speakers=speakers,
        constraints=constraints,
        decisions=decisions,
    )


def resolve_speakers(
    message: str,
    group: "GroupSession",
    *,
    enable_self_select: bool = True,
    constraint_parser: Optional[str] = None,
) -> List[str]:
    """Public list-only wrapper retained for simple callers and tests."""
    return resolve_speakers_with_meta(
        message,
        group,
        enable_self_select=enable_self_select,
        constraint_parser=constraint_parser,
    ).speakers


def who_speaks(
    message: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> List[str]:
    """Decide which students reply after a teacher message.

    Priority:
    1. Addressed names (names win over discuss cues)
    2. Discussion cue → full round-robin in profile_ids order
    3. Default → first profile only
    """
    if not profile_ids:
        return []

    from app.group_constraint_llm import resolve_constraint_parser

    constraints = resolve_constraint_parser()(
        message, profile_ids, display_names
    )
    if constraints.must_speak:
        return constraints.must_speak
    if constraints.mode in ("discuss", "critique"):
        return constraints.may_speak
    return [pid for pid in profile_ids if pid not in constraints.must_not_speak][:1]
