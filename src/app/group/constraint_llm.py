"""LLM-based speak-constraint parsing for group facilitation (Track C bake-off)."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence

from app.group.orchestrator import SpeakConstraints, parse_speak_constraints
from app.llm import complete
from app.llm.roles import LLMRole

logger = logging.getLogger(__name__)

_SYSTEM = """You parse a middle-school teacher's facilitation message for a small-group math discussion.

Return JSON only with keys:
  must_speak: profile_ids directly told to answer/speak (actors)
  may_speak: profile_ids eligible to volunteer (discuss/open/critique); empty if mode is direct
  must_not_speak: profile_ids told to watch/listen/observe/stay quiet/sit out
  mode: one of "direct" | "discuss" | "open" | "critique"
  rationale: short string

Rules:
- Names used only as topics (e.g. "Jordan's method", "agree with Jordan") are NOT speakers.
- "watch", "listen", "observe", "hang back", "sit this one out", "stay quiet" → must_not_speak.
- If the teacher names specific students to discuss, those go in may_speak and mode=discuss; must_speak=[].
- If one or more students are directly asked a question and it is not a discuss cue, mode=direct and must_speak=those actors.
- "Anyone disagree?" / critique without names → mode=critique, may_speak=all roster minus must_not.
- Open questions with no names → mode=open, may_speak=all minus must_not.
- Only use profile_ids from the provided roster. Prefer lowercase ids as given.
"""


def _normalize_ids(
    values: Optional[Sequence[str]],
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> List[str]:
    aliases = {pid.lower(): pid for pid in profile_ids}
    for pid, name in (display_names or {}).items():
        if name:
            aliases[str(name).lower()] = pid
        aliases[str(pid).lower()] = pid
    out: List[str] = []
    seen = set()
    for raw in values or []:
        key = str(raw).strip().lower().lstrip("@")
        pid = aliases.get(key)
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _parse_json_constraints(
    raw: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> SpeakConstraints:
    data = json.loads(raw)
    mode = str(data.get("mode") or "open").lower().strip()
    if mode not in ("direct", "discuss", "open", "critique"):
        mode = "open"
    must = _normalize_ids(data.get("must_speak"), profile_ids, display_names)
    may = _normalize_ids(data.get("may_speak"), profile_ids, display_names)
    must_not = _normalize_ids(data.get("must_not_speak"), profile_ids, display_names)
    must = [p for p in must if p not in must_not]
    may = [p for p in may if p not in must_not]
    if mode == "direct":
        may = []
    elif not may:
        may = [p for p in profile_ids if p not in must_not]
    return SpeakConstraints(
        must_speak=must,
        may_speak=may,
        must_not_speak=must_not,
        mode=mode,
    )


def parse_speak_constraints_llm(
    message: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> SpeakConstraints:
    """LLM parse of teacher facilitation constraints."""
    roster_lines = [
        f"- {pid} (display: {display_names.get(pid, pid)})" for pid in profile_ids
    ]
    user = (
        "Roster profile_ids:\n"
        + "\n".join(roster_lines)
        + f"\n\nTeacher message:\n{message}\n"
    )
    raw = complete(LLMRole.GROUP_CONSTRAINT, _SYSTEM, user)
    return _parse_json_constraints(raw, profile_ids, display_names)


def merge_constraints_regex_veto(
    llm: SpeakConstraints,
    regex: SpeakConstraints,
) -> SpeakConstraints:
    """Hybrid: keep LLM parse but never unmute regex exclusions."""
    must_not = list(
        dict.fromkeys(list(llm.must_not_speak) + list(regex.must_not_speak))
    )
    must = [p for p in llm.must_speak if p not in must_not]
    may = [p for p in llm.may_speak if p not in must_not]
    mode = llm.mode
    # If regex saw discuss and LLM missed it, prefer discuss when exclude present
    if regex.mode == "discuss" and mode == "direct" and must_not:
        mode = "discuss"
        must = []
        if not may:
            may = [
                p
                for p in (regex.may_speak or list(dict.fromkeys(llm.must_speak + llm.may_speak)))
                if p not in must_not
            ]
    if mode == "direct":
        may = []
    return SpeakConstraints(
        must_speak=must,
        may_speak=may,
        must_not_speak=must_not,
        mode=mode,
    )


def parse_speak_constraints_hybrid(
    message: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> SpeakConstraints:
    """LLM constraints with regex must_not veto (recommended if LLM is adopted)."""
    regex = parse_speak_constraints(message, profile_ids, display_names)
    try:
        llm = parse_speak_constraints_llm(message, profile_ids, display_names)
    except Exception as exc:  # noqa: BLE001 — bake-off / demo resilience
        logger.warning("LLM constraint parse failed; using regex only: %s", exc)
        return regex
    return merge_constraints_regex_veto(llm, regex)


def resolve_constraint_parser(name: Optional[str] = None):
    """Return constraint parser: hybrid (default) | llm | regex.

    ``regex`` remains available for the hybrid veto and offline gold baselines.
    Production group sessions default to ``hybrid``.
    """
    from app.config import GROUP_CONSTRAINT_PARSER

    key = (name or GROUP_CONSTRAINT_PARSER or "hybrid").lower().strip()
    if key == "regex":
        return parse_speak_constraints
    if key == "llm":
        return parse_speak_constraints_llm
    if key == "hybrid":
        return parse_speak_constraints_hybrid
    logger.warning("Unknown GROUP_CONSTRAINT_PARSER=%r; using hybrid", key)
    return parse_speak_constraints_hybrid
