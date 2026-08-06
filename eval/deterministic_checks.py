"""Rule-based eval checks (no LLM) for control-based student simulation."""

import re
from functools import lru_cache
from typing import Dict, List, Optional

from app.knowledge.kg_config import load_misconception_catalog
from app.knowledge.task_answer import is_part_whole_word_problem

ANXIETY_PHRASES = (
    "i have no idea",
    "i don't know",
    "i dont know",
    "i'm not sure",
    "im not sure",
    "i'm so confused",
    "im so confused",
    "can we skip",
    "this is too hard",
    "what do you think",
)

CONFIDENT_PHRASES = (
    "easy!",
    "alright!",
    "no problem",
    "that's easy",
    "obviously",
    "definitely",
    "i know this",
    "looks fine",
)

# Advance signals: equals / arithmetic ops — not a bare fraction like 1/4.
EQUATION_CHAIN = re.compile(
    r"\b\d+(?:\s*/\s*\d+)?\s*="
    r"|\b\d+\s*[+\-]\s*\d+"
    r"|\b\d+\s*(?:plus|minus|times)\s*\d+"
    r"|\bx\s*="
)


def _reply_lower(turn: dict) -> str:
    return (turn.get("reply") or "").lower()


@lru_cache(maxsize=1)
def _misconceptions_by_id() -> Dict[str, dict]:
    catalog = load_misconception_catalog().get("misconceptions", [])
    return {m.get("id", ""): m for m in catalog if m.get("id")}


def _patterns_from_entry(entry: Optional[dict]) -> List[str]:
    if not entry:
        return []
    return [str(p) for p in (entry.get("detection_patterns") or []) if p]


def _l1_detection_patterns(turn: dict) -> List[str]:
    """Collect L1 math-slip patterns from the active misconception and typical_errors.

    Level 0 (shutdown / avoidance) is excluded: it is not an enacted procedure slip,
    so matching it must not fail expert_slip / Morgan persona checks.
    """
    patterns: List[str] = []

    mid = turn.get("misconception_id") or ""
    misc = _misconceptions_by_id().get(mid)
    if misc and int(misc.get("stack_level") or 99) == 1:
        patterns.extend(_patterns_from_entry(misc))

    for err in turn.get("typical_errors") or []:
        if int(err.get("stack_level", 99)) != 1:
            continue
        patterns.extend(_patterns_from_entry(err))

    # Also pull catalog L1 misconceptions for constructs this turn exercises.
    construct_ids = {
        turn.get("construct_id") or turn.get("primary_construct") or "",
    }
    for err in turn.get("typical_errors") or []:
        if err.get("construct"):
            construct_ids.add(err["construct"])
    for misc in _misconceptions_by_id().values():
        if int(misc.get("stack_level") or 99) != 1:
            continue
        if misc.get("construct_id") in construct_ids:
            patterns.extend(_patterns_from_entry(misc))

    # De-dupe while preserving order.
    return list(dict.fromkeys(p for p in patterns if p))


def _matches_any_pattern(text: str, patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern
        except re.error:
            if pattern.lower() in text.lower():
                return pattern
    return None


def _is_answer_leak(reply: str, expected: str, task_text: str = "") -> bool:
    """True when reply states the correct final answer, not when echoing problem givens."""
    if not expected:
        return False
    lower = reply.lower()
    exp = expected.strip().lower()
    if exp not in lower:
        return False
    task_lower = (task_text or "").lower()
    exp_nospace = re.sub(r"\s+", "", exp)
    compact = re.sub(r"\s+", "", lower)
    if exp in task_lower:
        conclusion_patterns = (
            rf"answeris[^.]*{re.escape(exp_nospace)}",
            rf"(left|remaining)is\s*{re.escape(exp_nospace)}",
            rf"equals?\s*{re.escape(exp_nospace)}",
            rf"is\s*{re.escape(exp_nospace)}\s*(left|remaining)?$",
        )
        if any(re.search(p, compact) for p in conclusion_patterns):
            return True
        return lower.count(exp) > task_lower.count(exp)
    return True


def check_stall_on_vague(turn: dict) -> dict:
    """When stall_active, student should not advance the solution."""
    if not turn.get("stall_active"):
        return {"check": "stall_on_vague", "pass": True, "detail": "not a stall turn"}
    lower = _reply_lower(turn)
    if any(p in lower for p in ("not sure what", "what should i do", "what do you want")):
        return {"check": "stall_on_vague", "pass": True, "detail": "asked for direction"}
    if EQUATION_CHAIN.search(turn.get("reply") or ""):
        return {
            "check": "stall_on_vague",
            "pass": False,
            "detail": "computed new equation while stall_active",
        }
    expected = (turn.get("expected_answer") or "").strip().lower()
    if expected and _is_answer_leak(
        turn.get("reply") or "", expected, turn.get("task_text") or ""
    ):
        return {
            "check": "stall_on_vague",
            "pass": False,
            "detail": "gave final answer while stall_active",
        }
    return {"check": "stall_on_vague", "pass": True, "detail": "no obvious solution advance"}


def check_mode_plausible(turn: dict) -> dict:
    """WRONG/CONFUSED on early turns should not sound fully expert-correct."""
    mode = turn.get("behavior_mode", "")
    turn_num = int(turn.get("turn", 99))
    if turn_num > 1 or mode not in ("WRONG", "CONFUSED_HELPSEEKING"):
        return {"check": "mode_plausible", "pass": True, "detail": "not early wrong/confused turn"}
    expected = (turn.get("expected_answer") or "").lower()
    if _is_answer_leak(turn.get("reply") or "", expected, turn.get("task_text") or ""):
        return {
            "check": "mode_plausible",
            "pass": False,
            "detail": f"{mode} turn 1 contains expected_answer",
        }
    lower = _reply_lower(turn)
    if mode == "WRONG" and "lcd is" in lower and "simplifies to" in lower:
        return {
            "check": "mode_plausible",
            "pass": False,
            "detail": "WRONG turn 1 used expert LCD+simplify phrasing",
        }
    return {"check": "mode_plausible", "pass": True, "detail": "plausible for mode"}


def check_answer_leak(turn: dict) -> dict:
    """WRONG/CONFUSED should not leak correct answer on turn 1."""
    mode = turn.get("behavior_mode", "")
    turn_num = int(turn.get("turn", 99))
    if turn_num > 1 or mode not in ("WRONG", "CONFUSED_HELPSEEKING"):
        return {"check": "answer_leak", "pass": True, "detail": "not applicable"}
    expected = (turn.get("expected_answer") or "").strip().lower()
    if not expected:
        return {"check": "answer_leak", "pass": True, "detail": "no expected_answer"}
    if _is_answer_leak(turn.get("reply") or "", expected, turn.get("task_text") or ""):
        return {"check": "answer_leak", "pass": False, "detail": "expected_answer in reply"}
    return {"check": "answer_leak", "pass": True, "detail": "no leak"}


def check_expert_slip(turn: dict) -> dict:
    """L3 NORMAL_ERROR_PROFILE should not enact L1 typical errors / misconceptions."""
    mode = turn.get("behavior_mode", "")
    level = int(turn.get("student_stack_level", 0) or turn.get("mastery", 0) or 0)
    if mode != "NORMAL_ERROR_PROFILE" or level < 3:
        return {"check": "expert_slip", "pass": True, "detail": "not L3 normal profile"}

    lower = _reply_lower(turn)
    hit = _matches_any_pattern(lower, _l1_detection_patterns(turn))
    if hit:
        return {
            "check": "expert_slip",
            "pass": False,
            "detail": f"L3 enacted L1 pattern matching {hit!r}",
        }

    # Part-whole word problems: LCD-subtract without acknowledging the shared whole.
    if is_part_whole_word_problem(
        turn.get("task_text") or "",
        {"problem_type": turn.get("problem_type")},
    ):
        if "lcd is" in lower and turn_num_is_first(turn) and "subtract" in lower:
            if "whole" not in lower and "one whole" not in lower:
                return {
                    "check": "expert_slip",
                    "pass": False,
                    "detail": "L3 defaulted to LCD subtract without part-whole framing",
                }

    return {"check": "expert_slip", "pass": True, "detail": "no L1 slip detected"}


def turn_num_is_first(turn: dict) -> bool:
    return int(turn.get("turn", 0)) <= 1


def check_persona_direction(turn: dict) -> dict:
    """Profile-specific voice constraints, without task-id branches."""
    profile = turn.get("profile_id", "")
    mode = turn.get("behavior_mode", "")
    lower = _reply_lower(turn)
    if profile == "jordan" and mode == "WRONG":
        if any(p in lower for p in ANXIETY_PHRASES):
            return {
                "check": "persona_direction",
                "pass": False,
                "detail": "Jordan WRONG turn sounds anxious instead of confident-wrong",
            }
        if not any(p in lower for p in CONFIDENT_PHRASES) and turn_num_is_first(turn):
            if "maybe" in lower or "i think" in lower:
                return {
                    "check": "persona_direction",
                    "pass": False,
                    "detail": "Jordan WRONG turn 1 hedges too much",
                }
    if (
        profile == "morgan"
        and mode == "NORMAL_ERROR_PROFILE"
        and turn_num_is_first(turn)
    ):
        hit = _matches_any_pattern(lower, _l1_detection_patterns(turn))
        if hit:
            return {
                "check": "persona_direction",
                "pass": False,
                "detail": f"Morgan enacted L1 pattern matching {hit!r}",
            }
    return {"check": "persona_direction", "pass": True, "detail": "persona OK"}


ALL_CHECKS = (
    check_stall_on_vague,
    check_mode_plausible,
    check_answer_leak,
    check_expert_slip,
    check_persona_direction,
)


def run_deterministic_checks(turn: dict) -> List[dict]:
    return [fn(turn) for fn in ALL_CHECKS]


def aggregate_deterministic(results: List[List[dict]]) -> dict:
    """Aggregate per-turn check lists into pass rates."""
    totals: Dict[str, Dict[str, int]] = {}
    for turn_results in results:
        for r in turn_results:
            name = r["check"]
            if name not in totals:
                totals[name] = {"pass": 0, "fail": 0}
            if r.get("pass"):
                totals[name]["pass"] += 1
            else:
                totals[name]["fail"] += 1
    rates = {}
    for name, counts in totals.items():
        n = counts["pass"] + counts["fail"]
        rates[name] = counts["pass"] / n if n else 0.0
    all_pass = sum(1 for tr in results for r in tr if r.get("pass"))
    all_n = sum(len(tr) for tr in results)
    return {
        "per_check": rates,
        "overall_pass_rate": all_pass / all_n if all_n else 0.0,
        "failed": [
            r
            for tr in results
            for r in tr
            if not r.get("pass")
        ],
    }
