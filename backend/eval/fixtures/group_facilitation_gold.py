"""Load Track C facilitation gold set and score constraint parsers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from app.group_orchestrator import SpeakConstraints, parse_speak_constraints

ConstraintParser = Callable[
    [str, Sequence[str], Dict[str, str]], SpeakConstraints
]

_GOLD_PATH = Path(__file__).resolve().parent / "group_facilitation_gold.yaml"


def load_facilitation_gold(path: Optional[Path] = None) -> dict:
    gold_path = path or _GOLD_PATH
    with open(gold_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_list(values: Optional[Sequence[str]]) -> List[str]:
    return list(values or [])


def constraint_match(
    predicted: SpeakConstraints,
    expected: dict,
) -> Dict[str, bool]:
    """Field-level exact matches (order-sensitive for speaker lists)."""
    return {
        "must_speak": predicted.must_speak == _norm_list(expected.get("must_speak")),
        "may_speak": predicted.may_speak == _norm_list(expected.get("may_speak")),
        "must_not_speak": predicted.must_not_speak
        == _norm_list(expected.get("must_not_speak")),
        "mode": predicted.mode == expected.get("mode"),
    }


def sets_equal_ignore_order(a: Sequence[str], b: Sequence[str]) -> bool:
    return set(a) == set(b)


def constraint_match_setwise(
    predicted: SpeakConstraints,
    expected: dict,
) -> Dict[str, bool]:
    """Same fields but speaker lists compared as sets (order-agnostic)."""
    return {
        "must_speak": sets_equal_ignore_order(
            predicted.must_speak, _norm_list(expected.get("must_speak"))
        ),
        "may_speak": sets_equal_ignore_order(
            predicted.may_speak, _norm_list(expected.get("may_speak"))
        ),
        "must_not_speak": sets_equal_ignore_order(
            predicted.must_not_speak, _norm_list(expected.get("must_not_speak"))
        ),
        "mode": predicted.mode == expected.get("mode"),
    }


def get_parser(name: str) -> ConstraintParser:
    """Return a constraint parser by bake-off name."""
    key = (name or "regex").lower().strip()
    if key == "regex":
        return parse_speak_constraints
    if key == "llm":
        from app.group_constraint_llm import parse_speak_constraints_llm

        return parse_speak_constraints_llm
    if key == "hybrid":
        from app.group_constraint_llm import parse_speak_constraints_hybrid

        return parse_speak_constraints_hybrid
    raise ValueError(f"Unknown parser: {name!r} (use regex|llm|hybrid)")


def score_parser_baseline(
    gold: Optional[dict] = None,
    *,
    parser: Optional[ConstraintParser] = None,
    parser_name: str = "regex",
    setwise: bool = True,
) -> Dict[str, Any]:
    """Run a constraint parser on all gold cases; return summary + per-case."""
    data = gold or load_facilitation_gold()
    roster = data["roster"]
    profile_ids = roster["profile_ids"]
    display_names = roster["display_names"]
    parse_fn = parser or get_parser(parser_name)
    matcher = constraint_match_setwise if setwise else constraint_match

    rows: List[dict] = []
    for case in data.get("cases") or []:
        pred = parse_fn(case["message"], profile_ids, display_names)
        fields = matcher(pred, case)
        exact = all(fields.values())
        hard_fail = bool(case.get("hard")) and (
            not fields["must_not_speak"] or not fields["mode"]
        )
        rows.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "hard": bool(case.get("hard")),
                "difficulty": case.get("difficulty"),
                "exact": exact,
                "hard_fail": hard_fail,
                "fields": fields,
                "predicted": {
                    "must_speak": pred.must_speak,
                    "may_speak": pred.may_speak,
                    "must_not_speak": pred.must_not_speak,
                    "mode": pred.mode,
                },
                "expected": {
                    "must_speak": _norm_list(case.get("must_speak")),
                    "may_speak": _norm_list(case.get("may_speak")),
                    "must_not_speak": _norm_list(case.get("must_not_speak")),
                    "mode": case.get("mode"),
                },
            }
        )

    n = len(rows) or 1
    exact_n = sum(1 for r in rows if r["exact"])
    hard_rows = [r for r in rows if r["hard"]]
    hard_ok = sum(
        1 for r in hard_rows if not r["hard_fail"] and r["fields"]["must_not_speak"]
    )
    by_field = {
        key: sum(1 for r in rows if r["fields"][key]) / n
        for key in ("must_speak", "may_speak", "must_not_speak", "mode")
    }
    return {
        "parser": parser_name if parser is None else getattr(parser, "__name__", "custom"),
        "n": len(rows),
        "exact_accuracy": exact_n / n,
        "exact_count": exact_n,
        "hard_n": len(hard_rows),
        "hard_exclude_ok": hard_ok,
        "hard_exclude_rate": (hard_ok / len(hard_rows)) if hard_rows else 1.0,
        "by_field": by_field,
        "failures": [r for r in rows if not r["exact"]],
        "hard_failures": [r for r in rows if r["hard_fail"]],
        "rows": rows,
    }


def score_regex_baseline(
    gold: Optional[dict] = None,
    *,
    setwise: bool = True,
) -> Dict[str, Any]:
    """Backward-compatible alias for the regex parser bake-off."""
    return score_parser_baseline(gold, parser_name="regex", setwise=setwise)


def format_score_report(summary: Dict[str, Any]) -> str:
    parser = summary.get("parser", "regex")
    lines = [
        f"Track C — facilitation gold ({parser})",
        f"  cases: {summary['n']}",
        f"  exact-set accuracy: {summary['exact_accuracy']:.1%} "
        f"({summary['exact_count']}/{summary['n']})",
        f"  hard exclude OK: {summary['hard_exclude_ok']}/{summary['hard_n']} "
        f"({summary['hard_exclude_rate']:.1%})",
        "  by field:",
    ]
    for key, rate in summary["by_field"].items():
        lines.append(f"    {key}: {rate:.1%}")
    if summary["failures"]:
        lines.append("  failures:")
        for row in summary["failures"]:
            lines.append(
                f"    - {row['id']} ({row.get('category')}) "
                f"pred={row['predicted']} exp={row['expected']}"
            )
    return "\n".join(lines)
