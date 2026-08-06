"""Expected-answer helpers for mastery heuristics.

Bank tasks carry `expected_answer` in metadata; this module only fills gaps for
custom wording. Strategies are registered by `answer_strategy` / `problem_type`,
so a new problem family is a metadata entry plus an optional strategy function —
not another branch of scenario regex.
"""

import logging
import re
from fractions import Fraction
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

FRAC_PAIR = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*([+\-*/])\s*(\d+)\s*/\s*(\d+)"
)
FRAC_LIST = re.compile(r"(\d+)\s*/\s*(\d+)")
FEE = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:[\w-]+\s+){0,2}fee",
    re.IGNORECASE,
)
RATE = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|each|a|/)\s+([a-z]+)",
    re.IGNORECASE,
)

AnswerStrategy = Callable[[str], Optional[str]]


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def try_compute_fraction_answer(task_text: str) -> Optional[str]:
    """Evaluate simple two-fraction expressions like 5/6 - 1/3."""
    match = FRAC_PAIR.search(task_text)
    if not match:
        return None

    a = Fraction(int(match.group(1)), int(match.group(2)))
    op = match.group(3)
    b = Fraction(int(match.group(4)), int(match.group(5)))

    try:
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        elif op == "/":
            if b == 0:
                return None
            result = a / b
        else:
            return None
    except (ZeroDivisionError, ValueError):
        return None

    return _format_fraction(result)


def try_compute_unit_rate(task_text: str) -> Optional[str]:
    """Heuristic for distance/time unit-rate wording."""
    lower = task_text.lower()
    if not any(k in lower for k in ("unit rate", "miles per hour", "per hour", " mph")):
        return None

    numbers = [float(n) for n in re.findall(r"\b(\d+(?:\.\d+)?)\b", task_text)]
    if len(numbers) < 2:
        return None

    a, b = sorted(numbers, reverse=True)[:2]
    if b == 0:
        return None

    rate = a / b
    if rate == int(rate):
        return f"{int(rate)} miles per hour"
    return f"{rate:g} miles per hour"


def try_compute_unit_price(task_text: str) -> Optional[str]:
    """Heuristic for price-per-quantity wording."""
    lower = task_text.lower()
    if not any(k in lower for k in ("unit price", "per oz", "per ounce", "per unit", "cost")):
        return None

    money = re.findall(r"\$?\s*(\d+(?:\.\d+)?)", task_text)
    qty = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|units?)\b", lower)
    if len(money) < 1 or len(qty) < 1:
        return None

    total = float(money[0].replace("$", ""))
    count = float(qty[0])
    if count == 0:
        return None

    price = total / count
    return f"${price:.2f} per oz"


def try_compute_linear_crossover(task_text: str) -> Optional[str]:
    """Crossover for one fixed fee beside the cheaper of exactly two rates."""
    fees = list(FEE.finditer(task_text))
    rates = list(RATE.finditer(task_text))
    if len(fees) != 1 or len(rates) != 2:
        return None

    fee = float(fees[0].group(1))
    values = [float(m.group(1)) for m in rates]
    if values[0] == values[1] or fee <= 0:
        return None

    low_idx = values.index(min(values))
    fee_pos = fees[0].start()
    nearest = min(range(2), key=lambda i: abs(rates[i].start() - fee_pos))
    if nearest != low_idx:
        return None

    crossover = fee / (max(values) - min(values))
    unit = rates[low_idx].group(2).lower()
    if not unit or unit in ("the", "one"):
        return f"{crossover:g}"
    if unit.endswith(("ss", "x", "z", "ch", "sh")):
        unit += "es"
    elif not unit.endswith("s"):
        unit += "s"
    return f"{crossover:g} {unit}"


def is_part_whole_word_problem(
    task_text: str, task_meta: Optional[dict] = None
) -> bool:
    """True when fractions in the story refer to parts of one shared whole."""
    if task_meta:
        kind = task_meta.get("problem_type") or task_meta.get("answer_strategy") or ""
        if kind == "part_whole_word":
            return True

    lower = (task_text or "").lower()
    if not lower:
        return False

    if (" ate " in lower or lower.startswith("ate ")) and any(
        k in lower for k in ("gave away", "gave ")
    ):
        return True

    if any(k in lower for k in ("left", "remaining", "how much is left")) and len(
        list(FRAC_LIST.finditer(task_text))
    ) >= 2:
        if any(k in lower for k in (" ate ", "used ", "spent ", "gave ", "shared ")):
            return True

    return False


def try_compute_fraction_word_problem(task_text: str) -> Optional[str]:
    """1 - a - b when the story removes two shares of one whole; else a - b."""
    lower = task_text.lower()
    if not any(k in lower for k in ("left", "remaining", " ate ", "gave away", "how much")):
        return None

    fracs = [
        Fraction(int(m.group(1)), int(m.group(2)))
        for m in FRAC_LIST.finditer(task_text)
    ]
    if len(fracs) < 2:
        return None

    if (" ate " in lower or lower.startswith("ate ")) and any(
        k in lower for k in ("gave away", "gave ")
    ):
        try:
            result = Fraction(1, 1) - fracs[0] - fracs[1]
            if result >= 0:
                return _format_fraction(result)
        except (ZeroDivisionError, ValueError):
            pass

    try:
        result = fracs[0] - fracs[1]
    except (ZeroDivisionError, ValueError):
        return None

    return _format_fraction(result)


def try_compute_ratio_answer(task_text: str) -> Optional[str]:
    """Extract two counted groups named in the problem and return a:b."""
    lower = task_text.lower()
    if "ratio" not in lower:
        return None

    # Prefer "N red" / "N blue" style counts when present; otherwise first two counts.
    named = re.findall(r"(\d+)\s+([a-z]+)", lower)
    if len(named) >= 2:
        return f"{named[0][0]}:{named[1][0]}"
    return None


def try_compute_frac_compare(task_text: str) -> Optional[str]:
    fracs = [
        Fraction(int(m.group(1)), int(m.group(2)))
        for m in FRAC_LIST.finditer(task_text)
    ]
    if len(fracs) < 2:
        return None
    larger = max(fracs)
    return f"{_format_fraction(larger)} is larger"


# Strategies keyed by answer_strategy / problem_type from task metadata.
ANSWER_STRATEGIES: Dict[str, AnswerStrategy] = {
    "fraction_expression": try_compute_fraction_answer,
    "part_whole_word": try_compute_fraction_word_problem,
    "ratio_identify": try_compute_ratio_answer,
    "unit_rate": try_compute_unit_rate,
    "unit_price": try_compute_unit_price,
    "linear_compare_word": try_compute_linear_crossover,
    "frac_compare": try_compute_frac_compare,
}

# Order for custom tasks with no declared strategy: cheapest / most specific first.
_FALLBACK_ORDER: List[str] = [
    "fraction_expression",
    "part_whole_word",
    "frac_compare",
    "ratio_identify",
    "unit_rate",
    "unit_price",
    "linear_compare_word",
]


def compute_expected_answer(
    task_text: str,
    task_family: str = "",
    task_meta: Optional[dict] = None,
) -> Optional[str]:
    """Best-effort expected answer. Bank metadata wins; strategies fill gaps."""
    meta = task_meta or {}
    if meta.get("expected_answer"):
        return str(meta["expected_answer"])

    strategy = (
        meta.get("answer_strategy")
        or meta.get("problem_type")
        or task_family
        or ""
    )
    # Family ids like frac_add_common map onto the expression strategy.
    if strategy.startswith("frac_add") or strategy.startswith("frac_sub"):
        strategy = "fraction_expression"
    if strategy == "linear_compare":
        strategy = "linear_compare_word"
    if strategy == "frac_word_part_whole":
        strategy = "part_whole_word"

    fn = ANSWER_STRATEGIES.get(strategy)
    if fn:
        answer = fn(task_text)
        if answer:
            return answer

    for key in _FALLBACK_ORDER:
        answer = ANSWER_STRATEGIES[key](task_text)
        if answer:
            return answer
    return None
