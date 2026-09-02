"""Hybrid task metadata resolver: bank lookup, metadata-driven rules, LLM fallback."""

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from app.constructs import load_task_metadata, lookup_task_metadata
from app.kg_config import get_construct_def, list_construct_ids
from app.llm import complete
from app.llm_config import LLMRole
from app.task_answer import compute_expected_answer

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7
FRAC_LIST = re.compile(r"(\d+)\s*/\s*(\d+)")


@dataclass
class RuleResult:
    constructs: List[str]
    confidence: float
    task_family: str
    pisa_attributes: List[str]
    target_stack_level: int


def _clamp_constructs(construct_ids: List[str]) -> List[str]:
    allowed = set(list_construct_ids())
    seen = set()
    out: List[str] = []
    for cid in construct_ids:
        if cid in allowed and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


@lru_cache(maxsize=1)
def _families() -> Dict[str, dict]:
    return dict(load_task_metadata().get("families") or {})


def _get_bank_task(task_id: str) -> Optional[dict]:
    for task in load_task_metadata().get("tasks", []):
        if task.get("task_id") == task_id:
            return deepcopy(task)
    return None


def _family_bank(family_id: str) -> Optional[dict]:
    fam = _families().get(family_id) or {}
    bank_id = fam.get("bank_task_id")
    return _get_bank_task(bank_id) if bank_id else None


def _enrich_from_bank(meta: dict, task_family: str) -> dict:
    bank = _family_bank(task_family)
    if not bank:
        return meta

    for key in (
        "typical_errors",
        "pisa_attributes",
        "target_stack_level",
        "problem_type",
        "answer_strategy",
        "expected_answer",
    ):
        if not meta.get(key) and bank.get(key):
            meta[key] = deepcopy(bank[key])

    if not meta.get("required_constructs"):
        meta["required_constructs"] = list(bank.get("required_constructs") or [])

    meta.setdefault("task_family", task_family)
    meta.setdefault("bank_template_id", bank.get("task_id"))
    return meta


def _signal_matches(signal: dict, text: str, fracs: List[Tuple[str, str]]) -> bool:
    """True when one routing signal from task_metadata.families fires."""
    if signal.get("require_number") and not re.search(r"\b\d+\b", text):
        return False

    min_fracs = int(signal.get("min_fractions") or 0)
    if min_fracs and len(fracs) < min_fracs:
        return False

    if signal.get("same_denominator"):
        if len(fracs) < 2 or fracs[0][1] != fracs[1][1]:
            return False

    if signal.get("unlike_denominators"):
        if len(fracs) < 2 or len({f[1] for f in fracs}) <= 1:
            return False

    if "all_pairs" in signal:
        if not any(a in text and b in text for a, b in signal["all_pairs"]):
            return False

    if "all" in signal and not all(term in text for term in signal["all"]):
        return False

    if "all_groups" in signal:
        for group in signal["all_groups"]:
            if not any(term in text for term in group):
                return False

    if "regex" in signal and not re.search(signal["regex"], text):
        return False

    terms = signal.get("any") or []
    if terms and not any(term in text for term in terms):
        return False

    # A signal with only structural constraints (fractions / denominators) still counts.
    has_content_gate = any(
        k in signal
        for k in ("all_pairs", "all", "all_groups", "regex", "any")
    )
    if has_content_gate:
        return True
    return bool(
        signal.get("same_denominator")
        or signal.get("unlike_denominators")
        or min_fracs
    )


def rule_classify(task_text: str) -> RuleResult:
    """Score families from metadata routing signals. No scenario names in code."""
    text = task_text.lower()
    fracs = FRAC_LIST.findall(task_text)
    scores: Dict[str, float] = {}

    for family_id, fam in _families().items():
        total = 0.0
        for signal in fam.get("signals") or []:
            if _signal_matches(signal, text, fracs):
                total += float(signal.get("weight") or 1.0)
        if total > 0:
            scores[family_id] = total

    if not scores:
        return RuleResult([], 0.0, "", [], 3)

    family = max(scores, key=lambda k: scores[k])
    raw = scores[family]
    confidence = min(0.95, 0.35 + 0.15 * raw)
    bank = _family_bank(family) or {}
    return RuleResult(
        constructs=list(bank.get("required_constructs") or []),
        confidence=confidence,
        task_family=family,
        pisa_attributes=list(
            bank.get("pisa_attributes") or ["mathematical_operation"]
        ),
        target_stack_level=int(bank.get("target_stack_level") or 3),
    )


def _llm_tag_task(task_text: str, canonical_equation: str = "") -> dict:
    constructs_block = []
    for cid in list_construct_ids():
        cdef = get_construct_def(cid) or {}
        constructs_block.append(f"- {cid}: {cdef.get('label', cid)}")

    family_ids = sorted(_families().keys()) + ["unknown"]
    system = (
        "You classify middle-school math problems into learning-progression constructs.\n"
        "Return JSON only with keys:\n"
        "  required_constructs (array of construct IDs from the allowed list),\n"
        "  target_stack_level (integer 2-3),\n"
        "  pisa_attributes (array from: mathematical_operation, mathematical_abstraction, "
        "logical_reasoning, mathematical_modeling, intuitive_imagination, data_analysis),\n"
        "  typical_errors (array of {error, construct, stack_level}),\n"
        f"  task_family (one of: {', '.join(family_ids)}).\n"
        "Use linear_compare when two options each combine a fixed amount and/or a per-unit rate "
        "and the question asks which is better.\n"
        "Use ONLY construct IDs from the allowed list. "
        "If the problem is outside rational-number, ratio, rate, or linear-comparison topics, "
        "set required_constructs to [] and task_family to unknown."
    )
    user_parts = [
        "Allowed constructs:\n" + "\n".join(constructs_block),
        f"\nProblem:\n{task_text}",
    ]
    if canonical_equation:
        user_parts.append(
            f"\nCanonical equation (use this interpretation):\n{canonical_equation}"
        )
    user = "".join(user_parts)

    try:
        raw = complete(LLMRole.TASK_TAGGER, system, user)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM task tagging failed: %s", exc)
        return {}


def _fill_expected_answer(meta: dict, task_text: str, family: str = "") -> dict:
    if meta.get("expected_answer"):
        return meta
    computed = compute_expected_answer(task_text, task_meta=meta, task_family=family)
    if computed:
        meta["expected_answer"] = computed
    return meta


def _meta_from_rule(rule: RuleResult, task_text: str) -> dict:
    meta = {
        "task_id": f"custom_{rule.task_family}",
        "description": task_text,
        "required_constructs": list(rule.constructs),
        "target_stack_level": rule.target_stack_level,
        "pisa_attributes": list(rule.pisa_attributes),
        "typical_errors": [],
        "task_family": rule.task_family,
    }
    meta = _enrich_from_bank(meta, rule.task_family)
    return _fill_expected_answer(meta, task_text, rule.task_family)


def _meta_from_llm(payload: dict, task_text: str) -> dict:
    family = payload.get("task_family") or "unknown"
    constructs = _clamp_constructs(payload.get("required_constructs") or [])

    meta = {
        "task_id": f"custom_{family}" if family != "unknown" else "custom_unknown",
        "description": task_text,
        "required_constructs": constructs,
        "target_stack_level": int(payload.get("target_stack_level") or 3),
        "pisa_attributes": list(
            payload.get("pisa_attributes") or ["mathematical_operation"]
        ),
        "typical_errors": list(payload.get("typical_errors") or []),
        "task_family": family,
    }

    if family in _families():
        meta = _enrich_from_bank(meta, family)

    return _fill_expected_answer(meta, task_text, family)


def resolve_task_metadata(
    task_text: str,
    task_id: str = "",
    force_lp: bool = True,
) -> dict:
    """
    Resolve task metadata for LP routing.

    Returns:
        {
            "task_metadata": dict,
            "resolution": {
                "source": "bank"|"rules"|"llm"|"bank+rules",
                "confidence": float,
                "task_family": str,
                "constructs": list,
            },
        }
    """
    text = (task_text or "").strip()
    if not text:
        raise ValueError("Task text is required")

    if task_id:
        bank = _get_bank_task(task_id)
        if bank:
            bank = deepcopy(bank)
            bank["description"] = text
            bank = _fill_expected_answer(
                bank, text, bank.get("task_family") or task_id
            )
            return {
                "task_metadata": bank,
                "resolution": {
                    "source": "bank",
                    "confidence": 1.0,
                    "task_family": bank.get("task_family") or task_id,
                    "constructs": list(bank.get("required_constructs") or []),
                },
            }

    bank_match = lookup_task_metadata(text)
    if bank_match and bank_match.get("required_constructs"):
        desc = (bank_match.get("description") or "").lower()
        text_lower = text.lower()
        overlap = bool(desc) and (desc in text_lower or text_lower in desc)
        if overlap:
            meta = deepcopy(bank_match)
            meta["description"] = text
            meta = _fill_expected_answer(
                meta, text, meta.get("task_family") or meta.get("task_id", "")
            )
            return {
                "task_metadata": meta,
                "resolution": {
                    "source": "bank",
                    "confidence": 0.95,
                    "task_family": meta.get("task_family") or meta.get("task_id", ""),
                    "constructs": list(meta.get("required_constructs") or []),
                },
            }

    rule = rule_classify(text)

    if rule.confidence >= CONFIDENCE_THRESHOLD and rule.constructs:
        meta = _meta_from_rule(rule, text)
        return {
            "task_metadata": meta,
            "resolution": {
                "source": "rules",
                "confidence": rule.confidence,
                "task_family": rule.task_family,
                "constructs": list(meta.get("required_constructs") or []),
            },
        }

    llm_payload = _llm_tag_task(text)
    meta = _meta_from_llm(llm_payload, text)
    family = llm_payload.get("task_family") or "unknown"

    if not meta.get("required_constructs"):
        if rule.constructs:
            meta = _meta_from_rule(rule, text)
            return {
                "task_metadata": meta,
                "resolution": {
                    "source": "rules",
                    "confidence": rule.confidence,
                    "task_family": rule.task_family,
                    "constructs": list(meta.get("required_constructs") or []),
                },
            }
        if force_lp:
            raise ValueError(
                "Could not classify this problem into the learning progression. "
                "Try a fraction, ratio, unit-rate, or two-plan rate-comparison problem."
            )
        return {
            "task_metadata": {},
            "resolution": {
                "source": "unknown",
                "confidence": 0.0,
                "task_family": "unknown",
                "constructs": [],
            },
        }

    source = "llm"
    confidence = 0.75
    if rule.constructs and rule.task_family == family:
        source = "bank+rules" if meta.get("bank_template_id") else "llm+rules"
        confidence = max(0.75, rule.confidence)

    return {
        "task_metadata": meta,
        "resolution": {
            "source": source,
            "confidence": confidence,
            "task_family": family,
            "constructs": list(meta.get("required_constructs") or []),
        },
    }
