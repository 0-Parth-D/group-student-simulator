"""Persona stability (Big Five) scoring from dialogue sample."""

import json
from typing import Dict, List, Optional

from app.student.demo_students import DEMO_STUDENTS
from app.llm import complete
from app.llm.roles import LLMRole

BFI_PROBE_PROMPT = """Based on this student's dialogue turns, rate them on:
Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness (each 1-7).
Dialogue:
{sampled_turns}
Return JSON with those five keys as integers.
"""

TRAIT_KEYS = (
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
)


def _profile_target_bf(profile_id: str) -> Dict[str, float]:
    """Map demo profile 0-1 big_five to 1-7 scale targets."""
    raw = DEMO_STUDENTS.get(profile_id, {}).get("big_five", {})
    out = {}
    for key in TRAIT_KEYS:
        k = key.lower()
        val = raw.get(k, 0.5)
        out[key] = 1.0 + val * 6.0
    return out


def persona_stability(
    sampled_turns: List[dict],
    profile_id: Optional[str] = None,
) -> dict:
    """Probe Big Five stability from a sample of student turns."""
    lines = []
    for t in sampled_turns:
        role = t.get("role", "student")
        content = (t.get("content") or t.get("reply") or "").strip()
        if content:
            lines.append(f"{role}: {content[:200]}")
    dialogue = "\n".join(lines) or "(empty)"
    try:
        raw = complete(
            LLMRole.EVAL_PERSONA,
            "You rate student persona traits from dialogue. Return JSON only.",
            BFI_PROBE_PROMPT.format(sampled_turns=dialogue[:3000]),
        )
        probed = json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}

    result = dict(probed)
    pid = profile_id or (sampled_turns[0].get("profile_id") if sampled_turns else "")
    if pid and pid in DEMO_STUDENTS:
        targets = _profile_target_bf(pid)
        deviations = {}
        for trait in TRAIT_KEYS:
            probed_val = float(probed.get(trait, probed.get(trait.lower(), 4)))
            target_val = targets[trait]
            deviations[trait] = abs(probed_val - target_val)
        result["profile_id"] = pid
        result["target_traits"] = targets
        result["trait_deviations"] = deviations
        result["mean_absolute_deviation"] = sum(deviations.values()) / len(deviations)
    return result
