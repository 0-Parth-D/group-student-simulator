"""Per-student learning rate parameters for LP mastery updates."""

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from app.student.models import Student

DEFAULT_PROFILE = {
    "gain_rate": 0.5,
    "slip_rate": 0.25,
    "retention": 0.7,
    "scaffold_sensitivity": 0.6,
    "forget_rate": 0.05,
    "max_step_gain": 1,
}

DEFAULT_CONSTRUCT_GAIN_MODIFIERS = {
    "part_whole": 1.2,
    "fraction_equivalence": 0.6,
    "ratio_concept": 1.0,
    "proportional_reasoning": 0.5,
    "additive_vs_multiplicative": 0.8,
    "linear_relationship": 0.7,
    "rate_comparison": 0.8,
    "symbolic_modeling": 0.5,
}

PROFILE_OVERRIDES: Dict[str, dict] = {
    "alex": {
        "gain_rate": 0.35,
        "slip_rate": 0.05,
        "retention": 0.95,
        "scaffold_sensitivity": 0.5,
        "forget_rate": 0.02,
    },
    "jordan": {
        "gain_rate": 0.10,
        "slip_rate": 0.20,
        "retention": 0.70,
        "scaffold_sensitivity": 1.4,
        "forget_rate": 0.10,
    },
    "sam": {
        "gain_rate": 0.20,
        "slip_rate": 0.12,
        "retention": 0.82,
        "scaffold_sensitivity": 0.9,
        "forget_rate": 0.05,
    },
    "morgan": {
        "gain_rate": 0.55,
        "slip_rate": 0.03,
        "retention": 0.98,
        "scaffold_sensitivity": 0.35,
        "forget_rate": 0.01,
        "construct_gain_modifiers": {
            "part_whole": 1.0,
            "fraction_equivalence": 1.0,
            "fraction_operations": 1.0,
            "ratio_concept": 1.0,
            "ratio_equivalence": 1.0,
            "unit_rate": 1.0,
            "proportional_reasoning": 0.8,
            "additive_vs_multiplicative": 1.0,
            "linear_relationship": 1.0,
            "rate_comparison": 1.0,
            "symbolic_modeling": 0.9,
        },
    },
}


@dataclass
class LearningProfile:
    gain_rate: float = 0.5
    slip_rate: float = 0.25
    retention: float = 0.7
    scaffold_sensitivity: float = 0.6
    forget_rate: float = 0.05
    max_step_gain: int = 1
    construct_gain_modifiers: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONSTRUCT_GAIN_MODIFIERS)
    )

    def effective_gain(self, construct_id: str) -> float:
        modifier = self.construct_gain_modifiers.get(construct_id, 1.0)
        return self.gain_rate * modifier

    def to_dict(self) -> dict:
        return asdict(self)


def _trait_modifiers(student: Student) -> dict:
    mods = {}
    if student.Conscientiousness == "High":
        mods["retention"] = 0.08
        mods["slip_rate"] = -0.05
    elif student.Conscientiousness == "Low":
        mods["retention"] = -0.08
        mods["slip_rate"] = 0.05

    if student.Neuroticism == "High":
        mods["gain_rate"] = -0.05
        mods["scaffold_sensitivity"] = 0.05
    elif student.Neuroticism == "Low":
        mods["gain_rate"] = 0.05

    if student.Openness == "High":
        mods["scaffold_sensitivity"] = 0.05
    elif student.Openness == "Low":
        mods["scaffold_sensitivity"] = -0.08

    return mods


def derive_learning_profile(
    student: Student,
    profile_id: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> LearningProfile:
    """Build learning profile from defaults, named overrides, and Big Five modifiers."""
    base = dict(DEFAULT_PROFILE)
    construct_modifiers = dict(DEFAULT_CONSTRUCT_GAIN_MODIFIERS)
    pid = (profile_id or student.student_id or "").lower()
    for key, profile_map in PROFILE_OVERRIDES.items():
        if key in pid:
            base.update({k: v for k, v in profile_map.items() if k != "construct_gain_modifiers"})
            if "construct_gain_modifiers" in profile_map:
                construct_modifiers.update(profile_map["construct_gain_modifiers"])
            break

    if overrides:
        if "construct_gain_modifiers" in overrides:
            construct_modifiers.update(overrides["construct_gain_modifiers"])
        base.update(
            {k: v for k, v in overrides.items() if k != "construct_gain_modifiers"}
        )

    for trait, delta in _trait_modifiers(student).items():
        base[trait] = max(0.05, min(0.99, base.get(trait, 0.5) + delta))

    return LearningProfile(
        gain_rate=base["gain_rate"],
        slip_rate=base["slip_rate"],
        retention=base["retention"],
        scaffold_sensitivity=base["scaffold_sensitivity"],
        forget_rate=base["forget_rate"],
        max_step_gain=int(base.get("max_step_gain", 1)),
        construct_gain_modifiers=construct_modifiers,
    )
