"""Task routing and answers must come from metadata, not scenario hardcoding."""

from __future__ import annotations

import pytest

from app.constructs import load_task_metadata
from app.mistakes import match_lp_constructs_by_keywords
from app.task_answer import ANSWER_STRATEGIES, compute_expected_answer
from app.task_tagger import rule_classify
from app.turn_classifier import SOCIAL_PATTERNS, _has_math_content, _heuristic_classify
from eval.deterministic_checks import check_expert_slip, check_persona_direction


def test_families_live_in_metadata_not_python():
    source = open("app/task_tagger.py", encoding="utf-8").read()
    assert "FAMILY_CONSTRUCTS" not in source
    assert "FAMILY_TO_BANK_ID" not in source
    families = load_task_metadata().get("families") or {}
    assert "linear_compare" in families
    assert families["linear_compare"]["bank_task_id"] == "phone_plans_linear_01"


@pytest.mark.parametrize(
    "text,family",
    [
        (
            "Plan A charges a $20 fee plus $0.10 per text. Plan B charges $0.30 per text. When is each better?",
            "linear_compare",
        ),
        ("Add fractions with common denominators: 2/5 + 1/5", "frac_add_common"),
        ("Which is larger: 1/4 or 1/3?", "frac_compare"),
        ("There are 2 red marbles and 5 blue marbles. Write the ratio of red to blue.", "ratio_identify"),
        ("A car travels 150 miles in 3 hours. What is the unit rate?", "unit_rate"),
    ],
)
def test_rule_classify_uses_metadata_families(text, family):
    result = rule_classify(text)
    assert result.task_family == family
    assert result.constructs
    assert result.confidence >= 0.5


def test_bank_expected_answer_wins_over_strategies():
    meta = {"expected_answer": "BANKS_WIN", "answer_strategy": "unit_rate"}
    assert (
        compute_expected_answer(
            "A car travels 150 miles in 3 hours. What is the unit rate?",
            task_meta=meta,
        )
        == "BANKS_WIN"
    )


@pytest.mark.parametrize("strategy", sorted(ANSWER_STRATEGIES))
def test_every_answer_strategy_is_callable(strategy):
    assert callable(ANSWER_STRATEGIES[strategy])


def test_strategy_registry_covers_bank_answer_strategies():
    strategies = {
        t.get("answer_strategy")
        for t in load_task_metadata()["tasks"]
        if t.get("answer_strategy")
    }
    # ratio_equivalence has no symbolic solver yet — that's fine if absent.
    missing = strategies - set(ANSWER_STRATEGIES) - {"ratio_equivalence"}
    assert not missing, f"bank strategies without registry entry: {missing}"


def test_mistakes_reads_kg_cues_not_a_local_dict():
    import app.mistakes as m

    assert not hasattr(m, "LP_CONSTRUCT_KEYWORDS")
    hits = match_lp_constructs_by_keywords(
        "find a common denominator then subtract",
        ["fraction_equivalence", "fraction_operations"],
    )
    assert "fraction_equivalence" in hits
    assert "fraction_operations" in hits


def test_turn_classifier_has_no_demo_roster_names():
    blob = " ".join(SOCIAL_PATTERNS).lower()
    for name in ("jordan", "alex", "maya", "riley"):
        assert name not in blob


def test_math_detection_uses_structure_and_cues_not_fraction_literals():
    assert _has_math_content("set them equal and solve for x")
    assert _has_math_content("which plan is the better deal")
    assert not _has_math_content("how are you feeling today")
    assert _heuristic_classify("okay") == "vague_acknowledgment"
    assert _heuristic_classify("correct") == "vague_acknowledgment"
    assert _heuristic_classify("go on") == "vague_directive"


def test_expert_slip_uses_detection_patterns_not_task_id():
    turn = {
        "behavior_mode": "NORMAL_ERROR_PROFILE",
        "student_stack_level": 3,
        "task_id": "totally_new_task",
        "construct_id": "unit_rate",
        "typical_errors": [
            {
                "error": "writes 10x instead of 0.1x",
                "construct": "unit_rate",
                "stack_level": 1,
                "detection_patterns": [r"\b20\s*\+\s*10\s*x\b"],
            }
        ],
        "reply": "Plan A is just 20 + 10x",
    }
    result = check_expert_slip(turn)
    assert result["pass"] is False


def test_expert_slip_passes_when_no_l1_pattern_matches():
    turn = {
        "behavior_mode": "NORMAL_ERROR_PROFILE",
        "student_stack_level": 3,
        "construct_id": "unit_rate",
        "typical_errors": [
            {
                "error": "writes 10x",
                "construct": "unit_rate",
                "stack_level": 1,
                "detection_patterns": [r"\b20\s*\+\s*10\s*x\b"],
            }
        ],
        "reply": "Plan A is 20 plus 0.1x because each text costs a dime.",
    }
    assert check_expert_slip(turn)["pass"] is True


def test_jordan_persona_check_is_task_id_free():
    turn = {
        "profile_id": "jordan",
        "behavior_mode": "NORMAL_ERROR_PROFILE",
        "turn": 1,
        "construct_id": "fraction_equivalence",
        "misconception_id": "feq_subtract_across_bar",
        "reply": "5 minus 1 is 4, 6 minus 3 is 3, so 4/3",
    }
    result = check_persona_direction(turn)
    assert result["pass"] is False
