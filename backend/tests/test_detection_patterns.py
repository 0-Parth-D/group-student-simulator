"""Every L1 misconception needs detection_patterns; false positives must stay rare."""

from __future__ import annotations

import re

import pytest

from app.kg_config import load_misconception_catalog
from eval.deterministic_checks import (
    _l1_detection_patterns,
    check_expert_slip,
    check_persona_direction,
)


@pytest.fixture(autouse=True)
def _clear_catalog_cache():
    load_misconception_catalog.cache_clear()
    from eval import deterministic_checks as dc

    dc._misconceptions_by_id.cache_clear()
    yield
    load_misconception_catalog.cache_clear()
    dc._misconceptions_by_id.cache_clear()


def _catalog():
    return load_misconception_catalog().get("misconceptions", [])


def test_every_l1_misconception_has_detection_patterns():
    missing = [
        m["id"]
        for m in _catalog()
        if int(m.get("stack_level", 99)) == 1 and not m.get("detection_patterns")
    ]
    assert not missing, f"L1 missing detection_patterns: {missing}"


def test_l0_shutdown_has_patterns_but_is_not_an_expert_slip():
    shutdown = next(m for m in _catalog() if m["id"] == "ur_no_attempt_shutdown")
    assert shutdown.get("detection_patterns")
    assert int(shutdown["stack_level"]) == 0

    turn = {
        "behavior_mode": "NORMAL_ERROR_PROFILE",
        "student_stack_level": 3,
        "construct_id": "unit_rate",
        "misconception_id": "ur_no_attempt_shutdown",
        "reply": "I don't even know where to start. Can we skip it?",
    }
    # L0 patterns themselves must not be mixed into the L1 bag.
    l0_pats = set(shutdown["detection_patterns"])
    assert l0_pats.isdisjoint(_l1_detection_patterns(turn))
    # Shutdown wording is not an L1 math slip.
    assert check_expert_slip(turn)["pass"] is True


@pytest.mark.parametrize(
    "misc_id",
    [
        m["id"]
        for m in load_misconception_catalog().get("misconceptions", [])
        if int(m.get("stack_level", 99)) == 1
    ],
)
def test_example_wrong_reply_matches_own_patterns(misc_id):
    entry = next(m for m in _catalog() if m["id"] == misc_id)
    reply = (entry.get("example_wrong_reply") or "").lower()
    patterns = entry.get("detection_patterns") or []
    assert patterns
    assert any(re.search(p, reply, re.I) for p in patterns), (
        f"{misc_id} example does not match its patterns: {patterns!r} / {reply!r}"
    )


def test_correct_point_one_x_does_not_match_cents_as_ten_x():
    entry = next(m for m in _catalog() if m["id"] == "ur_cents_vs_dollars_slope")
    reply = "plan a is 20 + 0.10x and plan b is 0.30x"
    assert not any(re.search(p, reply, re.I) for p in entry["detection_patterns"])
    wrong = "plan a is 20 + 10x because each text is 10 cents"
    assert any(re.search(p, wrong, re.I) for p in entry["detection_patterns"])


def test_correct_ratio_does_not_match_reverse_order_literal():
    entry = next(m for m in _catalog() if m["id"] == "rc_reverses_ratio_order")
    correct = "the ratio of red to blue is 2:5"
    assert not any(re.search(p, correct, re.I) for p in entry["detection_patterns"])
    wrong = "the ratio is 5:2 because there are 5 blue and 2 red"
    assert any(re.search(p, wrong, re.I) for p in entry["detection_patterns"])


def test_expert_slip_flags_l1_pattern_on_l3_normal():
    turn = {
        "behavior_mode": "NORMAL_ERROR_PROFILE",
        "student_stack_level": 3,
        "construct_id": "ratio_concept",
        "misconception_id": "rc_reverses_ratio_order",
        "reply": "The ratio is 5:2 because there are 5 blue and 2 red.",
    }
    assert check_expert_slip(turn)["pass"] is False


def test_expert_slip_passes_when_no_l1_pattern_hits():
    turn = {
        "behavior_mode": "NORMAL_ERROR_PROFILE",
        "student_stack_level": 3,
        "construct_id": "ratio_concept",
        "misconception_id": "rc_reverses_ratio_order",
        "reply": "Red to blue is 2:5 — two red for every five blue.",
    }
    assert check_expert_slip(turn)["pass"] is True


def test_jordan_persona_uses_same_l1_patterns():
    turn = {
        "profile_id": "jordan",
        "behavior_mode": "NORMAL_ERROR_PROFILE",
        "turn": 1,
        "construct_id": "unit_rate",
        "misconception_id": "ur_inverts_ratio",
        "reply": "3 divided by 150 equals 0.02.",
    }
    assert check_persona_direction(turn)["pass"] is False
