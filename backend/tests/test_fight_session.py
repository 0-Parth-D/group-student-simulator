"""Frozen FE openers and claim-lock critic for phone-plans fight."""

from unittest.mock import patch

from app.fight_stance import FE_SEED_LINES, MAYA_TABLE_ID
from app.group_sessions import FE_STARTER_PROFILES, group_session_store
from app.kg_config import load_misconception_catalog
from app.misconception_store import _all_misconceptions
from app.models import Student
from app.response_refinement import check_claim_consistency, critique_student_reply


def setup_function(_):
    load_misconception_catalog.cache_clear()
    _all_misconceptions.cache_clear()


def teardown_function(_):
    load_misconception_catalog.cache_clear()
    _all_misconceptions.cache_clear()


def test_fe_starter_start_uses_frozen_seeds_no_llm():
    with patch.object(
        group_session_store, "student_reply", side_effect=AssertionError("no LLM")
    ):
        group = group_session_store.create(
            profile_ids=list(FE_STARTER_PROFILES),
            task_id="phone_plans_linear_01",
        )
        group, replies, *_ = group_session_store.start(group)

    assert {r["speaker_id"] for r in replies} == {"maya", "jordan"}
    by_id = {r["speaker_id"]: r["content"] for r in replies}
    assert by_id["maya"] == FE_SEED_LINES["maya"]
    assert by_id["jordan"] == FE_SEED_LINES["jordan"]
    assert group.students["maya"].active_misc_id == MAYA_TABLE_ID
    assert group.students["maya"].implied_plan.upper() == "B"
    assert group.students["jordan"].implied_plan.upper() == "A"
    assert group.students["maya"].misc_pinned is True


def test_claim_consistency_rejects_yeah_same_and_flip():
    fail = check_claim_consistency(
        "Yeah same, Plan B is better.",
        "A",
        is_peer_turn=True,
    )
    assert fail is not None
    assert fail["ok"] is False
    assert "yeah_same" in fail["issues"] or "claim_flip" in fail["issues"]

    ok = check_claim_consistency(
        "No — Plan A is still better because ten cents is less.",
        "A",
        is_peer_turn=True,
    )
    assert ok is None


def test_critique_claim_short_circuits_llm():
    student = Student(
        student_id="Jordan",
        Openness="High",
        Conscientiousness="Low",
        Extraversion="High",
        Agreeableness="Low",
        Neuroticism="Low",
    )
    with patch("app.response_refinement.complete") as mocked:
        out = critique_student_reply(
            "Yeah same.",
            {
                "behavior_mode": "NORMAL_ERROR_PROFILE",
                "primary_construct": "rate_comparison",
                "student_stack_level": 3,
                "implied_plan": "A",
            },
            student=student,
            is_peer_turn=True,
            implied_plan="A",
        )
    assert out["ok"] is False
    mocked.assert_not_called()
