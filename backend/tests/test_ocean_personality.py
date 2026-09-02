"""Tests for OCEAN trait meanings and combination summaries."""

from app.demo_students import build_lp_student
from app.personality import (
    build_personality_export,
    build_personality_expression_block,
    ocean_combination_summary,
    ocean_trait_meanings,
)


def test_jordan_normal_voice_does_not_demand_wrong_step():
    student = build_lp_student("jordan")
    text = build_personality_expression_block(
        student,
        {
            "behavior_mode": "NORMAL_ERROR_PROFILE",
            "help_seek_style": "talkative_confident",
            "student_stack_level": 3,
        },
        "math_scaffold",
    )
    assert "commit to a wrong math step" not in text.lower()



def test_ocean_trait_meanings_cover_all_traits():
    student = build_lp_student("alex")
    meanings = ocean_trait_meanings(student)
    assert set(meanings) == {
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    }
    assert meanings["Extraversion"]["level"] == "Low"
    assert meanings["Extraversion"]["label"] == "Quiet"
    assert meanings["Extraversion"]["summary"]
    assert len(meanings["Extraversion"]["behaviors"]) == 3


def test_ocean_combination_summary_uses_authored_copy():
    student = build_lp_student("maya")
    summary = ocean_combination_summary(student, "maya")
    assert "stress-sensitive" in summary.lower()
    assert "organized" in summary.lower()


def test_build_personality_export_shape():
    student = build_lp_student("jordan")
    block = build_personality_export(student, "jordan")
    assert block["Conscientiousness"] == "High"
    assert "trait_meanings" in block
    assert "combination_summary" in block
    assert block["combination_summary"]
