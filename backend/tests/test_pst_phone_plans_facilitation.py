"""PST phone-plans teacher script fixture."""

from eval.fixtures.pst_phone_plans_facilitation import (
    load_pst_phone_plans_facilitation,
    teacher_messages,
    teacher_script_steps,
)


def test_teacher_script_has_twelve_turns():
    fixture = load_pst_phone_plans_facilitation()
    messages = teacher_messages(fixture)
    steps = teacher_script_steps(fixture)
    assert len(messages) == 12
    assert len(steps) == 12
    assert [step["message"] for step in steps] == messages


def test_teacher_script_includes_comprehension_and_clarify_turns():
    messages = teacher_messages()
    assert "Just say it" in messages
    assert any("understant what Maya pointed out" in msg for msg in messages)


def test_fixture_metadata_matches_replay_roster():
    fixture = load_pst_phone_plans_facilitation()
    assert fixture["task_id"] == "phone_plans_linear_01"
    assert fixture["profile_ids"] == ["maya", "jordan"]
