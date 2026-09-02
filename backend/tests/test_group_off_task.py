"""Off-task turns must not be scored as failed math turns."""

from app.expected_behavior import ROLE_CLAUSES
from eval.group_quality import collect_student_turns, is_off_task
from eval.group_script import mark_off_task_turns


def _export(records):
    return {
        "session_id": "s1",
        "profile_ids": ["jordan"],
        "task_text": "Which is larger: 1/4 or 1/3?",
        "task_metadata": {"required_constructs": ["part_whole"]},
        "students": {"jordan": {"eval_log": records}},
        "transcript": [],
    }


def test_classifier_label_marks_a_turn_off_task():
    assert is_off_task({"teacher_label": "social"})
    assert is_off_task({"teacher_label": "off_topic"})
    assert not is_off_task({"teacher_label": "math_scaffold"})


def test_explicit_runner_flag_overrides_a_misclassified_label():
    """The classifier defaults unknown prose to math_scaffold; intent must win."""
    record = {"teacher_label": "math_scaffold", "is_off_task": True}
    assert is_off_task(record)


def test_script_intent_is_stamped_onto_the_right_turn():
    script = [
        {"label": "peer_prompt", "text": "..."},
        {"label": "off_task", "text": "..."},
    ]
    # Opener is turn 1, so script[0] is turn 2 and script[1] is turn 3.
    export = _export([{"turn": 2}, {"turn": 3}, {"turn": 4}])
    mark_off_task_turns(export, script)
    marked = {
        r["turn"] for r in export["students"]["jordan"]["eval_log"] if r.get("is_off_task")
    }
    assert marked == {3}


def test_no_off_task_step_marks_nothing():
    export = _export([{"turn": 2}, {"turn": 3}])
    mark_off_task_turns(export, [{"label": "math_scaffold", "text": "..."}])
    assert all(
        not r.get("is_off_task") for r in export["students"]["jordan"]["eval_log"]
    )


def test_off_task_turns_get_the_off_task_expectation():
    export = _export(
        [
            {
                "turn": 2,
                "teacher_label": "math_scaffold",
                "teacher_message": "First step?",
                "reply": "Find a common denominator?",
                "construct_id": "part_whole",
                "mastery": 2,
                "misconception_id": "pw_larger_denominator_larger_fraction",
                "behavior_mode": "WRONG",
            },
            {
                "turn": 3,
                "teacher_label": "math_scaffold",
                "is_off_task": True,
                "teacher_message": "Is everyone coming to the assembly?",
                "reply": "Oh yeah, is it last period?",
                "construct_id": "part_whole",
                "mastery": 2,
                "misconception_id": "pw_larger_denominator_larger_fraction",
                "behavior_mode": "WRONG",
            },
        ]
    )
    turns = collect_student_turns(export)
    by_turn = {t["turn"]: t["expected_behavior"] for t in turns}

    assert ROLE_CLAUSES["off_task"] in by_turn[3]
    assert "knowledge state" not in by_turn[3].lower()
    # The genuine math turn keeps its full expectation.
    assert "knowledge state" in by_turn[2].lower()


def test_off_task_turns_are_counted_but_not_cognitively_scored():
    from eval.group_quality import score_group_quality

    export = _export(
        [
            {"turn": 2, "teacher_label": "math_scaffold", "reply": "um ok"},
            {"turn": 3, "teacher_label": "social", "reply": "yeah I'm going"},
        ]
    )
    report = score_group_quality(export, skip_llm=True, include_deterministic=False)
    assert report["n_student_turns"] == 2
    assert report["n_off_task_turns"] == 1
