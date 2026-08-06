"""Track C — group deterministic checks + hybrid constraint merge."""

from app.group.constraint_llm import merge_constraints_regex_veto
from app.group.orchestrator import SpeakConstraints
from eval.group_deterministic_checks import (
    run_group_deterministic_checks,
    score_group_export,
)


def _sample_export() -> dict:
    return {
        "session_id": "test-export",
        "task_text": "Maria ate 3/8 of a pizza then gave away 1/4. How much is left?",
        "task_metadata": {
            "task_id": "frac_word_maria_pizza_01",
            "expected_answer": "3/8",
        },
        "task_resolution": {"expected_answer": "3/8"},
        "profile_ids": ["jordan", "sam", "alex"],
        "students": {
            "jordan": {"behavior_mode": "WRONG"},
            "sam": {"behavior_mode": "NORMAL_ERROR_PROFILE"},
            "alex": {"behavior_mode": "NORMAL_ERROR_PROFILE"},
        },
        "transcript": [
            {
                "turn": 1,
                "speaker_type": "teacher",
                "speaker_id": "teacher",
                "content": "Jordan, what's your first step?",
            },
            {
                "turn": 1,
                "speaker_type": "student",
                "speaker_id": "jordan",
                "content": "Easy! Just subtract across — 3/8 minus 1/4 is 2/4.",
            },
            {
                "turn": 2,
                "speaker_type": "teacher",
                "speaker_id": "teacher",
                "content": "Alex and Sam discuss; Jordan please watch",
            },
            {
                "turn": 2,
                "speaker_type": "student",
                "speaker_id": "alex",
                "content": "I think we need the whole pizza.",
            },
            {
                "turn": 2,
                "speaker_type": "student",
                "speaker_id": "sam",
                "content": "Yeah maybe start from 8/8?",
            },
        ],
        "orchestration_log": [
            {
                "turn": 1,
                "teacher_message": "Jordan, what's your first step?",
                "mode": "direct",
                "must_speak": ["jordan"],
                "may_speak": [],
                "speakers": ["jordan"],
                "observers": [],
            },
            {
                "turn": 2,
                "teacher_message": "Alex and Sam discuss; Jordan please watch",
                "mode": "discuss",
                "must_speak": [],
                "may_speak": ["alex", "sam"],
                "speakers": ["alex", "sam"],
                "observers": ["jordan"],
            },
        ],
    }


def test_group_export_passes_clean_demo_shape():
    summary = score_group_export(_sample_export())
    assert summary["overall_pass_rate"] == 1.0


def test_observer_speaking_fails():
    export = _sample_export()
    export["transcript"].append(
        {
            "turn": 2,
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "I still think subtract across!",
        }
    )
    results = run_group_deterministic_checks(export)
    obs = [
        r
        for r in results
        if r["check"] == "observers_never_reply" and r.get("turn") == 2
    ]
    assert obs and obs[0]["pass"] is False


def test_wrong_jordan_answer_leak_fails():
    export = _sample_export()
    export["transcript"][1]["content"] = "The answer is 3/8."
    results = run_group_deterministic_checks(export)
    leaks = [r for r in results if r["check"] == "group_early_answer_leak"]
    jordan = next(r for r in leaks if r.get("profile_id") == "jordan")
    assert jordan["pass"] is False


def test_hybrid_regex_veto_keeps_watch():
    llm = SpeakConstraints(
        must_speak=["jordan", "sam"],
        may_speak=[],
        must_not_speak=[],
        mode="direct",
    )
    regex = SpeakConstraints(
        must_speak=[],
        may_speak=["sam", "alex"],
        must_not_speak=["jordan"],
        mode="discuss",
    )
    merged = merge_constraints_regex_veto(llm, regex)
    assert "jordan" in merged.must_not_speak
    assert "jordan" not in merged.must_speak
    assert "jordan" not in merged.may_speak
