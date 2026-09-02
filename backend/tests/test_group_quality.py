"""Track C — group quality turn collection + scoring (mocked LLM)."""

from eval.group_quality import collect_student_turns, score_group_quality


def _export_with_eval_logs() -> dict:
    return {
        "session_id": "gq-test",
        "profile_ids": ["alex", "maya", "jordan"],
        "task_text": "Maria ate 3/8 then gave away 1/4. How much is left?",
        "task_metadata": {
            "task_id": "frac_word_maria_pizza_01",
            "expected_answer": "3/8",
            "required_constructs": ["part_whole"],
        },
        "task_resolution": {"expected_answer": "3/8"},
        "students": {
            "jordan": {
                "behavior_mode": "WRONG",
                "mastery_state": {"part_whole": 1},
                "eval_log": [
                    {
                        "turn": 1,
                        "role": "student",
                        "type": "math",
                        "profile_id": "jordan",
                        "reply": "Easy! Subtract across — 2/4.",
                        "behavior_mode": "WRONG",
                        "teacher_message": "Jordan, what's your first step?",
                        "teacher_label": "math_scaffold",
                        "construct_id": "part_whole",
                        "mastery": 1,
                    }
                ],
            },
            "maya": {
                "behavior_mode": "NORMAL_ERROR_PROFILE",
                "mastery_state": {"part_whole": 3},
                "eval_log": [
                    {
                        "turn": 2,
                        "role": "student",
                        "type": "math",
                        "profile_id": "maya",
                        "reply": "I think maybe start from the whole?",
                        "behavior_mode": "NORMAL_ERROR_PROFILE",
                        "teacher_message": "[Alex]: I think we need the whole.",
                        "teacher_label": "math_scaffold",
                        "construct_id": "part_whole",
                        "mastery": 3,
                    }
                ],
            },
            "alex": {
                "behavior_mode": "NORMAL_ERROR_PROFILE",
                "mastery_state": {"part_whole": 3},
                "eval_log": [
                    {
                        "turn": 2,
                        "role": "student",
                        "type": "math",
                        "profile_id": "alex",
                        "reply": "Yeah.",
                        "behavior_mode": "NORMAL_ERROR_PROFILE",
                        "teacher_message": "Alex and Sam discuss; Jordan please watch",
                        "teacher_label": "math_scaffold",
                        "construct_id": "part_whole",
                        "mastery": 3,
                    }
                ],
            },
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
                "content": "Easy! Subtract across — 2/4.",
            },
            {
                "turn": 2,
                "speaker_type": "teacher",
                "speaker_id": "teacher",
                "content": "Alex and Maya discuss; Jordan please watch",
            },
            {
                "turn": 2,
                "speaker_type": "student",
                "speaker_id": "alex",
                "content": "Yeah.",
            },
            {
                "turn": 2,
                "speaker_type": "student",
                "speaker_id": "maya",
                "content": "I think maybe start from the whole?",
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
                "may_speak": ["alex", "maya"],
                "speakers": ["alex", "maya"],
                "observers": ["jordan"],
            },
        ],
    }


def test_collect_marks_peer_turns_from_bracket_incoming():
    turns = collect_student_turns(_export_with_eval_logs())
    by_pid = {t["profile_id"]: t for t in turns}
    assert by_pid["jordan"]["is_peer_turn"] is False
    assert by_pid["maya"]["is_peer_turn"] is True
    assert by_pid["alex"]["is_peer_turn"] is False


def test_score_group_quality_skip_llm_still_has_deterministic():
    report = score_group_quality(_export_with_eval_logs(), skip_llm=True)
    assert report["n_student_turns"] == 3
    assert report["deterministic"]["overall_pass_rate"] == 1.0
    assert report["summary"]["cognitive_mean"] is None


def test_score_group_quality_with_mocked_judges(monkeypatch):
    import eval.group_quality as gq

    monkeypatch.setattr(
        gq,
        "cognitive_fidelity",
        lambda turn: {"score": 4, "violated_constraint": None},
    )
    monkeypatch.setattr(
        gq,
        "persona_stability",
        lambda sampled, profile_id=None: {
            "mean_absolute_deviation": 0.8,
            "profile_id": profile_id,
        },
    )
    monkeypatch.setattr(
        gq,
        "group_discourse_fidelity",
        lambda **kwargs: {
            "score": 5,
            "tutor_like": False,
            "peer_move": "build",
            "rationale": "ok",
            "profile_id": kwargs.get("profile_id"),
        },
    )
    report = score_group_quality(_export_with_eval_logs(), skip_llm=False)
    assert report["summary"]["cognitive_mean"] == 4.0
    assert abs(report["summary"]["persona_mad_mean"] - 0.8) < 1e-6
    assert report["summary"]["discourse_mean"] == 5.0
    assert report["summary"]["tutor_like_rate"] == 0.0
    assert "jordan" in report["per_profile"]
    assert report["per_profile"]["jordan"]["n_turns"] == 1
