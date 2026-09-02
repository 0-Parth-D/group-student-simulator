"""Tests for game-style flow orchestrator (mirrors pst-training-game)."""

from unittest.mock import patch

import pytest

from app.demo_students import DEMO_STUDENTS
from app.game_flow_orchestrator import (
    ORCHESTRATOR_SYSTEM_2,
    OrchestratorDecision,
    _apply_extraversion_bias,
    _apply_move_bias,
    _apply_speaker_guard,
    build_orchestrator_system,
    count_students_since_teacher,
    parse_orchestrator_response,
    roster_display_names,
    run_game_flow_continuation,
    student_msg_cap_for_roster,
)
from app.group_sessions import (
    FE_STARTER_PROFILES,
    GroupSession,
    GroupSessionStore,
    StudentSlot,
)
from app.models import Student


PROFILE_IDS = ["alex", "maya", "jordan"]
DISPLAY_NAMES = {"alex": "Alex", "maya": "Maya", "jordan": "Jordan"}
FE_IDS = ["maya", "jordan"]
FE_DISPLAY = {"maya": "Maya", "jordan": "Jordan"}


def _student(profile_id: str, extraversion: str = "Low") -> Student:
    return Student(
        student_id=DISPLAY_NAMES.get(profile_id, profile_id.capitalize()),
        Openness="High",
        Conscientiousness="Low",
        Extraversion=extraversion,
        Agreeableness="High",
        Neuroticism="Low",
    )


def _group(profile_ids=None, display_names=None) -> GroupSession:
    ids = list(profile_ids or PROFILE_IDS)
    if display_names is not None:
        names = dict(display_names)
    elif set(ids) == set(PROFILE_IDS):
        names = dict(DISPLAY_NAMES)
    else:
        names = {pid: pid.capitalize() for pid in ids}
    slots = {
        pid: StudentSlot(
            pid,
            _student(pid, "High" if pid == "jordan" else "Low"),
            mastery_state={"part_whole": 3},
            behavior_profile={
                "behavior_mode": "NORMAL_ERROR_PROFILE",
                "primary_construct": "part_whole",
            },
        )
        for pid in ids
    }
    g = GroupSession(
        session_id="gf-test",
        task_text="Compare phone plans.",
        task_metadata={"required_constructs": ["part_whole"]},
        students=slots,
        profile_ids=ids,
        display_names=names,
        started=True,
        turn=1,
    )
    # Prefer Maya as the opening student when on roster (matches guard/bias tests).
    opener = "maya" if "maya" in ids else ids[0]
    opener_name = names[opener]
    g.transcript = [
        {
            "turn": 1,
            "speaker_type": "teacher",
            "speaker_id": "teacher",
            "content": f"{opener_name}, thoughts?",
        },
        {
            "turn": 1,
            "speaker_type": "student",
            "speaker_id": opener,
            "content": "I think Plan B is cheaper at first.",
        },
    ]
    g.last_mode = "direct"
    g.last_teacher_message = f"{opener_name}, thoughts?"
    g.last_observers = []
    g.last_may_speak = list(ids)
    return g


def _fe_group() -> GroupSession:
    return _group(profile_ids=FE_IDS, display_names=FE_DISPLAY)


def test_count_students_since_teacher():
    g = _group()
    assert count_students_since_teacher(g.transcript) == 1
    g.transcript.append(
        {
            "turn": 1,
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "Not really, look at the equation.",
        }
    )
    assert count_students_since_teacher(g.transcript) == 2


def test_parse_orchestrator_continue():
    raw = """{"action": "continue", "next_speaker": "Alex", "responding_to": "Maya",
    "expected_move": "asking", "context": "confused by Jordan"}"""
    d = parse_orchestrator_response(raw)
    assert d.action == "continue"
    assert d.next_speaker == "Alex"
    assert d.expected_move == "asking"


def test_parse_orchestrator_stop_on_bad_json():
    assert parse_orchestrator_response("not json").action == "stop"


def test_parse_rejects_alex_when_not_on_roster():
    raw = """{"action": "continue", "next_speaker": "Alex", "responding_to": "Maya",
    "expected_move": "claim", "context": "oops"}"""
    d = parse_orchestrator_response(raw, valid_speakers=("Maya", "Jordan"))
    assert d.action == "stop"
    assert d.next_speaker is None


def test_parse_accepts_maya_jordan_on_fe_roster():
    raw = """{"action": "continue", "next_speaker": "Jordan", "responding_to": "Maya",
    "expected_move": "claim", "context": "defends rates"}"""
    d = parse_orchestrator_response(raw, valid_speakers=("Maya", "Jordan"))
    assert d.action == "continue"
    assert d.next_speaker == "Jordan"


def test_fe_roster_system_and_cap():
    g = _fe_group()
    assert roster_display_names(g) == ("Maya", "Jordan")
    assert student_msg_cap_for_roster(g) == 3
    assert build_orchestrator_system(g) == ORCHESTRATOR_SYSTEM_2
    assert "Alex" not in build_orchestrator_system(g)


def test_apply_move_bias_nudges_evidence_for_maya_critique():
    g = _fe_group()
    g.transcript.extend(
        [
            {
                "speaker_type": "teacher",
                "content": "Maya find what are the wrong things in Jordan's claim",
            },
            {
                "speaker_type": "student",
                "speaker_id": "jordan",
                "content": "Plan A is 0.10 per text so it's better.",
            },
        ]
    )
    decision = OrchestratorDecision(
        "continue", "Maya", "Jordan", "claim", "react"
    )
    biased = _apply_move_bias(g, decision)
    assert biased.expected_move == "evidence"


def test_speaker_guard_blocks_back_to_back():
    g = _group()
    # Maya just spoke; orchestrator wrongly picks Maya again.
    decision = OrchestratorDecision("continue", "Maya", "group", "claim", "again")
    out = _apply_speaker_guard(g, decision)
    assert out.next_speaker != "Maya"
    # Prefer High-E Jordan over Low-E Alex when swapping.
    assert out.next_speaker == "Jordan"


def test_speaker_guard_allows_addressed_back_to_back():
    g = _group()
    decision = OrchestratorDecision("continue", "Maya", "Maya", "claim", "answering Maya")
    out = _apply_speaker_guard(g, decision)
    assert out.next_speaker == "Maya"


def test_speaker_guard_keeps_different_speaker():
    g = _group()
    # Maya spoke; orchestrator correctly picks Jordan — must NOT steal to Alex.
    decision = OrchestratorDecision("continue", "Jordan", "Maya", "claim", "corrects")
    out = _apply_speaker_guard(g, decision)
    assert out.next_speaker == "Jordan"


def test_fe_guard_never_introduces_alex():
    g = _fe_group()
    decision = OrchestratorDecision("continue", "Maya", "group", "claim", "again")
    out = _apply_speaker_guard(g, decision)
    assert out.next_speaker == "Jordan"
    assert out.next_speaker != "Alex"


def test_extraversion_bias_swaps_quiet_debater_to_jordan():
    g = _group()
    g.transcript = [
        {
            "turn": 1,
            "speaker_type": "teacher",
            "speaker_id": "teacher",
            "content": "Discuss your reasonings.",
        },
        {
            "turn": 1,
            "speaker_type": "student",
            "speaker_id": "maya",
            "content": "Plan A has the fee.",
        },
    ]
    decision = OrchestratorDecision("continue", "Alex", "group", "claim", "debate")
    out = _apply_extraversion_bias(g, decision)
    assert out.next_speaker == "Jordan"


def test_extraversion_bias_disabled_on_fe_two_student():
    g = _fe_group()
    decision = OrchestratorDecision("continue", "Maya", "group", "claim", "debate")
    out = _apply_extraversion_bias(g, decision)
    assert out.next_speaker == "Maya"


def test_extraversion_bias_keeps_asking_quiet_student():
    g = _group()
    decision = OrchestratorDecision("continue", "Alex", "Maya", "asking", "confused")
    out = _apply_extraversion_bias(g, decision)
    assert out.next_speaker == "Alex"


def test_wrong_mode_low_e_is_quietly_wrong():
    from app.personality import build_personality_expression_block

    alex = _student("alex", "Low")
    block = build_personality_expression_block(
        alex,
        {"behavior_mode": "WRONG", "help_seek_style": "one_word_silence"},
        "math_scaffold",
    )
    assert "quietly wrong" in block.lower()
    assert "Do NOT dominate" in block
    assert "You can be talkative" not in block


def test_wrong_mode_high_e_stays_talkative_confident():
    from app.personality import build_personality_expression_block

    jordan = _student("jordan", "High")
    block = build_personality_expression_block(
        jordan,
        {"behavior_mode": "WRONG", "help_seek_style": "talkative_confident"},
        "math_scaffold",
    )
    assert "confidence" in block.lower() or "talkative" in block.lower()


def test_jordan_phone_plans_not_full_correct_crossover():
    pp = DEMO_STUDENTS["jordan"]["past_performance"]
    phone = next(p for p in pp if p["task_id"] == "phone_plans_linear_01")
    assert phone["reply_quality"] != "correct"
    assert "x = 100" not in phone["student_reply"]
    assert "fee" in phone["observed_error"].lower() or "rate" in phone["student_reply"].lower()


def test_maya_phone_plans_table_partial():
    pp = DEMO_STUDENTS["maya"]["past_performance"]
    phone = next(p for p in pp if p["task_id"] == "phone_plans_linear_01")
    assert phone["reply_quality"] == "partial"
    assert "table" in phone["student_reply"].lower()


def test_create_fe_starter_two_students():
    store = GroupSessionStore()
    group = store.create(profile_ids=FE_STARTER_PROFILES, task_id="phone_plans_linear_01")
    assert group.profile_ids == ["maya", "jordan"]
    assert set(group.display_names.values()) == {"Maya", "Jordan"}


def test_create_rejects_one_or_four_profiles():
    store = GroupSessionStore()
    with pytest.raises(ValueError, match="2–3"):
        store.create(profile_ids=["maya"])
    with pytest.raises(ValueError, match="2–3"):
        store.create(profile_ids=["alex", "maya", "jordan", "riley"])


@patch("app.game_flow_orchestrator.decide_orchestrator_turn")
@patch.object(GroupSessionStore, "student_reply")
def test_run_game_flow_continuation_stops_on_orchestrator(mock_reply, mock_decide):
    mock_decide.side_effect = [
        OrchestratorDecision("continue", "Jordan", "Maya", "claim", "adds math"),
        OrchestratorDecision("stop", None, None, None, "natural pause"),
    ]
    mock_reply.return_value = ("Yeah, x = 100 texts.", False, [], "math_eval", False, False)

    store = GroupSessionStore()
    group = _group()
    store._sessions[group.session_id] = group

    replies, stop = run_game_flow_continuation(
        group, store=store, initial_reply_count=1, max_rounds=4, student_cap=4
    )
    assert len(replies) == 1
    # Guard must keep Jordan (High E) — no longer steal to Alex.
    assert replies[0]["speaker_id"] == "jordan"
    assert stop == "natural pause"
    assert mock_reply.call_count == 1
    assert mock_reply.call_args.kwargs.get("game_flow") is True


@patch("app.game_flow_orchestrator.decide_orchestrator_turn")
@patch.object(GroupSessionStore, "student_reply")
def test_fe_game_flow_never_schedules_alex(mock_reply, mock_decide):
    mock_decide.side_effect = [
        OrchestratorDecision("continue", "Jordan", "Maya", "claim", "reacts"),
        OrchestratorDecision("stop", None, None, None, "pause"),
    ]
    mock_reply.return_value = ("Plan A is better, lower rate.", False, [], "math_eval", False, False)

    store = GroupSessionStore()
    group = _fe_group()
    store._sessions[group.session_id] = group

    replies, stop = run_game_flow_continuation(
        group, store=store, initial_reply_count=1, max_rounds=4, student_cap=4
    )
    assert len(replies) == 1
    assert replies[0]["speaker_id"] == "jordan"
    assert all(r["speaker_id"] != "alex" for r in replies)
