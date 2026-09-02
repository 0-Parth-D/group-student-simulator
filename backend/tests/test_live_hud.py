"""Tests for live HUD payload builders."""

from app.group_sessions import GroupSession, StudentSlot
from app.live_hud import (
    build_live_hud_payload,
    build_roster_entry,
    build_speak_events,
)
from app.models import Student


def _student(name: str) -> Student:
    return Student(
        student_id=name,
        Openness="High",
        Conscientiousness="Low",
        Extraversion="Low",
        Agreeableness="High",
        Neuroticism="Low",
    )


def _group() -> GroupSession:
    ids = ["alex", "maya", "jordan"]
    slots = {
        pid: StudentSlot(
            pid,
            _student(pid.capitalize()),
            mastery_state={"unit_rate": 2, "linear_relationship": 1},
            behavior_profile={
                "behavior_mode": "NORMAL_ERROR_PROFILE",
                "primary_construct": "unit_rate",
                "student_stack_level": 2,
                "target_stack_level": 3,
            },
            receptivity=0.55,
        )
        for pid in ids
    }
    g = GroupSession(
        session_id="live-test",
        task_text="Phone plans task",
        task_metadata={
            "task_id": "phone_plans_linear_01",
            "required_constructs": ["unit_rate", "linear_relationship"],
            "target_stack_level": 3,
        },
        students=slots,
        profile_ids=ids,
        display_names={"alex": "Alex", "maya": "Maya", "jordan": "Jordan"},
        started=True,
        turn=2,
        last_mode="direct",
        last_teacher_message="Maya, what do you think?",
    )
    g.last_speak_decisions = [
        {
            "profile_id": "maya",
            "will_speak": True,
            "score": 1.0,
            "reason": "must_speak (directly addressed)",
        },
        {
            "profile_id": "alex",
            "will_speak": False,
            "score": 0.2,
            "reason": "not eligible this turn",
        },
    ]
    g.orchestration_log.append(
        {
            "turn": 2,
            "teacher_message": "Maya, what do you think?",
            "mode": "direct",
            "speakers": ["maya"],
        }
    )
    return g


def test_roster_includes_barrier_and_knowledge():
    g = _group()
    g.students["maya"].active_misc_id = "pr_table_missing_warrants"
    g.students["maya"].implied_plan = "B"
    g.students["maya"].claim_id = "plan_b_table"
    row = build_roster_entry(g, "alex")
    assert row["barrier"]["category"] == "epistemic"
    assert row["knowledge"]["task_construct_levels"]["unit_rate"] == 2
    assert row["personality"]["Openness"] == "High"
    assert "trait_meanings" in row["personality"]
    assert row["personality"]["trait_meanings"]["Extraversion"]["label"] == "Quiet"
    assert row["personality"]["combination_summary"]
    maya = build_roster_entry(g, "maya")
    assert maya["active_misc_id"] == "pr_table_missing_warrants"
    assert maya["implied_plan"] == "B"
    assert maya["claim_id"] == "plan_b_table"


def test_speak_events_primary_and_continuation():
    g = _group()
    primary = [
        {
            "speaker_id": "maya",
            "content": "I think Plan B is cheaper at first.",
            "source": "primary",
            "speak_reason": "must_speak (directly addressed)",
        }
    ]
    multi = [
        {
            "speaker_id": "jordan",
            "content": "Set them equal.",
            "source": "game_flow",
            "speak_reason": "game orchestrator → claim: adds procedure",
            "orchestrator": {
                "expected_move": "claim",
                "responding_to": "Maya",
                "context": "adds procedure",
            },
        }
    ]
    events = build_speak_events(g, primary, multi)
    assert len(events) == 2
    assert events[0]["profile_id"] == "maya"
    assert "must_speak" in events[0]["speak_reason"]
    assert events[1]["source"] == "game_flow"


def test_live_hud_payload_shape():
    g = _group()
    payload = build_live_hud_payload(g, [], [])
    assert payload["session_id"] == "live-test"
    assert len(payload["roster"]) == 3
    assert "task" in payload
    assert "current_turn" in payload
    assert "turn_history" in payload
