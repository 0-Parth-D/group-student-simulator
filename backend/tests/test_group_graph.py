"""Unit tests for group context trim/wrap and LangGraph peer continuation."""

from __future__ import annotations

from app.group_context import (
    eligible_speakers,
    trim_private_hist,
    wrap_user_message,
)
from app.group_sessions import GroupSession, GroupSessionStore, StudentSlot
from app.models import Student


PROFILE_IDS = ["alex", "maya", "jordan"]
DISPLAY_NAMES = {"alex": "Alex", "maya": "Maya", "jordan": "Jordan"}


def _student(profile_id: str) -> Student:
    return Student(
        student_id=DISPLAY_NAMES[profile_id],
        Openness="High",
        Conscientiousness="Low",
        Extraversion="High",
        Agreeableness="High",
        Neuroticism="Low",
    )


def _group(**kwargs) -> GroupSession:
    slots = {
        pid: StudentSlot(
            pid,
            _student(pid),
            mastery_state={"part_whole": 1},
            behavior_profile={
                "behavior_mode": "WRONG",
                "primary_construct": "part_whole",
            },
        )
        for pid in PROFILE_IDS
    }
    base = dict(
        session_id="test-peer-graph",
        task_text="Maria ate pizza. How much is left?",
        task_metadata={"required_constructs": ["part_whole"]},
        students=slots,
        profile_ids=list(PROFILE_IDS),
        display_names=dict(DISPLAY_NAMES),
        started=True,
        last_may_speak=list(PROFILE_IDS),
        last_mode="discuss",
        last_teacher_message="Discuss together",
        last_observers=[],
        turn=2,
        transcript=[
            {
                "turn": 2,
                "speaker_type": "teacher",
                "speaker_id": "teacher",
                "content": "Discuss together",
            },
            {
                "turn": 2,
                "speaker_type": "student",
                "speaker_id": "jordan",
                "content": "I think half is left",
            },
        ],
    )
    base.update(kwargs)
    return GroupSession(**base)


def test_trim_private_hist_caps_messages():
    slot = StudentSlot("jordan", _student("jordan"))
    slot.student_hist = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    trim_private_hist(slot, max_messages=12)
    assert len(slot.student_hist) == 12
    assert slot.student_hist[0]["content"] == "m8"
    assert slot.student_hist[-1]["content"] == "m19"


def test_wrap_user_message_includes_task_and_multi_round_line():
    group = _group()
    wrapped = wrap_user_message(
        group, "maya", "[Jordan]: half", multi_round=True, transcript_window=10
    )
    assert "Task: Maria ate pizza" in wrapped
    assert "Do not ask the teacher what to do next" in wrapped
    assert "Respond as Maya." in wrapped
    assert "Transcript so far:" in wrapped


def test_wrap_user_message_omits_transcript_when_disabled():
    group = _group()
    wrapped = wrap_user_message(
        group,
        "maya",
        "[Jordan]: half",
        multi_round=True,
        include_transcript=False,
    )
    assert "Transcript so far:" not in wrapped
    assert "Task: Maria ate pizza" in wrapped
    assert "[Jordan]: half" in wrapped


def test_eligible_speakers_excludes_observers():
    assert eligible_speakers(
        PROFILE_IDS, ["alex", "maya", "jordan"], ["jordan"]
    ) == ["alex", "maya"]
    assert eligible_speakers(PROFILE_IDS, None, ["jordan"]) == ["alex", "maya"]


def test_peer_continuation_respects_max_rounds(monkeypatch):
    from app import group_graph
    from app.group_speak_policy import SpeakDecision

    group = _group()
    store = GroupSessionStore()
    store._sessions[group.session_id] = group
    calls = []

    def always_volunteer(candidates, g, constraints, teacher_message, **_kwargs):
        pid = list(candidates)[0]
        decisions = [
            SpeakDecision(p, p == pid, 0.9 if p == pid else 0.1, "test").to_dict()
            for p in g.profile_ids
        ]
        return [pid], decisions

    def fake_reply(g, profile_id, incoming, source, turn, **_kwargs):
        calls.append(profile_id)
        g.transcript.append(
            {
                "turn": turn,
                "speaker_type": "student",
                "speaker_id": profile_id,
                "content": f"{profile_id} cont",
            }
        )
        return f"{profile_id} cont", False, [], "peer", False, False

    monkeypatch.setattr("app.group_graph.volunteer_select", always_volunteer)
    monkeypatch.setattr(store, "student_reply", fake_reply)
    group_graph._PEER_GRAPH = None

    replies, stop = group_graph.run_peer_continuation(
        group, max_rounds=2, store=store
    )
    assert len(replies) == 2
    assert stop == "max_rounds"
    assert len(calls) == 2


def test_peer_continuation_stops_when_no_volunteers(monkeypatch):
    from app import group_graph
    from app.group_speak_policy import SpeakDecision

    group = _group()
    store = GroupSessionStore()
    store._sessions[group.session_id] = group
    calls = []

    def no_volunteers(candidates, g, constraints, teacher_message, **_kwargs):
        decisions = [
            SpeakDecision(p, False, 0.1, "quiet").to_dict() for p in g.profile_ids
        ]
        return [], decisions

    def fake_reply(g, profile_id, incoming, source, turn, **_kwargs):
        calls.append(profile_id)
        return f"{profile_id} cont", False, [], "peer", False, False

    monkeypatch.setattr("app.group_graph.volunteer_select", no_volunteers)
    monkeypatch.setattr(store, "student_reply", fake_reply)
    group_graph._PEER_GRAPH = None

    replies, stop = group_graph.run_peer_continuation(
        group, max_rounds=8, store=store
    )
    assert replies == []
    assert stop == "no_volunteers"
    assert calls == []


def test_peer_continuation_never_selects_observers(monkeypatch):
    from app import group_graph
    from app.group_speak_policy import SpeakDecision

    group = _group(
        last_observers=["jordan"],
        last_may_speak=["maya", "alex"],
    )
    store = GroupSessionStore()
    store._sessions[group.session_id] = group

    def volunteers(candidates, g, constraints, teacher_message, **_kwargs):
        assert "jordan" not in candidates
        pid = list(candidates)[0]
        decisions = [
            SpeakDecision(p, p == pid, 0.9 if p == pid else 0.1, "test").to_dict()
            for p in g.profile_ids
        ]
        return [pid], decisions

    def fake_reply(g, profile_id, incoming, source, turn, **_kwargs):
        assert profile_id != "jordan"
        g.transcript.append(
            {
                "turn": turn,
                "speaker_type": "student",
                "speaker_id": profile_id,
                "content": f"{profile_id} cont",
            }
        )
        return f"{profile_id} cont", False, [], "peer", False, False

    monkeypatch.setattr("app.group_graph.volunteer_select", volunteers)
    monkeypatch.setattr(store, "student_reply", fake_reply)
    group_graph._PEER_GRAPH = None

    replies, stop = group_graph.run_peer_continuation(
        group, max_rounds=2, store=store
    )
    assert all(r["speaker_id"] != "jordan" for r in replies)
    assert stop == "max_rounds"
    assert len(replies) == 2


def test_peer_continuation_downweights_last_speaker(monkeypatch):
    from app import group_graph
    from app.group_speak_policy import SpeakDecision

    group = _group()
    store = GroupSessionStore()
    store._sessions[group.session_id] = group
    selected = []

    def fake_volunteer(candidates, g, constraints, teacher_message, **kwargs):
        last = kwargs.get("last_speaker")
        ranked = [pid for pid in ("maya", "alex", "jordan") if pid in candidates]
        if last and last in ranked and len(ranked) > 1:
            ranked = [pid for pid in ranked if pid != last] + [last]
        pick = ranked[:1]
        decisions = [
            SpeakDecision(pid, pid in pick, 0.9 if pid in pick else 0.4, "test").to_dict()
            for pid in g.profile_ids
        ]
        return pick, decisions

    def fake_reply(g, profile_id, incoming, source, turn, **_kwargs):
        selected.append(profile_id)
        g.transcript.append(
            {
                "turn": turn,
                "speaker_type": "student",
                "speaker_id": profile_id,
                "content": f"{profile_id} cont",
            }
        )
        return f"{profile_id} cont", False, [], "peer", False, False

    monkeypatch.setattr("app.group_graph.volunteer_select", fake_volunteer)
    monkeypatch.setattr(store, "student_reply", fake_reply)
    group_graph._PEER_GRAPH = None

    replies, stop = group_graph.run_peer_continuation(
        group, max_rounds=1, store=store
    )
    assert len(replies) == 1
    assert selected[0] != "jordan"
    assert stop == "max_rounds"
