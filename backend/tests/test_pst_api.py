"""Tests for /api/pst/* connector facade (no live LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.group_sessions import FE_STARTER_PROFILES, GroupSession, StudentSlot
from app.main import app
from app.models import Student


client = TestClient(app)


def _slot(pid: str, name: str) -> StudentSlot:
    return StudentSlot(
        pid,
        Student(
            student_id=name,
            Openness="High",
            Conscientiousness="Low",
            Extraversion="High" if pid == "jordan" else "Low",
            Agreeableness="High",
            Neuroticism="Low",
        ),
        mastery_state={"linear_relationship": 2},
        behavior_profile={
            "behavior_mode": "NORMAL_ERROR_PROFILE",
            "primary_construct": "linear_relationship",
        },
    )


def _fake_group(session_id: str = "pst-test-1") -> GroupSession:
    ids = list(FE_STARTER_PROFILES)
    names = {"maya": "Maya", "jordan": "Jordan"}
    return GroupSession(
        session_id=session_id,
        task_text="Compare phone plans.",
        task_metadata={"task_id": "phone_plans_linear_01"},
        students={pid: _slot(pid, names[pid]) for pid in ids},
        profile_ids=ids,
        display_names=names,
        started=True,
        turn=1,
    )


def test_create_pst_session_fe_starter():
    with (
        patch("app.pst_api.group_session_store.create") as mock_create,
        patch("app.pst_api.group_session_store.start") as mock_start,
    ):
        g = _fake_group()
        mock_create.return_value = g
        mock_start.return_value = (g, [], False, [], "math_scaffold")

        res = client.post("/api/pst/sessions", json={})
        assert res.status_code == 200
        data = res.json()
        assert data["started"] is True
        assert data["profile_ids"] == ["maya", "jordan"]
        assert data["display_names"]["maya"] == "Maya"
        assert "session_id" in data
        assert "live_hud" in data
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["profile_ids"] == list(FE_STARTER_PROFILES)
        mock_start.assert_called_once()


def test_turn_non_stream_mocked():
    g = _fake_group()
    with (
        patch("app.pst_api.group_session_store.get", return_value=g),
        patch("app.pst_api.group_session_store.respond") as mock_respond,
    ):
        mock_respond.return_value = (
            g,
            [{"speaker_id": "maya", "content": "Table says Plan B."}],
            False,
            [],
            "math_scaffold",
            [],
            "orchestrator_stop",
        )
        res = client.post(
            f"/api/pst/sessions/{g.session_id}/turn?stream=false",
            json={"message": "What first?"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == g.session_id
        assert data["replies"][0]["speaker_id"] == "maya"
        assert "live_hud" in data


def test_turn_stream_mocked():
    g = _fake_group()

    def fake_respond(group, teacher_msg, *, on_event=None):
        if on_event:
            on_event({"type": "speak_plan", "speakers": ["maya"], "thinking": ["maya"]})
            on_event(
                {
                    "type": "reply",
                    "kind": "primary",
                    "reply": {"speaker_id": "maya", "content": "hi"},
                }
            )
        return (
            group,
            [{"speaker_id": "maya", "content": "hi"}],
            False,
            [],
            "math_scaffold",
            [],
            "done",
        )

    with (
        patch("app.pst_api.group_session_store.get", return_value=g),
        patch("app.group_http.group_session_store.respond", side_effect=fake_respond),
    ):
        res = client.post(
            f"/api/pst/sessions/{g.session_id}/turn",
            json={"message": "Maya, thoughts?"},
        )
        assert res.status_code == 200
        assert "ndjson" in res.headers.get("content-type", "")
        lines = [ln for ln in res.text.strip().split("\n") if ln.strip()]
        events = [json.loads(ln) for ln in lines]
        types = [e.get("type") for e in events]
        assert "speak_plan" in types
        assert "reply" in types
        assert "done" in types


def test_delete_unknown_404():
    with patch(
        "app.pst_api.group_session_store.get",
        side_effect=KeyError("missing"),
    ):
        res = client.delete("/api/pst/sessions/does-not-exist")
        assert res.status_code == 404


def test_delete_ok():
    g = _fake_group()
    with (
        patch("app.pst_api.group_session_store.get", return_value=g),
        patch("app.pst_api.group_session_store.delete") as mock_del,
    ):
        res = client.delete(f"/api/pst/sessions/{g.session_id}")
        assert res.status_code == 200
        assert res.json()["ok"] is True
        mock_del.assert_called_once_with(g.session_id)


def test_turn_injects_board_note_and_proximity():
    g = _fake_group()
    with (
        patch("app.pst_api.group_session_store.get", return_value=g),
        patch("app.pst_api.group_session_store.respond") as mock_respond,
    ):
        mock_respond.return_value = (
            g,
            [{"speaker_id": "maya", "content": "ok"}],
            False,
            [],
            "math_scaffold",
            [],
            "orchestrator_stop",
        )
        res = client.post(
            f"/api/pst/sessions/{g.session_id}/turn?stream=false",
            json={
                "message": "What first?",
                "board_note": 'wrote "Plan A = 25"',
                "proximity": {"Maya": "near", "jordan": "far"},
            },
        )
        assert res.status_code == 200
        mock_respond.assert_called_once()
        sent = mock_respond.call_args.args[1]
        assert "What first?" in sent
        assert '[Also wrote on the blackboard: wrote "Plan A = 25"]' in sent
        assert "Standing near: Maya" in sent
        assert "farther from: Jordan" in sent


def test_enrich_teacher_message_unit():
    from app.pst_api import enrich_teacher_message

    g = _fake_group()
    out = enrich_teacher_message(
        "Hi",
        board_note="drew a line",
        proximity={"maya": "near", "Jordan": "far"},
        group=g,
    )
    assert out.startswith("Hi\n")
    assert "[Also wrote on the blackboard: drew a line]" in out
    assert "Standing near: Maya" in out
    assert "farther from: Jordan" in out


@pytest.mark.parametrize(
    "path",
    [
        "/api/pst/audio/transcribe",
        "/api/pst/audio/speech",
        "/api/pst/board/analyze",
    ],
)
def test_stubs_501(path):
    res = client.post(path, json={})
    assert res.status_code == 501
    body = res.json()
    assert body.get("stub") is True
    assert "not implemented" in body.get("detail", "")
