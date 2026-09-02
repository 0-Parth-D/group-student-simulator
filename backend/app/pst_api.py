"""PST training-game connector facade (/api/pst/*).

Stable contract for the frontend switch (Phase 3): FE-starter Maya+Jordan sessions,
one streamed teacher turn, and stubs for audio/board.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.group_http import (
    group_message_payload,
    stream_group_respond,
    student_slot_summary,
)
from app.group_sessions import (
    DEFAULT_GROUP_TASK_ID,
    FE_STARTER_PROFILES,
    group_session_store,
)
from app.live_hud import build_live_hud_payload

router = APIRouter(prefix="/api/pst", tags=["pst-connector"])

_STUB = {"detail": "not implemented", "stub": True}


class CreatePstSessionRequest(BaseModel):
    task_id: Optional[str] = None
    task_text: Optional[str] = None
    profile_ids: Optional[List[str]] = None


class PstTurnRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # FE game context — injected into the teacher message before LP respond.
    board_note: Optional[str] = None
    proximity: Optional[Dict[str, Any]] = None


class PstStubBody(BaseModel):
    """Minimal body so clients can POST without failing validation."""

    pass


def _display_name(group, key: str) -> str:
    """Map profile id or display name → display name."""
    k = (key or "").strip()
    if not k:
        return k
    names = getattr(group, "display_names", None) or {}
    if k in names:
        return names[k]
    lower = k.lower()
    if lower in names:
        return names[lower]
    for pid, name in names.items():
        if str(name).lower() == lower:
            return name
    return k


def enrich_teacher_message(
    message: str,
    *,
    board_note: Optional[str] = None,
    proximity: Optional[Dict[str, Any]] = None,
    group=None,
) -> str:
    """Append blackboard + standing-near cues (parity with FE local transcript/prompts)."""
    text = (message or "").strip()
    parts: List[str] = []
    note = (board_note or "").strip()
    if note:
        parts.append(f"[Also wrote on the blackboard: {note}]")
    if proximity and isinstance(proximity, dict):
        near: List[str] = []
        far: List[str] = []
        for key, val in proximity.items():
            label = _display_name(group, str(key)) if group is not None else str(key)
            v = str(val).lower() if val is not None else ""
            if v == "near":
                near.append(label)
            elif v == "far":
                far.append(label)
        bits: List[str] = []
        if near:
            bits.append(f"Standing near: {', '.join(near)}")
        if far:
            bits.append(f"farther from: {', '.join(far)}")
        if bits:
            parts.append("[" + "; ".join(bits) + "]")
    if not parts:
        return text
    return text + "\n" + "\n".join(parts)


@router.post("/sessions")
def create_pst_session(
    body: Optional[CreatePstSessionRequest] = Body(default=None),
):
    """Create FE-starter group session and auto-start (one round-trip for Briefing Start)."""
    req = body or CreatePstSessionRequest()
    try:
        group = group_session_store.create(
            profile_ids=req.profile_ids or list(FE_STARTER_PROFILES),
            task_text=req.task_text,
            task_id=req.task_id or DEFAULT_GROUP_TASK_ID,
        )
        group, start_replies, vague_warning, active, turn_mode = group_session_store.start(
            group
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "session_id": group.session_id,
        "started": True,
        "profile_ids": group.profile_ids,
        "display_names": dict(group.display_names),
        "task_id": group.task_metadata.get("task_id"),
        "task_text": group.task_text,
        "turn": group.turn,
        "replies": start_replies,
        "vague_warning": vague_warning,
        "active_concepts": active,
        "turn_mode": turn_mode,
        "students": [student_slot_summary(group, pid) for pid in group.profile_ids],
        "live_hud": build_live_hud_payload(
            group, start_replies, [], stop_reason=""
        ),
    }


@router.post("/sessions/{session_id}/turn")
def pst_teacher_turn(
    session_id: str,
    body: PstTurnRequest,
    stream: bool = Query(
        True,
        description="If true (default), stream NDJSON as each student reply finalizes",
    ),
):
    """One teacher turn: LP students + game-flow orch (replaces FE orch/student loop)."""
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")

    teacher_msg = enrich_teacher_message(
        body.message,
        board_note=body.board_note,
        proximity=body.proximity,
        group=group,
    )

    if not stream:
        try:
            (
                group,
                replies,
                vague_warning,
                active,
                turn_mode,
                multi_round_replies,
                stop_reason,
            ) = group_session_store.respond(group, teacher_msg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return group_message_payload(
            group,
            replies,
            multi_round_replies,
            vague_warning,
            active,
            turn_mode,
            stop_reason,
        )

    return stream_group_respond(group, teacher_msg)


@router.delete("/sessions/{session_id}")
def delete_pst_session(session_id: str):
    try:
        group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")
    group_session_store.delete(session_id)
    return {"ok": True, "deleted": session_id}


@router.post("/audio/transcribe")
def pst_audio_transcribe(_body: PstStubBody = PstStubBody()):
    return JSONResponse(status_code=501, content=_STUB)


@router.post("/audio/speech")
def pst_audio_speech(_body: PstStubBody = PstStubBody()):
    return JSONResponse(status_code=501, content=_STUB)


@router.post("/board/analyze")
def pst_board_analyze(_body: PstStubBody = PstStubBody()):
    return JSONResponse(status_code=501, content=_STUB)
