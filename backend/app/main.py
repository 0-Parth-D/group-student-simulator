import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv()

from app.constructs import construct_summary
from app.config import CORS_ORIGINS
from app.data import DEFAULT_TASK, PROFILES, get_profile
from app.demo_students import build_lp_student
from app.group_http import (
    group_message_payload,
    stream_group_respond,
    student_slot_summary,
)
from app.group_sessions import (
    DEFAULT_GROUP_PROFILES,
    DEFAULT_GROUP_TASK_ID,
    group_session_store,
)
from app.personality import build_personality_export
from app.pisa import pisa_summary
from app.pst_api import router as pst_router
from app.session_shared import kg_summary
from app.live_hud import build_live_hud_payload, build_roster_entry, build_task_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Group Student Simulator",
    description="LP-based group student simulation for PST training (ArguMath)",
    version="0.3.0",
)

_cors_origins = (
    ["*"]
    if not CORS_ORIGINS
    else [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pst_router)


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class CreateGroupSessionRequest(BaseModel):
    profile_ids: Optional[list[str]] = None
    task_text: Optional[str] = None
    task_id: Optional[str] = None
    orchestration: str = "teacher_directed_self_select"
    enable_self_select: Optional[bool] = None


def _student_slot_summary(group, profile_id: str) -> dict:
    return student_slot_summary(group, profile_id)


@app.get("/api/health")
def health():
    has_key = bool(os.getenv("TAMU_CHAT_API_KEY"))
    from app.kg_config import load_knowledge_graph
    from app import kg_store
    from app.config import USE_QDRANT

    try:
        load_knowledge_graph()
        kg_ok = True
    except Exception:
        kg_ok = False

    return {
        "status": "ok",
        "tamu_configured": has_key,
        "lp_stack": {
            "knowledge_graph_yaml": kg_ok,
            "neo4j": kg_store.neo4j_enabled(),
            "qdrant_enabled": USE_QDRANT,
        },
    }


@app.on_event("startup")
def warm_misconception_indexes():
    from app.config import USE_QDRANT

    if not USE_QDRANT:
        return
    try:
        from app.misconception_store import (
            _ensure_catalog_index,
            _get_qdrant_client,
            build_instance_index,
        )

        client = _get_qdrant_client()
        if client:
            build_instance_index(client)
            _ensure_catalog_index(client)
            logger.info("Qdrant misconception indexes ready")
    except Exception as exc:
        logger.warning("Qdrant index warmup skipped: %s", exc)


@app.get("/api/profiles")
def list_profiles():
    rows = []
    for p in PROFILES:
        student = build_lp_student(p["id"])
        rows.append(
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "personality": build_personality_export(student, p["id"]),
            }
        )
    return rows


@app.get("/api/profiles/{profile_id}")
def get_profile_detail(profile_id: str):
    try:
        profile = get_profile(profile_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Profile not found")

    student = build_lp_student(profile_id)
    lp = student.learning_profile
    return {
        "id": profile["id"],
        "name": profile["name"],
        "description": profile["description"],
        "personality": build_personality_export(student, profile_id),
        "default_task": DEFAULT_TASK,
        "kg_concepts": kg_summary(student),
        "kg_node_count": len(student.construct_mastery),
        "construct_levels": construct_summary(student.construct_mastery),
        "pisa_profile": pisa_summary(student.pisa_profile),
        "learning_profile": lp.to_dict() if lp else {},
    }


@app.post("/api/group-sessions")
def create_group_session(body: CreateGroupSessionRequest):
    try:
        group = group_session_store.create(
            profile_ids=body.profile_ids or DEFAULT_GROUP_PROFILES,
            task_text=body.task_text,
            task_id=body.task_id or DEFAULT_GROUP_TASK_ID,
            orchestration=body.orchestration,
            enable_self_select=body.enable_self_select,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "session_id": group.session_id,
        "profile_ids": group.profile_ids,
        "orchestration": group.orchestration,
        "enable_self_select": group.enable_self_select,
        "task_text": group.task_text,
        "task_metadata": {
            "task_id": group.task_metadata.get("task_id"),
            "constructs": group.task_metadata.get(
                "required_constructs", group.task_metadata.get("constructs", [])
            ),
            "pisa_attributes": group.task_metadata.get("pisa_attributes", []),
        },
        "task_resolution": group.task_resolution,
        "expected_answer": group.task_metadata.get("expected_answer"),
        "students": [
            _student_slot_summary(group, pid) for pid in group.profile_ids
        ],
        "live_hud": build_live_hud_payload(group),
    }


@app.post("/api/group-sessions/{session_id}/start")
def start_group_session(session_id: str):
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")

    group, replies, vague_warning, active, turn_mode = group_session_store.start(
        group
    )
    return {
        "session_id": group.session_id,
        "replies": replies,
        "transcript": group.transcript,
        "turn": group.turn,
        "vague_warning": vague_warning,
        "active_concepts": active,
        "turn_mode": turn_mode,
        "speakers_considered": group.profile_ids,
        "observers": group.last_observers,
        "speak_decisions": group.last_speak_decisions,
        "speak_reasons": {
            decision["profile_id"]: decision["reason"]
            for decision in group.last_speak_decisions
        },
        "students": [
            _student_slot_summary(group, pid) for pid in group.profile_ids
        ],
        "live_hud": build_live_hud_payload(
            group, replies, [], stop_reason=""
        ),
    }


def _group_message_payload(
    group,
    replies,
    multi_round_replies,
    vague_warning,
    active,
    turn_mode,
    stop_reason,
):
    return group_message_payload(
        group,
        replies,
        multi_round_replies,
        vague_warning,
        active,
        turn_mode,
        stop_reason,
    )


@app.post("/api/group-sessions/{session_id}/message")
def send_group_message(
    session_id: str,
    body: MessageRequest,
    stream: bool = Query(
        False,
        description="If true, stream NDJSON events as each student reply finalizes",
    ),
):
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")

    teacher_msg = body.message.strip()

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
        return _group_message_payload(
            group,
            replies,
            multi_round_replies,
            vague_warning,
            active,
            turn_mode,
            stop_reason,
        )

    return stream_group_respond(group, teacher_msg)


@app.post("/api/group-sessions/{session_id}/advance")
def advance_group_session(session_id: str):
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")
    try:
        group, multi_round_replies, stop_reason = group_session_store.advance(group)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "session_id": group.session_id,
        "replies": multi_round_replies,
        "multi_round_replies": multi_round_replies,
        "stop_reason": stop_reason,
        "transcript": group.transcript,
        "turn": group.turn,
        "teaching_warning": group.last_teaching_warning,
        "pedagogical_move": group.last_pedagogical_move,
        "learning_events": group.last_learning_events,
        "observers": group.last_observers,
        "speak_decisions": group.last_speak_decisions,
        "speak_reasons": {
            decision["profile_id"]: decision["reason"]
            for decision in group.last_speak_decisions
        },
        "students": [
            _student_slot_summary(group, pid) for pid in group.profile_ids
        ],
        "live_hud": build_live_hud_payload(
            group, [], multi_round_replies, stop_reason=stop_reason or ""
        ),
    }


@app.get("/api/group-sessions/{session_id}/live")
def get_group_session_live(session_id: str):
    """Live HUD snapshot: roster, current-turn speak reasons, metrics, history."""
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")
    return build_live_hud_payload(group)


@app.get("/api/group-sessions/{session_id}/roster")
def get_group_session_roster(session_id: str):
    """Student personalities + task knowledge levels for the active group session."""
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")
    return {
        "session_id": group.session_id,
        "turn": group.turn,
        "task": build_task_summary(group),
        "roster": [build_roster_entry(group, pid) for pid in group.profile_ids],
    }


@app.get("/api/group-sessions/{session_id}/export")
def export_group_session(session_id: str):
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")
    return group_session_store.export(group)


@app.delete("/api/group-sessions/{session_id}")
def delete_group_session(session_id: str):
    group_session_store.delete(session_id)
    return {"deleted": session_id}
