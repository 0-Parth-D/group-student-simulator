import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv()

from app.knowledge.constructs import construct_summary
from app.student.data import DEFAULT_TASK, PROFILES, get_profile
from app.group.sessions import (
    DEFAULT_GROUP_PROFILES,
    DEFAULT_GROUP_TASK_ID,
    group_session_store,
)
from app.knowledge.mistakes import playbook_summary
from app.knowledge.pisa import pisa_summary
from app.student.sessions import kg_summary, session_store
from eval.log_schema import export_session_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(
    title="Middle-School Student Simulator",
    description="LP-based student simulation for teacher training",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    profile_id: str = "jordan"
    task_text: Optional[str] = None
    task_id: Optional[str] = None


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class CreateGroupSessionRequest(BaseModel):
    profile_ids: Optional[list[str]] = None
    task_text: Optional[str] = None
    task_id: Optional[str] = None
    orchestration: str = "teacher_directed_self_select"
    enable_self_select: Optional[bool] = None


def _student_slot_summary(group, profile_id: str) -> dict:
    slot = group.students[profile_id]
    bp = slot.behavior_profile or {}
    return {
        "profile_id": profile_id,
        "display_name": group.display_names.get(profile_id, profile_id),
        "student_id": slot.student.student_id,
        "predicted_behavior": slot.predicted_behavior,
        "mastery_state": slot.mastery_state,
        "receptivity": getattr(slot, "receptivity", None),
        "personality": slot.student.personality_dict(),
        "behavior_profile": {
            "primary_construct": bp.get("primary_construct"),
            "likely_correctness": bp.get("likely_correctness"),
            "behavior_mode": bp.get("behavior_mode"),
            "error_types": bp.get("error_types", [])[:4],
            "student_stack_level": bp.get("student_stack_level"),
            "target_stack_level": bp.get("target_stack_level"),
        },
    }


@app.get("/api/health")
def health():
    has_key = bool(os.getenv("TAMU_CHAT_API_KEY"))
    from app.knowledge.kg_config import load_knowledge_graph
    from app.knowledge import store as kg_store
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
        from app.knowledge.misconception_store import (
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
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
            "personality": {
                "Openness": p["Openness"],
                "Conscientiousness": p["Conscientiousness"],
                "Extraversion": p["Extraversion"],
                "Agreeableness": p["Agreeableness"],
                "Neuroticism": p["Neuroticism"],
            },
        }
        for p in PROFILES
    ]


@app.get("/api/profiles/{profile_id}")
def get_profile_detail(profile_id: str):
    try:
        profile = get_profile(profile_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Profile not found")

    student = session_store.prepare_student(profile_id)
    lp = student.learning_profile
    return {
        "id": profile["id"],
        "name": profile["name"],
        "description": profile["description"],
        "personality": student.personality_dict(),
        "default_task": DEFAULT_TASK,
        "kg_concepts": kg_summary(student),
        "kg_node_count": len(student.construct_mastery),
        "construct_levels": construct_summary(student.construct_mastery),
        "pisa_profile": pisa_summary(student.pisa_profile),
        "learning_profile": lp.to_dict() if lp else {},
    }


@app.post("/api/sessions")
def create_session(body: CreateSessionRequest):
    try:
        session = session_store.create(
            body.profile_id,
            body.task_text,
            task_id=body.task_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Profile not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "session_id": session.session_id,
        "student_id": session.student.student_id,
        "task_text": session.task_text,
        "predicted_behavior": session.predicted_behavior,
        "top_concepts": session.top_p_concepts,
        "personality": session.student.personality_dict(),
        "kg_concepts": kg_summary(session.student),
        "error_playbook": playbook_summary(session.error_playbook),
        "construct_levels": construct_summary(session.student.construct_mastery),
        "pisa_profile": pisa_summary(session.student.pisa_profile),
        "behavior_profile": {
            "primary_construct": session.behavior_profile.get("primary_construct"),
            "likely_correctness": session.behavior_profile.get("likely_correctness"),
            "behavior_mode": session.behavior_profile.get("behavior_mode"),
            "error_types": session.behavior_profile.get("error_types", [])[:4],
            "weak_attributes": session.behavior_profile.get("weak_attributes", []),
            "student_stack_level": session.behavior_profile.get("student_stack_level"),
            "target_stack_level": session.behavior_profile.get("target_stack_level"),
        },
        "task_metadata": {
            "task_id": session.task_metadata.get("task_id"),
            "constructs": session.task_metadata.get(
                "required_constructs", session.task_metadata.get("constructs", [])
            ),
            "pisa_attributes": session.task_metadata.get("pisa_attributes", []),
        },
        "mastery_state": session.mastery_state,
        "task_resolution": session.task_resolution,
        "expected_answer": session.task_metadata.get("expected_answer"),
    }


@app.post("/api/sessions/{session_id}/start")
def start_session(session_id: str):
    try:
        session = session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    session, vague_warning, active, turn_mode = session_store.start(session)
    return {
        "session_id": session.session_id,
        "messages": session.messages,
        "turn": session.turn,
        "vague_warning": vague_warning,
        "active_concepts": active,
        "turn_mode": turn_mode,
        "scaffold_events": session.last_scaffold_events,
        "mastery_state": session.mastery_state,
    }


@app.post("/api/sessions/{session_id}/message")
def send_message(session_id: str, body: MessageRequest):
    try:
        session = session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session, vague_warning, active, turn_mode = session_store.respond(
            session, body.message.strip()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "session_id": session.session_id,
        "messages": session.messages[-2:],
        "turn": session.turn,
        "vague_warning": vague_warning,
        "active_concepts": active,
        "turn_mode": turn_mode,
        "scaffold_events": session.last_scaffold_events,
        "mastery_state": session.mastery_state,
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    session_store.delete(session_id)
    return {"deleted": session_id}


@app.get("/api/sessions/{session_id}/export")
def export_session(session_id: str):
    """Export session as eval scenario JSON (rich turn logs)."""
    try:
        session = session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    expected = {
        rec["turn"]: rec.get("expected_behavior", "")
        for rec in getattr(session, "eval_log", [])
        if rec.get("expected_behavior")
    }
    scenario = export_session_scenario(
        session,
        session.profile_id or session.student.student_id,
        expected_behaviors=expected or None,
    )
    return scenario


# --- Group sessions (Model A: teacher-facilitated 3-student) ---


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
    }


@app.post("/api/group-sessions/{session_id}/message")
def send_group_message(session_id: str, body: MessageRequest):
    try:
        group = group_session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group session not found")

    try:
        (
            group,
            replies,
            vague_warning,
            active,
            turn_mode,
            multi_round_replies,
            stop_reason,
        ) = group_session_store.respond(group, body.message.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "session_id": group.session_id,
        "replies": replies,
        "multi_round_replies": multi_round_replies,
        "stop_reason": stop_reason,
        "transcript": group.transcript,
        "turn": group.turn,
        "vague_warning": vague_warning,
        "teaching_warning": group.last_teaching_warning,
        "teaching_warning_message": (
            "Giving the answer doesn't raise student mastery — try a scaffold or question instead."
            if group.last_teaching_warning
            else ""
        ),
        "pedagogical_move": group.last_pedagogical_move,
        "learning_events": group.last_learning_events,
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
    }


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


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")
