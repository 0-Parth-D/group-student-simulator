"""Shared HTTP helpers for group-session message payloads and NDJSON streaming."""

from __future__ import annotations

import json
import logging
import queue
import threading

from fastapi.responses import StreamingResponse

from app.group_sessions import group_session_store
from app.live_hud import build_live_hud_payload
from app.personality import build_personality_export

logger = logging.getLogger(__name__)


def student_slot_summary(group, profile_id: str) -> dict:
    slot = group.students[profile_id]
    bp = slot.behavior_profile or {}
    return {
        "profile_id": profile_id,
        "display_name": group.display_names.get(profile_id, profile_id),
        "student_id": slot.student.student_id,
        "predicted_behavior": slot.predicted_behavior,
        "mastery_state": slot.mastery_state,
        "receptivity": getattr(slot, "receptivity", None),
        "personality": build_personality_export(
            slot.student,
            profile_id,
        ),
        "behavior_profile": {
            "primary_construct": bp.get("primary_construct"),
            "likely_correctness": bp.get("likely_correctness"),
            "behavior_mode": bp.get("behavior_mode"),
            "error_types": bp.get("error_types", [])[:4],
            "student_stack_level": bp.get("student_stack_level"),
            "target_stack_level": bp.get("target_stack_level"),
            "active_misc_id": getattr(slot, "active_misc_id", None)
            or bp.get("active_misc_id"),
            "implied_plan": getattr(slot, "implied_plan", None)
            or bp.get("implied_plan"),
        },
    }


def group_message_payload(
    group,
    replies,
    multi_round_replies,
    vague_warning,
    active,
    turn_mode,
    stop_reason,
) -> dict:
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
        "students": [student_slot_summary(group, pid) for pid in group.profile_ids],
        "live_hud": build_live_hud_payload(
            group, replies, multi_round_replies, stop_reason=stop_reason or ""
        ),
    }


def stream_group_respond(group, teacher_msg: str) -> StreamingResponse:
    """NDJSON stream: speak_plan → thinking → reply* → done | error."""
    event_q: queue.Queue = queue.Queue()

    def on_event(evt: dict) -> None:
        event_q.put(evt)

    def worker() -> None:
        try:
            (
                g,
                replies,
                vague_warning,
                active,
                turn_mode,
                multi_round_replies,
                stop_reason,
            ) = group_session_store.respond(group, teacher_msg, on_event=on_event)
            payload = group_message_payload(
                g,
                replies,
                multi_round_replies,
                vague_warning,
                active,
                turn_mode,
                stop_reason,
            )
            event_q.put({"type": "done", **payload})
        except ValueError as exc:
            event_q.put({"type": "error", "detail": str(exc)})
        except Exception as exc:
            logger.exception("stream message failed")
            event_q.put({"type": "error", "detail": str(exc)})
        finally:
            event_q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_iter():
        while True:
            item = event_q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_iter(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
