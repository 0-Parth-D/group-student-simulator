"""Structured live-session payloads for game HUD (speak reasons, metrics, roster)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from app.constructs import construct_summary
from app.demo_students import DEMO_STUDENTS, lp_profile_meta
from app.personality import build_personality_export
from app.pisa import pisa_summary

if TYPE_CHECKING:
    from app.group_sessions import GroupSession

# PST-training-game aligned barriers (phone-plans / linear group task).
PST_GROUP_BARRIERS: Dict[str, dict] = {
    "alex": {
        "category": "epistemic",
        "label": "Rate of change",
        "summary": (
            "Can do arithmetic but treats 10¢/text as a flat +10, not 0.1× texts. "
            "Needs concrete numbers; superficial acceptance when lost."
        ),
        "discourse_role": "superficial_acceptance",
    },
    "maya": {
        "category": "communicative",
        "label": "Missing warrants",
        "summary": (
            "Table-based intuition up to ~50 texts; struggles to explain why patterns "
            "hold or connect table to equations."
        ),
        "discourse_role": "missing_warrants",
    },
    "jordan": {
        "category": "technological",
        "label": "Procedural imitation",
        "summary": (
            "Equation-first and fast; shallow on meaning of rates. Resists tables/graphs "
            "unless pushed; corrects peers quickly."
        ),
        "discourse_role": "procedural_imitation",
    },
}


def _preview(text: str, limit: int = 140) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def _task_relevant_mastery(
    mastery_state: dict, required_constructs: Sequence[str]
) -> dict:
    req = list(required_constructs or [])
    if not req:
        return dict(mastery_state or {})
    return {cid: int(mastery_state.get(cid, 0)) for cid in req if cid in mastery_state}


def build_roster_entry(group: "GroupSession", profile_id: str) -> dict:
    slot = group.students[profile_id]
    bp = slot.behavior_profile or {}
    meta = lp_profile_meta(profile_id)
    demo = DEMO_STUDENTS.get(profile_id, {})
    barrier = PST_GROUP_BARRIERS.get(profile_id, {})
    required = (group.task_metadata or {}).get("required_constructs") or []
    primary = bp.get("primary_construct") or (required[0] if required else None)

    return {
        "profile_id": profile_id,
        "display_name": group.display_names.get(profile_id, profile_id.capitalize()),
        "barrier": {
            "category": barrier.get("category"),
            "label": barrier.get("label"),
            "summary": barrier.get("summary"),
            "discourse_role": barrier.get("discourse_role"),
        },
        "active_misc_id": getattr(slot, "active_misc_id", "")
        or (slot.misconceptions[0].get("id") if slot.misconceptions else ""),
        "implied_plan": getattr(slot, "implied_plan", "")
        or (bp.get("implied_plan") or ""),
        "claim_id": getattr(slot, "claim_id", "") or (bp.get("claim_id") or ""),
        "personality": build_personality_export(
            slot.student,
            profile_id,
            persona_blurb=demo.get("persona") or meta.get("description"),
        ),
        "big_five": demo.get("big_five") or {},
        "knowledge": {
            "task_id": (group.task_metadata or {}).get("task_id"),
            "primary_construct": primary,
            "behavior_mode": bp.get("behavior_mode"),
            "likely_correctness": bp.get("likely_correctness"),
            "student_stack_level": bp.get("student_stack_level"),
            "target_stack_level": bp.get("target_stack_level"),
            "predicted_behavior": slot.predicted_behavior,
            "receptivity": round(float(getattr(slot, "receptivity", 0.5)), 3),
            "mastery_state": dict(slot.mastery_state or {}),
            "task_construct_levels": _task_relevant_mastery(
                slot.mastery_state or {}, required
            ),
            "construct_summary": construct_summary(slot.mastery_state or {}),
            "pisa_profile": pisa_summary(slot.student.pisa_profile),
            "error_types": (bp.get("error_types") or [])[:4],
            "stall_active": bool(getattr(slot, "stall_active", False)),
        },
        "learning_profile": (
            slot.student.learning_profile.to_dict()
            if slot.student.learning_profile
            else meta.get("learning_profile") or {}
        ),
        "fight_progress": {
            "phase": getattr(group, "fight_phase", "fight"),
            "solved_crossover": bool(getattr(slot, "solved_crossover", False)),
            "stated_equations": bool(getattr(slot, "stated_equations", False)),
        },
    }


def build_task_summary(group: "GroupSession") -> dict:
    meta = group.task_metadata or {}
    return {
        "task_id": meta.get("task_id"),
        "problem_type": meta.get("problem_type"),
        "description": group.task_text,
        "required_constructs": meta.get("required_constructs") or [],
        "target_stack_level": meta.get("target_stack_level"),
        "expected_answer": meta.get("expected_answer"),
        "facilitator_solution": meta.get("facilitator_solution"),
    }


def _decision_map(group: "GroupSession") -> Dict[str, dict]:
    return {
        str(d.get("profile_id")): d
        for d in (group.last_speak_decisions or [])
        if d.get("profile_id")
    }


def build_speak_events(
    group: "GroupSession",
    primary_replies: Sequence[dict],
    multi_round_replies: Sequence[dict],
    *,
    stop_reason: str = "",
) -> List[dict]:
    """Ordered speak events for the current teacher turn."""
    decisions = _decision_map(group)
    events: List[dict] = []
    order = 0

    def append_event(reply: dict, default_source: str) -> None:
        nonlocal order
        pid = str(reply.get("speaker_id") or "")
        if not pid:
            return
        order += 1
        decision = decisions.get(pid, {})
        orch = reply.get("orchestrator") or {}
        source = reply.get("source") or default_source
        reason = reply.get("speak_reason") or decision.get("reason") or source
        if orch.get("context"):
            reason = f"{reason}; {orch['context']}" if reason else orch["context"]

        events.append(
            {
                "order": order,
                "profile_id": pid,
                "display_name": group.display_names.get(
                    pid, pid.capitalize()
                ),
                "source": source,
                "will_speak": decision.get("will_speak", True),
                "volunteer_score": decision.get("score"),
                "speak_reason": reason,
                "orchestrator": {
                    "expected_move": orch.get("expected_move")
                    or reply.get("expected_move"),
                    "responding_to": orch.get("responding_to")
                    or reply.get("responding_to"),
                    "context": orch.get("context") or reply.get("context"),
                },
                "help_seeking": bool(reply.get("help_seeking")),
                "is_question": bool(reply.get("is_question")),
                "reply_preview": _preview(reply.get("content") or ""),
            }
        )

    for reply in primary_replies or []:
        append_event(reply, "primary")
    for reply in multi_round_replies or []:
        append_event(reply, "continuation")

    return events


def _last_orchestration_entry(group: "GroupSession") -> dict:
    log = group.orchestration_log or []
    return log[-1] if log else {}


def build_current_turn_insights(
    group: "GroupSession",
    primary_replies: Optional[Sequence[dict]] = None,
    multi_round_replies: Optional[Sequence[dict]] = None,
    *,
    stop_reason: str = "",
) -> dict:
    """Metrics for the latest teacher turn (or empty shell before first message)."""
    orch = _last_orchestration_entry(group)
    primary = list(primary_replies or [])
    multi = list(multi_round_replies or [])

    return {
        "turn": group.turn,
        "teacher_message": orch.get("teacher_message")
        or group.last_teacher_message
        or "",
        "constraint_mode": orch.get("mode") or group.last_mode or "",
        "turn_mode": orch.get("turn_mode"),
        "pedagogical_move": group.last_pedagogical_move or "",
        "teaching_warning": bool(group.last_teaching_warning),
        "vague_warning": bool(orch.get("vague_warning")),
        "observers": list(group.last_observers or []),
        "must_speak": list(orch.get("must_speak") or []),
        "may_speak": list(orch.get("may_speak") or group.last_may_speak or []),
        "stop_reason": stop_reason or orch.get("stop_reason") or "",
        "speak_decisions": list(group.last_speak_decisions or []),
        "speak_reasons": {
            d["profile_id"]: d.get("reason")
            for d in (group.last_speak_decisions or [])
            if d.get("profile_id")
        },
        "speak_events": build_speak_events(
            group, primary, multi, stop_reason=stop_reason
        ),
        "learning_events": list(group.last_learning_events or []),
        "receptivity": {
            pid: round(float(group.students[pid].receptivity), 3)
            for pid in group.profile_ids
            if pid in group.students
        },
        "active_concepts": orch.get("active_concepts"),
    }


def build_turn_history(group: "GroupSession", limit: int = 12) -> List[dict]:
    """Condensed orchestration log for HUD timeline."""
    rows: List[dict] = []
    for entry in (group.orchestration_log or [])[-limit:]:
        rows.append(
            {
                "turn": entry.get("turn"),
                "kind": entry.get("kind") or "teacher_wave",
                "mode": entry.get("mode"),
                "speakers": entry.get("speakers") or [],
                "observers": entry.get("observers") or [],
                "round": entry.get("round"),
                "stop_reason": entry.get("stop_reason"),
                "orchestrator": entry.get("orchestrator"),
                "teacher_message_preview": _preview(
                    entry.get("teacher_message") or "", 100
                ),
            }
        )
    return rows


def build_live_hud_payload(
    group: "GroupSession",
    primary_replies: Optional[Sequence[dict]] = None,
    multi_round_replies: Optional[Sequence[dict]] = None,
    *,
    stop_reason: str = "",
) -> dict:
    """Full live HUD snapshot for game overlay."""
    return {
        "session_id": group.session_id,
        "turn": group.turn,
        "orchestration": group.orchestration,
        "fight_phase": getattr(group, "fight_phase", "fight"),
        "breakthrough_bullets": list(
            getattr(group, "breakthrough_bullets", None) or []
        ),
        "task": build_task_summary(group),
        "roster": [
            build_roster_entry(group, pid) for pid in group.profile_ids
        ],
        "current_turn": build_current_turn_insights(
            group,
            primary_replies,
            multi_round_replies,
            stop_reason=stop_reason,
        ),
        "turn_history": build_turn_history(group),
    }
