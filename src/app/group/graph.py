"""LangGraph peer-continuation loop for discuss/open/critique group turns."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.group.context import eligible_speakers, trim_private_hist
from app.group.orchestrator import SpeakConstraints
from app.group.speak_policy import volunteer_select
from app.knowledge.turn_logic import prepare_peer_user_message

logger = logging.getLogger(__name__)

# Compiled once; nodes resolve the live GroupSession via session_id + active store.
_PEER_GRAPH = None
_ACTIVE_STORE = None


class PeerContinuationState(TypedDict, total=False):
    session_id: str
    eligible: List[str]
    observers: List[str]
    rounds_done: int
    max_rounds: int
    last_speaker: Optional[str]
    selected: Optional[str]
    replies: List[dict]
    stop_reason: str
    empty_selects: int
    mode: str
    teacher_message: str
    speak_decisions: List[dict]


def _store():
    from app.group.sessions import group_session_store

    return _ACTIVE_STORE or group_session_store


def _last_student_utterance(group) -> tuple[Optional[str], Optional[str]]:
    for entry in reversed(group.transcript or []):
        if entry.get("speaker_type") == "student":
            return entry.get("speaker_id"), entry.get("content") or ""
    return None, None


def _select_speaker(state: PeerContinuationState) -> PeerContinuationState:
    if state.get("stop_reason"):
        return state

    rounds_done = int(state.get("rounds_done") or 0)
    max_rounds = int(state.get("max_rounds") or 0)
    if rounds_done >= max_rounds:
        return {**state, "selected": None, "stop_reason": "max_rounds"}

    group = _store().get(state["session_id"])
    observers = list(state.get("observers") or group.last_observers or [])
    eligible = eligible_speakers(
        group.profile_ids,
        state.get("eligible") or group.last_may_speak,
        observers,
    )
    if not eligible:
        return {
            **state,
            "selected": None,
            "eligible": [],
            "stop_reason": "no_eligible",
        }

    mode = state.get("mode") or group.last_mode or "discuss"
    # Use open-style thresholds; never force a participation floor.
    select_mode = "open" if mode == "discuss" else mode
    if select_mode not in ("discuss", "open", "critique"):
        select_mode = "open"
    constraints = SpeakConstraints(
        must_speak=[],
        may_speak=list(eligible),
        must_not_speak=list(observers),
        mode=select_mode,
    )
    last_id, last_content = _last_student_utterance(group)
    teacher_message = (
        state.get("teacher_message")
        or group.last_teacher_message
        or ""
    )
    loop_last = state.get("last_speaker") or last_id
    selected_ids, decisions = volunteer_select(
        eligible,
        group,
        constraints,
        teacher_message,
        last_peer_reply=last_content,
        last_speaker=loop_last,
    )
    selected = selected_ids[0] if selected_ids else None
    if selected in observers:
        selected = None

    if not selected:
        # Natural end: nobody volunteered this round.
        return {
            **state,
            "selected": None,
            "eligible": eligible,
            "observers": observers,
            "speak_decisions": decisions,
            "empty_selects": int(state.get("empty_selects") or 0) + 1,
            "stop_reason": "no_volunteers",
        }

    return {
        **state,
        "selected": selected,
        "eligible": eligible,
        "observers": observers,
        "speak_decisions": decisions,
        "empty_selects": 0,
        "mode": mode,
        "teacher_message": teacher_message,
        "last_speaker": loop_last,
    }


def _generate_reply(state: PeerContinuationState) -> PeerContinuationState:
    selected = state.get("selected")
    if not selected or state.get("stop_reason"):
        return state

    store = _store()
    group = store.get(state["session_id"])
    last_id, last_content = _last_student_utterance(group)
    if last_id and last_content is not None:
        prev_name = group.display_names.get(last_id, last_id.capitalize())
        incoming = prepare_peer_user_message(prev_name, last_content)
    else:
        incoming = prepare_peer_user_message(
            "Peer", group.last_teacher_message or "Continue the discussion."
        )

    reply_text, _, _, _, help_seeking, is_question = store.student_reply(
        group,
        selected,
        incoming,
        "peer",
        group.turn,
        allow_help_seeking=False,
        multi_round=True,
    )
    trim_private_hist(group.students[selected])

    reply = {
        "speaker_id": selected,
        "content": reply_text,
        "help_seeking": help_seeking,
        "is_question": is_question,
        "source": "peer_continuation",
    }
    replies = list(state.get("replies") or [])
    replies.append(reply)
    rounds_done = int(state.get("rounds_done") or 0) + 1

    group.orchestration_log.append(
        {
            "turn": group.turn,
            "kind": "peer_continuation",
            "mode": state.get("mode") or group.last_mode,
            "speakers": [selected],
            "observers": list(state.get("observers") or group.last_observers),
            "speak_decisions": list(state.get("speak_decisions") or []),
            "round": rounds_done,
            "stop_policy": "volunteer_or_cap",
        }
    )
    group.last_speak_decisions = list(state.get("speak_decisions") or [])

    return {
        **state,
        "replies": replies,
        "rounds_done": rounds_done,
        "last_speaker": selected,
        "selected": None,
    }


def _check_stop(state: PeerContinuationState) -> PeerContinuationState:
    if state.get("stop_reason"):
        return state
    rounds_done = int(state.get("rounds_done") or 0)
    max_rounds = int(state.get("max_rounds") or 0)
    if rounds_done >= max_rounds:
        return {**state, "stop_reason": "max_rounds"}
    if not state.get("eligible"):
        return {**state, "stop_reason": "no_eligible"}
    return state


def _route_after_select(state: PeerContinuationState) -> str:
    if state.get("stop_reason") or not state.get("selected"):
        return "stop"
    return "generate"


def _route_after_stop(state: PeerContinuationState) -> str:
    if state.get("stop_reason"):
        return "end"
    rounds_done = int(state.get("rounds_done") or 0)
    max_rounds = int(state.get("max_rounds") or 0)
    if rounds_done >= max_rounds:
        return "end"
    return "select"


def build_peer_graph():
    graph = StateGraph(PeerContinuationState)
    graph.add_node("select_speaker", _select_speaker)
    graph.add_node("generate_reply", _generate_reply)
    graph.add_node("check_stop", _check_stop)
    graph.set_entry_point("select_speaker")
    graph.add_conditional_edges(
        "select_speaker",
        _route_after_select,
        {"generate": "generate_reply", "stop": "check_stop"},
    )
    graph.add_edge("generate_reply", "check_stop")
    graph.add_conditional_edges(
        "check_stop",
        _route_after_stop,
        {"select": "select_speaker", "end": END},
    )
    return graph.compile()


def get_peer_graph():
    global _PEER_GRAPH
    if _PEER_GRAPH is None:
        _PEER_GRAPH = build_peer_graph()
    return _PEER_GRAPH


def run_peer_continuation(
    group,
    max_rounds: int | None = None,
    *,
    store=None,
) -> tuple[List[dict], str]:
    """Continue peer turns until no volunteers or safety cap; return (replies, stop_reason)."""
    global _ACTIVE_STORE
    from app.config import GROUP_PEER_MAX_ROUNDS
    from app.group.sessions import group_session_store

    cap = GROUP_PEER_MAX_ROUNDS if max_rounds is None else int(max_rounds)
    active = store or group_session_store
    active._sessions[group.session_id] = group
    observers = list(group.last_observers or [])
    eligible = eligible_speakers(
        group.profile_ids, group.last_may_speak, observers
    )
    last_id, _ = _last_student_utterance(group)
    initial: PeerContinuationState = {
        "session_id": group.session_id,
        "eligible": eligible,
        "observers": observers,
        "rounds_done": 0,
        "max_rounds": max(0, cap),
        "last_speaker": last_id,
        "selected": None,
        "replies": [],
        "stop_reason": "",
        "empty_selects": 0,
        "mode": group.last_mode or "discuss",
        "teacher_message": group.last_teacher_message or "",
        "speak_decisions": [],
    }
    if initial["max_rounds"] <= 0:
        return [], "max_rounds"
    if not eligible:
        return [], "no_eligible"

    _ACTIVE_STORE = active
    try:
        # Safety: LangGraph recursion limit should exceed the peer cap.
        result: Dict[str, Any] = get_peer_graph().invoke(
            initial,
            config={"recursion_limit": max(50, cap * 4 + 10)},
        )
    finally:
        _ACTIVE_STORE = None

    replies = list(result.get("replies") or [])
    stop_reason = result.get("stop_reason") or "max_rounds"
    logger.info(
        "peer_continuation session=%s rounds=%s stop=%s",
        group.session_id,
        len(replies),
        stop_reason,
    )
    return replies, stop_reason
