"""Eval turn log schema and session export helpers."""

from typing import Dict, Optional

from app.eval_log import build_eval_turn_record

__all__ = ["build_eval_turn_record", "export_session_scenario"]


def export_session_scenario(
    session,
    profile_id: str,
    expected_behaviors: Optional[Dict[int, str]] = None,
) -> dict:
    """Export a ChatSession as an eval scenario wrapper."""
    task_id = (session.task_metadata or {}).get("task_id", "")
    scenario_id = f"{profile_id}__{task_id}" if task_id else profile_id
    turns = []
    for record in getattr(session, "eval_log", []) or []:
        turn_num = record.get("turn", 0)
        enriched = dict(record)
        if expected_behaviors and turn_num in expected_behaviors:
            enriched["expected_behavior"] = expected_behaviors[turn_num]
        turns.append(enriched)
    return {
        "scenario_id": scenario_id,
        "profile_id": profile_id,
        "task_id": task_id,
        "task_text": session.task_text,
        "turns": turns,
    }
