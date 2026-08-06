"""Track C — deterministic checks on group session exports (no LLM)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from eval.deterministic_checks import (
    _is_answer_leak,
    aggregate_deterministic,
)


def _teacher_message_for_turn(export: dict, turn: int) -> str:
    for entry in export.get("orchestration_log") or []:
        if int(entry.get("turn", -1)) == turn and entry.get("teacher_message"):
            return entry["teacher_message"]
    for entry in export.get("transcript") or []:
        if (
            int(entry.get("turn", -1)) == turn
            and entry.get("speaker_type") == "teacher"
        ):
            return entry.get("content") or ""
    return ""


def _student_replies_on_turn(export: dict, turn: int) -> List[dict]:
    return [
        e
        for e in (export.get("transcript") or [])
        if int(e.get("turn", -1)) == turn and e.get("speaker_type") == "student"
    ]


def check_observers_never_reply(export: dict, log_entry: dict) -> dict:
    """Students listed as observers must not appear in that turn's replies."""
    observers = set(log_entry.get("observers") or [])
    if not observers:
        return {
            "check": "observers_never_reply",
            "pass": True,
            "detail": "no observers",
            "turn": log_entry.get("turn"),
        }
    speakers = {
        e.get("speaker_id")
        for e in _student_replies_on_turn(export, int(log_entry.get("turn", -1)))
    }
    bad = sorted(observers & speakers)
    return {
        "check": "observers_never_reply",
        "pass": not bad,
        "detail": "ok" if not bad else f"observers spoke: {bad}",
        "turn": log_entry.get("turn"),
    }


def check_speakers_subset_of_eligible(export: dict, log_entry: dict) -> dict:
    """Actual speakers must be in must_speak ∪ may_speak when those are set."""
    must = set(log_entry.get("must_speak") or [])
    may = set(log_entry.get("may_speak") or [])
    eligible = must | may
    speakers = set(log_entry.get("speakers") or [])
    observers = set(log_entry.get("observers") or [])
    if not eligible:
        # Opener / incomplete log — only enforce observer disjointness
        bad = sorted(speakers & observers)
        return {
            "check": "speakers_subset_of_eligible",
            "pass": not bad,
            "detail": (
                "no eligible lists; observers disjoint"
                if not bad
                else f"speakers∩observers={bad}"
            ),
            "turn": log_entry.get("turn"),
        }
    bad = sorted(speakers - eligible)
    return {
        "check": "speakers_subset_of_eligible",
        "pass": not bad,
        "detail": "ok" if not bad else f"ineligible speakers: {bad}",
        "turn": log_entry.get("turn"),
    }


def check_must_speak_appeared(export: dict, log_entry: dict) -> dict:
    """Every must_speak student should have replied on that turn."""
    must = list(log_entry.get("must_speak") or [])
    if not must:
        return {
            "check": "must_speak_appeared",
            "pass": True,
            "detail": "no must_speak",
            "turn": log_entry.get("turn"),
        }
    replied = {
        e.get("speaker_id")
        for e in _student_replies_on_turn(export, int(log_entry.get("turn", -1)))
    }
    missing = [pid for pid in must if pid not in replied]
    return {
        "check": "must_speak_appeared",
        "pass": not missing,
        "detail": "ok" if not missing else f"missing must_speak: {missing}",
        "turn": log_entry.get("turn"),
    }


def check_at_least_one_speaker(export: dict, log_entry: dict) -> dict:
    speakers = log_entry.get("speakers") or []
    ok = len(speakers) >= 1
    return {
        "check": "at_least_one_speaker",
        "pass": ok,
        "detail": "ok" if ok else "empty speakers list",
        "turn": log_entry.get("turn"),
    }


def _early_wrong_student_turns(export: dict) -> List[dict]:
    """Build pseudo-turns for answer-leak checks from first student reply per profile."""
    task_text = export.get("task_text") or ""
    meta = export.get("task_metadata") or {}
    expected = (
        (export.get("task_resolution") or {}).get("expected_answer")
        or meta.get("expected_answer")
        or ""
    )
    students = export.get("students") or {}
    seen: set[str] = set()
    out: List[dict] = []
    for entry in export.get("transcript") or []:
        if entry.get("speaker_type") != "student":
            continue
        pid = entry.get("speaker_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        slot = students.get(pid) or {}
        out.append(
            {
                "profile_id": pid,
                "reply": entry.get("content") or "",
                "behavior_mode": slot.get("behavior_mode") or "",
                "turn": 1,
                "expected_answer": expected,
                "task_text": task_text,
                "task_id": meta.get("task_id") or "",
            }
        )
    return out


def check_group_early_answer_leak(export: dict) -> List[dict]:
    """WRONG/CONFUSED first replies should not leak expected_answer."""
    results = []
    for turn in _early_wrong_student_turns(export):
        mode = turn.get("behavior_mode") or ""
        if mode not in ("WRONG", "CONFUSED_HELPSEEKING"):
            results.append(
                {
                    "check": "group_early_answer_leak",
                    "pass": True,
                    "detail": f"{turn.get('profile_id')}: not WRONG/CONFUSED",
                    "profile_id": turn.get("profile_id"),
                }
            )
            continue
        leaked = _is_answer_leak(
            turn.get("reply") or "",
            turn.get("expected_answer") or "",
            turn.get("task_text") or "",
        )
        results.append(
            {
                "check": "group_early_answer_leak",
                "pass": not leaked,
                "detail": (
                    f"{turn.get('profile_id')}: leak"
                    if leaked
                    else f"{turn.get('profile_id')}: no leak"
                ),
                "profile_id": turn.get("profile_id"),
            }
        )
    return results


ORCH_CHECKS = (
    check_observers_never_reply,
    check_speakers_subset_of_eligible,
    check_must_speak_appeared,
    check_at_least_one_speaker,
)


def run_group_deterministic_checks(export: dict) -> List[dict]:
    """Run all group D/O checks; returns flat list of check dicts."""
    results: List[dict] = []
    for entry in export.get("orchestration_log") or []:
        for fn in ORCH_CHECKS:
            results.append(fn(export, entry))
    results.extend(check_group_early_answer_leak(export))
    return results


def score_group_export(export: dict) -> dict:
    """Aggregate pass rates for one group export."""
    results = run_group_deterministic_checks(export)
    # reuse aggregate shape by wrapping each result as a one-item "turn"
    summary = aggregate_deterministic([[r] for r in results])
    summary["n_checks"] = len(results)
    summary["session_id"] = export.get("session_id")
    return summary
