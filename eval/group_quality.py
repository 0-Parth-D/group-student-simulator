"""Track C — group conversation quality: K / P / G (+ optional D/O)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.student.expected_behavior import (
    NON_MATH_LABELS,
    expected_behavior_for_record,
    role_for_turn,
)
from eval.believability import believability, linguistic_proxies
from eval.cognitive_fidelity import cognitive_fidelity
from eval.group_deterministic_checks import score_group_export
from eval.group_discourse import group_discourse_fidelity
from eval.persona_stability import persona_stability

_PEER_INCOMING = re.compile(r"^\[[^\]]+\]:")


def _is_peer_turn(record: dict) -> bool:
    msg = (record.get("teacher_message") or "").lstrip()
    return bool(_PEER_INCOMING.match(msg))


def is_off_task(record: dict) -> bool:
    """Off-task by the runner's declared intent, or by the classifier's label."""
    return bool(record.get("is_off_task")) or (
        record.get("teacher_label") or ""
    ) in NON_MATH_LABELS


def collect_student_turns(export: dict) -> List[dict]:
    """Prefer per-slot eval_log; fall back to transcript + student snapshots."""
    turns: List[dict] = []
    students = export.get("students") or {}
    task_text = export.get("task_text") or ""
    meta = export.get("task_metadata") or {}
    expected = (
        (export.get("task_resolution") or {}).get("expected_answer")
        or meta.get("expected_answer")
        or ""
    )

    for pid, slot in students.items():
        seen_first = False
        for record in slot.get("eval_log") or []:
            enriched = dict(record)
            enriched.setdefault("profile_id", pid)
            enriched.setdefault("task_text", task_text)
            if expected and not enriched.get("expected_answer"):
                enriched["expected_answer"] = expected
            mode = enriched.get("behavior_mode") or slot.get("behavior_mode") or ""
            enriched["behavior_mode"] = mode
            enriched["is_peer_turn"] = _is_peer_turn(enriched)
            if not enriched.get("expected_behavior"):
                role = (
                    "off_task"
                    if is_off_task(enriched)
                    else role_for_turn(
                        enriched.get("teacher_label") or "",
                        is_peer_turn=enriched["is_peer_turn"],
                        is_first_turn=not seen_first,
                    )
                )
                enriched["expected_behavior"] = expected_behavior_for_record(
                    enriched, role
                )
            seen_first = True
            turns.append(enriched)

    if turns:
        return sorted(turns, key=lambda t: (int(t.get("turn", 0)), t.get("profile_id", "")))

    # Fallback: transcript-only exports
    primary = (meta.get("required_constructs") or [""])[0]
    for entry in export.get("transcript") or []:
        if entry.get("speaker_type") != "student":
            continue
        pid = entry.get("speaker_id") or ""
        slot = students.get(pid) or {}
        mode = slot.get("behavior_mode") or ""
        turns.append(
            {
                "turn": entry.get("turn"),
                "role": "student",
                "type": "math",
                "profile_id": pid,
                "reply": entry.get("content") or "",
                "behavior_mode": mode,
                "teacher_label": "math_scaffold",
                "expected_answer": expected,
                "task_text": task_text,
                "task_id": meta.get("task_id") or "",
                "construct_id": primary,
                "primary_construct": primary,
                "mastery": (slot.get("mastery_state") or {}).get(primary, 1),
                "is_peer_turn": False,
            }
        )
    # Mark peer turns using same-turn prior student
    by_turn: Dict[int, List[dict]] = {}
    for t in turns:
        by_turn.setdefault(int(t.get("turn") or 0), []).append(t)
    for turn_num, group in by_turn.items():
        for i, t in enumerate(group):
            if i > 0:
                t["is_peer_turn"] = True

    # Expected behavior needs is_peer_turn, so it is filled after peer marking.
    seen: set = set()
    for t in turns:
        pid = t.get("profile_id") or ""
        t["expected_behavior"] = expected_behavior_for_record(
            t,
            role_for_turn(
                t.get("teacher_label") or "",
                is_peer_turn=bool(t.get("is_peer_turn")),
                is_first_turn=pid not in seen,
            ),
        )
        seen.add(pid)
    return turns


def _transcript_window(export: dict, turn: int, limit: int = 8) -> List[str]:
    lines = []
    for entry in export.get("transcript") or []:
        if int(entry.get("turn") or 0) > turn:
            break
        st = entry.get("speaker_type")
        sid = entry.get("speaker_id") or ""
        label = "Teacher" if st == "teacher" else sid
        content = (entry.get("content") or "").strip()
        if content:
            lines.append(f"{label}: {content[:180]}")
    return lines[-limit:]


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def score_group_quality(
    export: dict,
    *,
    skip_llm: bool = False,
    include_deterministic: bool = True,
) -> Dict[str, Any]:
    """Score group authenticity: D/O + per-speaker K/P + peer G."""
    turns = collect_student_turns(export)
    report: Dict[str, Any] = {
        "session_id": export.get("session_id"),
        "profile_ids": list(export.get("profile_ids") or []),
        "n_student_turns": len(turns),
        "n_off_task_turns": sum(1 for t in turns if is_off_task(t)),
        "skip_llm": skip_llm,
        "per_profile": {},
        "cognitive_scores": [],
        "persona_scores": {},
        "discourse_scores": [],
        "believability_scores": [],
        "flagged": [],
        "summary": {},
    }

    if include_deterministic:
        report["deterministic"] = score_group_export(export)

    # Voice statistics need no LLM, so they are available even with judges skipped.
    report["linguistic_proxies"] = {
        "overall": linguistic_proxies([t.get("reply") or "" for t in turns]),
        "per_profile": {
            pid: linguistic_proxies(
                [t.get("reply") or "" for t in turns if t.get("profile_id") == pid]
            )
            for pid in sorted({t.get("profile_id") for t in turns if t.get("profile_id")})
        },
    }

    if skip_llm:
        report["summary"] = {
            "cognitive_mean": None,
            "persona_mad_mean": None,
            "discourse_mean": None,
            "believability_mean": None,
            "tutor_like_rate": None,
            "note": "LLM judges skipped; deterministic and linguistic proxies only",
        }
        return report

    # --- K: cognitive fidelity per turn ---
    # Off-task turns are skipped: "did this match the intended knowledge state" has no
    # answer when the teacher asked about the assembly. Scoring them punishes students
    # for correctly following the teacher off the math.
    cog_by_profile: Dict[str, List[float]] = {}
    for turn in turns:
        if is_off_task(turn):
            continue
        score = cognitive_fidelity(turn)
        score["turn"] = turn.get("turn")
        score["profile_id"] = turn.get("profile_id")
        report["cognitive_scores"].append(score)
        pid = turn.get("profile_id") or "?"
        cog_by_profile.setdefault(pid, []).append(float(score.get("score") or 0))
        if float(score.get("score") or 0) < 3:
            report["flagged"].append(
                {
                    "layer": "K",
                    "profile_id": pid,
                    "turn": turn.get("turn"),
                    "score": score.get("score"),
                    "detail": score.get("violated_constraint"),
                }
            )

    # --- P: persona stability per profile ---
    persona_mads: List[float] = []
    for pid in export.get("profile_ids") or sorted(
        {t.get("profile_id") for t in turns if t.get("profile_id")}
    ):
        sampled = [
            {
                "role": "student",
                "content": t.get("reply") or "",
                "profile_id": pid,
            }
            for t in turns
            if t.get("profile_id") == pid and (t.get("reply") or "").strip()
        ]
        if not sampled:
            continue
        persona = persona_stability(sampled, profile_id=pid)
        report["persona_scores"][pid] = persona
        mad = persona.get("mean_absolute_deviation")
        if isinstance(mad, (int, float)):
            persona_mads.append(float(mad))
            if float(mad) > 2.0:
                report["flagged"].append(
                    {
                        "layer": "P",
                        "profile_id": pid,
                        "mad": mad,
                        "detail": "high persona MAD vs demo Big Five",
                    }
                )

    # --- G: peer discourse on peer turns (fallback: all student turns if none) ---
    peer_turns = [t for t in turns if t.get("is_peer_turn")]
    discourse_targets = peer_turns or turns
    disc_by_profile: Dict[str, List[float]] = {}
    tutor_flags = 0
    for turn in discourse_targets:
        pid = turn.get("profile_id") or "?"
        gscore = group_discourse_fidelity(
            reply=turn.get("reply") or "",
            profile_id=pid,
            behavior_mode=turn.get("behavior_mode") or "",
            transcript_window=_transcript_window(
                export, int(turn.get("turn") or 0)
            ),
        )
        gscore["turn"] = turn.get("turn")
        gscore["is_peer_turn"] = bool(turn.get("is_peer_turn"))
        report["discourse_scores"].append(gscore)
        disc_by_profile.setdefault(pid, []).append(float(gscore.get("score") or 0))
        if gscore.get("tutor_like"):
            tutor_flags += 1
            report["flagged"].append(
                {
                    "layer": "G",
                    "profile_id": pid,
                    "turn": turn.get("turn"),
                    "detail": gscore.get("rationale") or "tutor_like",
                }
            )

    # --- V: believability (middle-school voice) per turn ---
    bel_by_profile: Dict[str, List[float]] = {}
    for turn in turns:
        vscore = believability(turn)
        vscore["turn"] = turn.get("turn")
        vscore["profile_id"] = turn.get("profile_id")
        report["believability_scores"].append(vscore)
        pid = turn.get("profile_id") or "?"
        bel_by_profile.setdefault(pid, []).append(float(vscore.get("score") or 0))
        if float(vscore.get("score") or 0) < 3:
            report["flagged"].append(
                {
                    "layer": "V",
                    "profile_id": pid,
                    "turn": turn.get("turn"),
                    "score": vscore.get("score"),
                    "detail": vscore.get("tell"),
                }
            )

    # --- per-profile rollup ---
    for pid in export.get("profile_ids") or []:
        report["per_profile"][pid] = {
            "cognitive_mean": _mean(cog_by_profile.get(pid, [])),
            "persona_mad": (report["persona_scores"].get(pid) or {}).get(
                "mean_absolute_deviation"
            ),
            "discourse_mean": _mean(disc_by_profile.get(pid, [])),
            "believability_mean": _mean(bel_by_profile.get(pid, [])),
            "n_turns": sum(1 for t in turns if t.get("profile_id") == pid),
        }

    n_disc = len(report["discourse_scores"]) or 1
    all_cog = [float(s.get("score") or 0) for s in report["cognitive_scores"]]
    report["summary"] = {
        "cognitive_mean": _mean(all_cog),
        "persona_mad_mean": _mean(persona_mads),
        "discourse_mean": _mean(
            [float(s.get("score") or 0) for s in report["discourse_scores"]]
        ),
        "believability_mean": _mean(
            [float(s.get("score") or 0) for s in report["believability_scores"]]
        ),
        "tutor_like_rate": tutor_flags / n_disc,
        "deterministic_pass_rate": (report.get("deterministic") or {}).get(
            "overall_pass_rate"
        ),
        "n_flagged": len(report["flagged"]),
    }
    return report


def format_group_quality_report(report: Dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    proxies = (report.get("linguistic_proxies") or {}).get("overall") or {}
    lines = [
        f"Track C — group quality ({report.get('session_id')})",
        f"  student turns: {report.get('n_student_turns')}",
        f"  K cognitive mean: {summary.get('cognitive_mean')}",
        f"  P persona MAD mean: {summary.get('persona_mad_mean')} (lower better)",
        f"  G discourse mean: {summary.get('discourse_mean')}",
        f"  G tutor-like rate: {summary.get('tutor_like_rate')}",
        f"  V believability mean: {summary.get('believability_mean')}",
        f"  V voice: FK grade={proxies.get('flesch_kincaid_grade')} "
        f"hedge={proxies.get('hedge_rate')} "
        f"long_turns={proxies.get('long_turn_rate')}",
        f"  D/O pass rate: {summary.get('deterministic_pass_rate')}",
        f"  flagged: {summary.get('n_flagged')}",
        "  per profile:",
    ]
    for pid, stats in (report.get("per_profile") or {}).items():
        lines.append(
            f"    {pid}: K={stats.get('cognitive_mean')} "
            f"P_MAD={stats.get('persona_mad')} "
            f"G={stats.get('discourse_mean')} "
            f"V={stats.get('believability_mean')} "
            f"turns={stats.get('n_turns')}"
        )
    flagged = report.get("flagged") or []
    if flagged:
        lines.append("  flags:")
        for item in flagged[:12]:
            lines.append(
                f"    - [{item.get('layer')}] {item.get('profile_id')} "
                f"turn={item.get('turn')} {item.get('detail')}"
            )
    return "\n".join(lines)
