#!/usr/bin/env python3
"""Score a pst-training-game merged session export (integration.backend_export)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.believability import linguistic_proxies
from eval.group_deterministic_checks import score_group_export


def extract_backend_export(merged: dict) -> dict:
    integration = merged.get("integration") or {}
    export = integration.get("backend_export")
    if not export or not isinstance(export, dict):
        raise ValueError(
            "Missing integration.backend_export — export from an LP-enabled session "
            "(End Discussion before reflection export)."
        )
    if export.get("error"):
        raise ValueError(f"Backend export error embedded in file: {export['error']}")
    return export


def student_reply_texts(export: dict) -> list[str]:
    texts = []
    for entry in export.get("transcript") or []:
        if entry.get("speaker_type") == "student":
            content = (entry.get("content") or "").strip()
            if content:
                texts.append(content)
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_path", type=Path, help="Path to pst-session JSON from Reflection export")
    parser.add_argument("--json", action="store_true", help="Full JSON summary")
    args = parser.parse_args()

    merged = json.loads(args.merged_path.read_text(encoding="utf-8"))
    export = extract_backend_export(merged)

    deterministic = score_group_export(export)
    replies = student_reply_texts(export)
    voice = linguistic_proxies(replies)

    summary = {
        "source_file": str(args.merged_path),
        "backend_session_id": export.get("session_id"),
        "n_transcript_entries": len(export.get("transcript") or []),
        "n_orchestration_log_entries": len(export.get("orchestration_log") or []),
        "deterministic": deterministic,
        "linguistic_proxies": voice,
        "game_teacher_messages": sum(
            1 for m in (merged.get("conversation") or []) if m.get("role") == "teacher"
        ),
        "backend_turn_log_entries": len((merged.get("integration") or {}).get("backend_turn_log") or []),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"session: {summary['backend_session_id']}")
        print(f"deterministic pass rate: {deterministic['overall_pass_rate']:.1%}")
        print(f"student replies (backend transcript): {len(replies)}")
        print(f"hedge_rate: {voice['hedge_rate']:.2f}")
        print(f"game teacher msgs: {summary['game_teacher_messages']}")
        failed = deterministic.get("failed") or []
        if failed:
            print("failed checks:")
            for row in failed[:5]:
                print(f"  - {row.get('check')}: {row.get('detail')} (turn={row.get('turn')})")

    ok = deterministic.get("overall_pass_rate", 0) == 1.0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
