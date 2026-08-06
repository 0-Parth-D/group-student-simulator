#!/usr/bin/env python3
"""Phase 0 CLI demo: teacher-facilitated 3-student group discussion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()

from app.group.sessions import (
    DEFAULT_GROUP_PROFILES,
    group_session_store,
)
from eval.group_script import build_group_script, mark_off_task_turns


DEMO_TASK_ID = "phone_plans_linear_01"


def _format_speaker(entry: dict, display_names: dict) -> str:
    if entry.get("speaker_type") == "teacher":
        return "Teacher"
    sid = entry.get("speaker_id", "")
    return display_names.get(sid) or sid.capitalize()


def print_transcript(group) -> None:
    print("\n" + "=" * 60)
    print("GROUP TRANSCRIPT")
    print("=" * 60)
    for entry in group.transcript:
        label = _format_speaker(entry, group.display_names)
        turn = entry.get("turn", "?")
        print(f"[turn {turn}] {label}: {entry.get('content', '')}")
        print()
    print("=" * 60)


def parse_profiles(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    return parts or list(DEFAULT_GROUP_PROFILES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a teacher-facilitated 3-student group demo (Model A)."
    )
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_GROUP_PROFILES),
        help="Comma-separated profile_ids (exactly 3). Default: jordan,sam,alex",
    )
    parser.add_argument(
        "--task-id",
        default=DEMO_TASK_ID,
        help=f"Bank task id (default: {DEMO_TASK_ID})",
    )
    parser.add_argument(
        "--task-text",
        default=None,
        help="Optional task text override",
    )
    parser.add_argument(
        "--no-off-task",
        action="store_true",
        help="Skip the off-task probe turn (keeps every turn on the math task)",
    )
    parser.add_argument(
        "--export-path",
        default=str(Path(__file__).resolve().parent / "results" / "group-demo-golden.json"),
        help="Path for the exported golden transcript JSON",
    )
    args = parser.parse_args()

    profile_ids = parse_profiles(args.profiles)
    print(
        f"Creating group session: profiles={profile_ids} task_id={args.task_id}"
    )
    group = group_session_store.create(
        profile_ids=profile_ids,
        task_id=args.task_id,
        task_text=args.task_text,
    )
    print(f"session_id={group.session_id}")
    print(f"task: {group.task_text}")
    for pid in group.profile_ids:
        slot = group.students[pid]
        mode = slot.behavior_profile.get("behavior_mode", "")
        print(
            f"  {group.display_names.get(pid, pid)} ({pid}): "
            f"mode={mode}"
        )

    print("\n--- START ---")
    group, replies, *_ = group_session_store.start(group)
    for r in replies:
        name = group.display_names.get(r["speaker_id"], r["speaker_id"])
        print(f"  {name}: {r['content'][:120]}...")

    script = build_group_script(
        [group.display_names.get(pid, pid) for pid in group.profile_ids],
        group.task_metadata,
        include_off_task=not args.no_off_task,
    )
    print(f"\nScript: {len(script)} teacher turns "
          f"({', '.join(sorted({s['label'] for s in script}))})")

    for step in script:
        print(f"\n--- TEACHER [{step['label']}]: {step['text']} ---")
        group, replies, *_ = group_session_store.respond(group, step["text"])
        for r in replies:
            name = group.display_names.get(r["speaker_id"], r["speaker_id"])
            print(f"  {name}: {r['content'][:160]}")

    print_transcript(group)
    export = group_session_store.export(group)
    mark_off_task_turns(export, script)
    export_path = Path(args.export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"Saved export: {export_path}")
    group_session_store.delete(group.session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
