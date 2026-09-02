#!/usr/bin/env python3
"""Replay PST phone-plans facilitation script and score discourse metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv()

from app.group_sessions import FE_STARTER_PROFILES, group_session_store
from eval.discourse_progress import score_export
from eval.fixtures.pst_phone_plans_facilitation import (
    load_pst_phone_plans_facilitation,
    teacher_messages,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "pst_phone_plans_facilitation.yaml"
)


def load_fixture(path: Path) -> dict:
    return load_pst_phone_plans_facilitation(path)


def run_replay(
    fixture: dict,
    *,
    use_llm: bool = True,
) -> dict:
    task_id = fixture.get("task_id") or "phone_plans_linear_01"
    profile_ids = fixture.get("profile_ids") or list(FE_STARTER_PROFILES)
    turns = teacher_messages(fixture)

    group = group_session_store.create(
        profile_ids=profile_ids,
        task_id=task_id,
    )
    group_session_store.start(group)

    for msg in turns:
        group, *_ = group_session_store.respond(group, msg)

    export = group_session_store.export(group)
    export["fight_phase"] = getattr(group, "fight_phase", "fight")
    export["crossover_unlocked"] = bool(getattr(group, "crossover_unlocked", False))
    export["breakthrough_bullets"] = list(
        getattr(group, "breakthrough_bullets", None) or []
    )
    # slot flags for scoring
    for pid in group.profile_ids:
        slot = group.students[pid]
        export.setdefault("slot_flags", {})[pid] = {
            "acknowledged_fee": getattr(slot, "acknowledged_fee", False),
            "crossover_intuition": getattr(slot, "crossover_intuition", False),
            "table_critique_peer": getattr(slot, "table_critique_peer", False),
            "fee_press_count": getattr(slot, "fee_press_count", 0),
            "stated_equations": getattr(slot, "stated_equations", False),
            "solved_crossover": getattr(slot, "solved_crossover", False),
        }

    metrics = score_export(export)
    group_session_store.delete(group.session_id)
    return {"export": export, "metrics": metrics, "use_llm": use_llm}


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay PST facilitation script.")
    parser.add_argument("--fixture", default=str(FIXTURE_PATH))
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results" / "pst_replay.json"),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Load existing export JSON and score only",
    )
    args = parser.parse_args()

    if args.score_only:
        data = json.loads(Path(args.out).read_text(encoding="utf-8"))
        export = data.get("export") or data
        metrics = score_export(export)
        print(json.dumps(metrics, indent=2))
        return 0

    fixture = load_fixture(Path(args.fixture))
    result = run_replay(fixture)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
