#!/usr/bin/env python3
"""Group battery: run the generic teacher script across many tasks and score each.

`run_group_demo` proves one conversation works. This sweeps the task bank the way
`run_battery` does for 1:1, so group behavior is measured as a property of the system
rather than of the phone-plan problem it was first tuned on.

A regression that only shows up on ratio tasks is invisible to a single-scenario demo.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()

from app.knowledge.constructs import load_task_metadata
from app.group.sessions import DEFAULT_GROUP_PROFILES, group_session_store
from eval.group_quality import score_group_quality
from eval.group_script import build_group_script, mark_off_task_turns

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def run_one(
    task_id: str,
    profile_ids: List[str],
    skip_llm: bool,
    include_off_task: bool,
    export_dir: Path,
) -> Dict:
    """Run and score a single group session. Failures are captured, not raised."""
    group = group_session_store.create(profile_ids=profile_ids, task_id=task_id)
    try:
        group, _replies, *_ = group_session_store.start(group)
        script = build_group_script(
            [group.display_names.get(pid, pid) for pid in group.profile_ids],
            group.task_metadata,
            include_off_task=include_off_task,
        )
        for step in script:
            group, _replies, *_ = group_session_store.respond(group, step["text"])

        export = group_session_store.export(group)
        mark_off_task_turns(export, script)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / f"{task_id}.json").write_text(
            json.dumps(export, indent=2), encoding="utf-8"
        )

        quality = score_group_quality(export, skip_llm=skip_llm)
        summary = quality.get("summary") or {}
        return {
            "task_id": task_id,
            "profile_ids": profile_ids,
            "n_teacher_turns": len(script),
            "script_labels": [s["label"] for s in script],
            "n_student_turns": quality.get("n_student_turns"),
            "n_off_task_turns": quality.get("n_off_task_turns"),
            "deterministic": (quality.get("deterministic") or {}).get(
                "overall_pass_rate"
            ),
            "cognitive_mean": summary.get("cognitive_mean"),
            "discourse_mean": summary.get("discourse_mean"),
            "believability_mean": summary.get("believability_mean"),
            "persona_mad_mean": summary.get("persona_mad_mean"),
            "tutor_like_rate": summary.get("tutor_like_rate"),
            "linguistic_proxies": (quality.get("linguistic_proxies") or {}).get("overall"),
            "n_flagged": len(quality.get("flagged") or []),
            "flagged": quality.get("flagged") or [],
        }
    except Exception as exc:  # one bad task must not abort the sweep
        return {"task_id": task_id, "profile_ids": profile_ids, "error": str(exc)}
    finally:
        group_session_store.delete(group.session_id)


def run_group_battery(
    task_ids: Optional[List[str]] = None,
    profile_ids: Optional[List[str]] = None,
    skip_llm: bool = False,
    include_off_task: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict:
    bank = [t["task_id"] for t in load_task_metadata().get("tasks", []) if t.get("task_id")]
    tasks = [t for t in (task_ids or bank) if t]
    profiles = list(profile_ids or DEFAULT_GROUP_PROFILES)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_root = output_dir or (RESULTS_ROOT / f"group-battery-{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for i, task_id in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task_id} ...", flush=True)
        result = run_one(
            task_id, profiles, skip_llm, include_off_task, out_root / "exports"
        )
        if result.get("error"):
            print(f"    ERROR: {result['error']}")
        else:
            print(
                f"    turns={result['n_student_turns']} "
                f"det={_fmt_pct(result['deterministic'])} "
                f"K={_fmt(result['cognitive_mean'])} "
                f"G={_fmt(result['discourse_mean'])} "
                f"B={_fmt(result['believability_mean'])}"
            )
        results.append(result)

    ok = [r for r in results if not r.get("error")]
    report = {
        "generated_at": stamp,
        "skip_llm": skip_llm,
        "include_off_task": include_off_task,
        "profile_ids": profiles,
        "task_ids": tasks,
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "summary": {
            "deterministic_mean": _mean([r["deterministic"] for r in ok if r.get("deterministic") is not None]),
            "cognitive_mean": _mean([r["cognitive_mean"] for r in ok if r.get("cognitive_mean") is not None]),
            "discourse_mean": _mean([r["discourse_mean"] for r in ok if r.get("discourse_mean") is not None]),
            "believability_mean": _mean([r["believability_mean"] for r in ok if r.get("believability_mean") is not None]),
            "persona_mad_mean": _mean([r["persona_mad_mean"] for r in ok if r.get("persona_mad_mean") is not None]),
        },
        "results": results,
    }

    report_path = out_root / "group_battery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote {report_path}")
    s = report["summary"]
    print(f"\n{len(ok)}/{len(results)} group sessions OK")
    print(f"  deterministic: {_fmt_pct(s['deterministic_mean'])}")
    if not skip_llm:
        print(f"  cognitive (K): {_fmt(s['cognitive_mean'])}   "
              f"discourse (G): {_fmt(s['discourse_mean'])}   "
              f"believability: {_fmt(s['believability_mean'])}")
    return report


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"


def _fmt_pct(v) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the group eval across many tasks")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids (default: all)")
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_GROUP_PROFILES),
        help="Comma-separated profile ids (exactly 3)",
    )
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--no-off-task",
        action="store_true",
        help="Skip the off-task probe turn in every session",
    )
    parser.add_argument("--output", default="", help="Output directory")
    args = parser.parse_args()

    report = run_group_battery(
        task_ids=[t.strip() for t in args.tasks.split(",") if t.strip()] or None,
        profile_ids=[p.strip() for p in args.profiles.split(",") if p.strip()] or None,
        skip_llm=args.skip_llm,
        include_off_task=not args.no_off_task,
        output_dir=Path(args.output) if args.output else None,
    )
    return 1 if report["n_ok"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
