#!/usr/bin/env python3
"""Run control-based eval battery across profiles and bank tasks."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()

from eval.fixtures.battery_fixtures import (
    get_teacher_follow_ups,
    get_turn_roles,
    list_bank_tasks,
    load_battery_config,
)
from eval.log_schema import export_session_scenario
from eval.run_eval import evaluate_scenario
from app.student.expected_behavior import expected_behavior_for_record
from app.student.sessions import session_store


def run_scenario(
    profile_id: str,
    task_meta: dict,
    skip_llm: bool = False,
) -> dict:
    """Run one profile x task scenario; return scenario dict + eval report."""
    task_id = task_meta["task_id"]
    task_text = task_meta["description"]
    session = session_store.create(
        profile_id=profile_id,
        task_text=task_text,
        task_id=task_id,
    )
    expected_behaviors: Dict[int, str] = {}
    turn_roles = get_turn_roles()

    def annotate_turns() -> None:
        for turn_rec in session.eval_log:
            t = turn_rec["turn"]
            eb = expected_behavior_for_record(turn_rec, turn_roles.get(t))
            expected_behaviors[t] = eb
            turn_rec["expected_behavior"] = eb

    session, _, _, _ = session_store.start(session)
    annotate_turns()

    for step in get_teacher_follow_ups(task_id):
        msg = step["message"]
        session, _, _, _ = session_store.respond(session, msg)
        annotate_turns()

    scenario = export_session_scenario(session, profile_id, expected_behaviors)
    report = evaluate_scenario(scenario, skip_llm=skip_llm)
    session_store.delete(session.session_id)

    return {
        "scenario": scenario,
        "report": asdict(report),
        "initial_behavior_mode": scenario["turns"][0].get("behavior_mode") if scenario["turns"] else "",
    }


def run_battery(
    profiles: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    skip_llm: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    cfg = load_battery_config()
    profile_list = profiles or cfg.get("profiles") or ["alex", "jordan", "sam", "morgan"]
    tasks = list_bank_tasks()
    if task_ids:
        tasks = [t for t in tasks if t["task_id"] in task_ids]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_root = output_dir or Path(__file__).resolve().parent / "results" / ts
    scenarios_dir = out_root / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_flagged = []

    for profile_id in profile_list:
        for task_meta in tasks:
            scenario_id = f"{profile_id}__{task_meta['task_id']}"
            print(f"Running {scenario_id}...", flush=True)
            try:
                outcome = run_scenario(profile_id, task_meta, skip_llm=skip_llm)
            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)
                outcome = {
                    "scenario": {"scenario_id": scenario_id, "error": str(exc)},
                    "report": {},
                    "error": str(exc),
                }
            results.append(
                {
                    "scenario_id": scenario_id,
                    "profile_id": profile_id,
                    "task_id": task_meta["task_id"],
                    "initial_behavior_mode": outcome.get("initial_behavior_mode", ""),
                    "deterministic_pass_rate": outcome.get("report", {}).get(
                        "deterministic", {}
                    ).get("overall_pass_rate", 0.0),
                    "cognitive_fidelity": outcome.get("report", {}).get(
                        "cognitive_fidelity", 0.0
                    ),
                    "behavioral_consistency": outcome.get("report", {}).get(
                        "behavioral_consistency", 0.0
                    ),
                    "believability": outcome.get("report", {}).get("believability", 0.0),
                    "persona_mad": (
                        outcome.get("report", {})
                        .get("persona_stability", {})
                        .get("mean_absolute_deviation")
                    ),
                    "error": outcome.get("error"),
                }
            )
            if outcome.get("scenario"):
                scen_path = scenarios_dir / f"{scenario_id}.json"
                scen_path.write_text(
                    json.dumps(outcome["scenario"], indent=2),
                    encoding="utf-8",
                )
            for ft in outcome.get("report", {}).get("flagged_turns", []):
                all_flagged.append({"scenario_id": scenario_id, **ft})

    by_profile: Dict[str, List[float]] = {}
    for r in results:
        if r.get("error"):
            continue
        by_profile.setdefault(r["profile_id"], []).append(
            r.get("deterministic_pass_rate", 0.0)
        )

    baseline = {
        "timestamp": ts,
        "skip_llm": skip_llm,
        "scenario_count": len(results),
        "profiles": profile_list,
        "task_ids": [t["task_id"] for t in tasks],
        "results": results,
        "summary": {
            "mean_deterministic_pass_rate": (
                sum(r.get("deterministic_pass_rate", 0) for r in results if not r.get("error"))
                / max(1, sum(1 for r in results if not r.get("error")))
            ),
            "by_profile_mean_deterministic": {
                p: sum(v) / len(v) if v else 0.0 for p, v in by_profile.items()
            },
        },
    }

    report_path = out_root / "baseline_report.json"
    report_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    flagged_path = out_root / "flagged_turns.json"
    flagged_path.write_text(json.dumps(all_flagged, indent=2), encoding="utf-8")

    errors = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    print(f"\nWrote {report_path}")
    print(f"Wrote {flagged_path}")
    if errors:
        print(f"\n{len(errors)}/{len(results)} scenarios failed:")
        for r in errors[:5]:
            print(f"  - {r['scenario_id']}: {r['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more (see baseline_report.json)")
    if ok:
        mean_det = baseline["summary"]["mean_deterministic_pass_rate"]
        print(
            f"\n{len(ok)} scenarios OK — mean deterministic pass: {mean_det:.1%}"
        )
        if not skip_llm:
            mean_cog = sum(r.get("cognitive_fidelity", 0) for r in ok) / len(ok)
            mean_beh = sum(r.get("behavioral_consistency", 0) for r in ok) / len(ok)
            mean_bel = sum(r.get("believability", 0) for r in ok) / len(ok)
            print(
                f"  cognitive_fidelity: {mean_cog:.2f}  "
                f"behavioral_consistency: {mean_beh:.1%}  "
                f"believability: {mean_bel:.2f}"
            )
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Run eval battery")
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile ids (default: all)",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task ids (default: all bank tasks)",
    )
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--output",
        default="",
        help="Output directory (default: eval/results/<timestamp>)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Documented seed (logged only)")
    args = parser.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()] or None
    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()] or None
    out_dir = Path(args.output) if args.output else None

    baseline = run_battery(
        profiles=profiles,
        task_ids=task_ids,
        skip_llm=args.skip_llm,
        output_dir=out_dir,
    )
    baseline["seed"] = args.seed
    if out_dir:
        (out_dir / "baseline_report.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )
    failed = sum(1 for r in baseline.get("results", []) if r.get("error"))
    if failed == len(baseline.get("results", [])):
        sys.exit(1)


if __name__ == "__main__":
    main()
