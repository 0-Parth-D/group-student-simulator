#!/usr/bin/env python3
"""Run Track C deterministic checks on a group session export JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.group_deterministic_checks import score_group_export


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path, help="Path to group export JSON")
    parser.add_argument("--json", action="store_true", help="Full JSON summary")
    args = parser.parse_args()
    export = json.loads(args.export_path.read_text(encoding="utf-8"))
    summary = score_group_export(export)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"session: {summary.get('session_id')}")
        print(f"checks: {summary.get('n_checks')}")
        print(f"overall_pass_rate: {summary['overall_pass_rate']:.1%}")
        print("per_check:")
        for name, rate in (summary.get("per_check") or {}).items():
            print(f"  {name}: {rate:.1%}")
        failed = summary.get("failed") or []
        if failed:
            print("failed:")
            for row in failed:
                print(f"  - {row.get('check')}: {row.get('detail')} (turn={row.get('turn')})")
    return 0 if summary["overall_pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
