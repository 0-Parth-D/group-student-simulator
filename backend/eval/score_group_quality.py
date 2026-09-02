#!/usr/bin/env python3
"""Score Track C group conversation quality (K/P/G + D/O) on an export JSON."""

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

from eval.group_quality import format_group_quality_report, score_group_quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path, help="Path to group export JSON")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only run deterministic D/O checks (no K/P/G judges)",
    )
    parser.add_argument("--json", action="store_true", help="Full JSON report")
    args = parser.parse_args()
    export = json.loads(args.export_path.read_text(encoding="utf-8"))
    report = score_group_quality(export, skip_llm=args.skip_llm)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_group_quality_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
