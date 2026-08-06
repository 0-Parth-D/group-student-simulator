#!/usr/bin/env python3
"""Score facilitation gold with regex / llm / hybrid constraint parsers."""

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

from eval.fixtures.group_facilitation_gold import (
    format_score_report,
    score_parser_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parser",
        choices=("regex", "llm", "hybrid"),
        default="hybrid",
        help="Constraint parser to score (default: hybrid)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full summary JSON (includes per-case rows)",
    )
    args = parser.parse_args()
    summary = score_parser_baseline(parser_name=args.parser)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_score_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
