"""Load PST phone-plans teacher facilitation script for replay evals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_FIXTURE_PATH = Path(__file__).resolve().parent / "pst_phone_plans_facilitation.yaml"


def load_pst_phone_plans_facilitation(path: Optional[Path] = None) -> dict:
    fixture_path = path or _FIXTURE_PATH
    with open(fixture_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def teacher_messages(fixture: Optional[dict] = None) -> List[str]:
    """Ordered teacher prompts for replay (verbatim from the session log)."""
    data = fixture or load_pst_phone_plans_facilitation()
    turns = data.get("teacher_turns")
    if turns:
        return list(turns)
    return [step["message"] for step in data.get("teacher_script") or [] if step.get("message")]


def teacher_script_steps(fixture: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Structured script with ids, labels, and facilitation notes."""
    data = fixture or load_pst_phone_plans_facilitation()
    steps = data.get("teacher_script")
    if steps:
        return [dict(step) for step in steps]
    return [
        {"turn": i + 1, "message": msg, "label": "unspecified"}
        for i, msg in enumerate(teacher_messages(data))
    ]
