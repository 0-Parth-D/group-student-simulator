"""Load eval battery fixtures and map turns to dialogue roles.

Expected-behavior text itself lives in `app.student.expected_behavior`; this module only
supplies the teacher script and the role each scripted turn plays.
"""

from pathlib import Path
from typing import Dict, List

import yaml

from app.student.behavior_router import route_behavior
from app.knowledge.constructs import load_task_metadata
from app.student.data import make_student
from app.student.expected_behavior import DEFAULT_ROLE
from app.knowledge import store as kg_store

_FIXTURES_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _FIXTURES_DIR / "teacher_scripts.yaml"


def load_battery_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_bank_tasks() -> List[dict]:
    return load_task_metadata().get("tasks", [])


def get_teacher_follow_ups(task_id: str) -> List[dict]:
    cfg = load_battery_config()
    cues = cfg.get("scaffold_cues") or {}
    follow_ups = []
    for step in cfg.get("default_math", {}).get("follow_ups", []):
        step = dict(step)
        if step.get("message") is None:
            step["message"] = cues.get(task_id, "what should you do next")
        follow_ups.append(step)
    return follow_ups


def get_turn_roles() -> Dict[int, str]:
    """Turn index to dialogue role, including the opener the session generates."""
    cfg = load_battery_config()
    roles = {1: cfg.get("opening_role") or DEFAULT_ROLE}
    for step in cfg.get("default_math", {}).get("follow_ups", []):
        turn = step.get("turn")
        if turn is not None:
            roles[int(turn)] = step.get("role") or DEFAULT_ROLE
    return roles


def predict_initial_mode(profile_id: str, task_meta: dict) -> str:
    student = make_student(profile_id)
    mastery = kg_store.init_mastery_state(student.student_id, student.construct_mastery)
    target = route_behavior(mastery, task_meta, student)
    return target.mode
