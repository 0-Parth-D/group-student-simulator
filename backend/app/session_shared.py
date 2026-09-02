"""Shared session helpers used by group_sessions (no 1:1 HTTP API in this repo)."""

from __future__ import annotations

from typing import List

from app.kg_config import MASTERY_LABELS, get_construct_def
from app.models import Student

TEACHER_OPENER_TEMPLATE = (
    "Here's the problem: {task}\n"
    "Take a look at it — what do you think the first step should be?"
)


def kg_summary(student: Student, limit: int = 12) -> List[dict]:
    """Compact construct mastery for profile API responses."""
    items = []
    for cid, level in list(student.construct_mastery.items())[:limit]:
        cdef = get_construct_def(cid) or {}
        items.append(
            {
                "concept": cdef.get("label", cid),
                "state": MASTERY_LABELS.get(level, "Unknown"),
                "construct_id": cid,
                "mastery_level": level,
            }
        )
    return items
